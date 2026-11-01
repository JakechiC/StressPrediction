"""Constrained design states and terminal sparse rewards in eV."""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .geometry import GeometryConfig, apply_actions, build_graphene

EV_TO_GPA_A3 = 160.21766208
ACTION_NAMES = ("rotate_0", "rotate_1", "rotate_2", "vacancy", "no_change")


def threshold_in_ev(spec: dict) -> float:
    """Require a value and explicit virial unit; never infer units from a number."""
    if not isinstance(spec, dict) or set(spec) != {"value", "unit"}:
        raise ValueError("reward.threshold requires exactly value and unit")
    if isinstance(spec["value"], bool) or not isinstance(spec["value"], (int, float)):
        raise ValueError("reward.threshold.value must be an explicit number")
    value = float(spec["value"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("reward.threshold.value must be finite and positive")
    if spec["unit"] == "eV":
        return value
    if spec["unit"] in {"GPa*A^3", "GPa·Å³"}:
        return value / EV_TO_GPA_A3
    raise ValueError("reward.threshold.unit must be eV or GPa*A^3 (not GPa)")


def hotspot_score(virial_ev: np.ndarray, top_k: int) -> tuple[float, np.ndarray]:
    """Return mean top-k full-matrix Frobenius norms and all per-atom norms."""
    virial_ev = np.asarray(virial_ev, dtype=float)
    if virial_ev.ndim != 3 or virial_ev.shape[1:] != (3, 3) or len(virial_ev) == 0:
        raise ValueError("virial must have shape [N,3,3] with N > 0")
    if not np.isfinite(virial_ev).all():
        raise ValueError("virial contains non-finite values")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k_atoms must be a positive integer")
    magnitudes = np.linalg.norm(virial_ev, axis=(1, 2))
    k = min(top_k, len(magnitudes))
    return float(np.partition(magnitudes, len(magnitudes) - k)[-k:].mean()), magnitudes


@dataclass(frozen=True)
class Constraints:
    rotations: int = 8
    vacancies: int = 9
    total_steps: int = 25

    def __post_init__(self):
        for name in ("rotations", "vacancies", "total_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.total_steps == 0 or self.rotations + self.vacancies > self.total_steps:
            raise ValueError("Defect counts cannot be met within the operable sites")

    def mask(self, step: int, rotations: int, vacancies: int) -> np.ndarray:
        """Forward-looking exact-count mask; an infeasible state raises an error."""
        if not 0 <= step <= self.total_steps or min(rotations, vacancies) < 0:
            raise ValueError("Invalid constraint state")
        if rotations + vacancies > step:
            raise ValueError("Operation counts exceed elapsed steps")
        if rotations > self.rotations or vacancies > self.vacancies:
            raise ValueError("Operation counts exceed constraints")
        if step == self.total_steps:
            if (rotations, vacancies) != (self.rotations, self.vacancies):
                raise ValueError("Terminal counts do not satisfy constraints")
            return np.zeros(5, dtype=bool)
        remaining = self.total_steps - step - 1
        valid = []
        for action in range(5):
            rot = rotations + int(action < 3)
            vac = vacancies + int(action == 3)
            required = max(0, self.rotations - rot) + max(0, self.vacancies - vac)
            valid.append(rot <= self.rotations and vac <= self.vacancies and required <= remaining)
        if not any(valid):
            raise ValueError("No feasible action; constraint state is inconsistent")
        return np.asarray(valid, dtype=bool)


class GrapheneEnv:
    """Sequential construction with inference only at the terminal step.

    predictor(atoms) must return atomic energies [N] and virials [N,3,3], both
    denormalized in eV. The predictor owns graph building and autograd settings.
    """

    def __init__(self, predictor: Callable, geometry: GeometryConfig, constraints: Constraints,
                 top_k_atoms: int, threshold_ev: float, reward_scale: float = 1.0):
        self.predictor = predictor
        self.geometry = geometry
        self.constraints = constraints
        if geometry.n_x_grid * geometry.n_y_grid != constraints.total_steps:
            raise ValueError("Geometry and constraint step counts differ")
        if isinstance(top_k_atoms, bool) or not isinstance(top_k_atoms, int) or top_k_atoms < 1:
            raise ValueError("top_k_atoms must be a positive integer")
        if not np.isfinite(threshold_ev) or threshold_ev <= 0:
            raise ValueError("threshold_ev must be finite and positive")
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        self.top_k_atoms = top_k_atoms
        self.threshold_ev = float(threshold_ev)
        self.reward_scale = float(reward_scale)
        self.pristine, self.sites = build_graphene(geometry)
        self.reset()

    @property
    def state_dim(self):
        return 2 * self.constraints.total_steps + 2

    @property
    def done(self):
        return len(self.actions) == self.constraints.total_steps

    def reset(self):
        self.actions = []
        self.rotations = self.vacancies = 0
        self.last_prediction = None
        return self.state()

    def state(self):
        n = self.constraints.total_steps
        canvas = np.full(n, 4, dtype=np.float32)
        canvas[:len(self.actions)] = self.actions
        position = np.zeros(n, dtype=np.float32)
        if not self.done:
            position[len(self.actions)] = 1
        counts = np.asarray([self.rotations / max(1, self.constraints.rotations),
                             self.vacancies / max(1, self.constraints.vacancies)], dtype=np.float32)
        return np.concatenate([canvas, position, counts])

    def action_mask(self):
        return self.constraints.mask(len(self.actions), self.rotations, self.vacancies)

    def structure(self):
        return apply_actions(self.pristine, self.sites, self.actions)

    def step(self, action: int):
        if self.done:
            raise ValueError("Episode is complete; call reset before step")
        if action not in range(5) or not self.action_mask()[action]:
            raise ValueError(f"Action {action} violates the remaining count constraints")
        self.actions.append(int(action))
        self.rotations += int(action < 3)
        self.vacancies += int(action == 3)
        reward = 0.0
        info = {"step": len(self.actions), "rotations": self.rotations, "vacancies": self.vacancies}
        if self.done:
            atoms = self.structure()
            energies, virial = self.predictor(atoms)
            energies = np.asarray(energies, dtype=float)
            virial = np.asarray(virial, dtype=float)
            if energies.shape != (len(atoms),) or not np.isfinite(energies).all():
                raise ValueError("Predictor must return finite energies with shape [N]")
            if virial.shape != (len(atoms), 3, 3):
                raise ValueError("Predictor virial shape does not match candidate atoms")
            score, magnitudes = hotspot_score(virial, self.top_k_atoms)
            self.last_prediction = (atoms, energies, virial, magnitudes)
            reward = self.reward_scale * max(0.0, self.threshold_ev - score)
            info.update(hotspot_ev=score, top_k_atoms=min(self.top_k_atoms, len(atoms)))
        return self.state(), reward, self.done, info
