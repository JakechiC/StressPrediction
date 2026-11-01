"""Planar construction used by the graphene design example (no relaxation)."""

from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass(frozen=True)
class GeometryConfig:
    orientation: str = "armchair"
    bond_length: float = 1.42
    n_armchair: int = 1
    n_zigzag: int = 2
    n_x_grid: int = 5
    n_y_grid: int = 5
    buffer_cells: int = 2
    vacuum: float = 20.0

    def __post_init__(self):
        if self.orientation not in {"armchair", "zigzag"}:
            raise ValueError("orientation must be armchair or zigzag")
        for name in ("n_armchair", "n_zigzag", "n_x_grid", "n_y_grid"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.buffer_cells, int) or self.buffer_cells < 1:
            raise ValueError("buffer_cells must be a positive integer")
        if not np.isfinite(self.bond_length) or self.bond_length <= 0:
            raise ValueError("bond_length must be finite and positive")
        if not np.isfinite(self.vacuum) or self.vacuum <= 0:
            raise ValueError("vacuum must be finite and positive")


def build_graphene(config: GeometryConfig) -> tuple[Atoms, list[int]]:
    """Build the original rectangular C4 lattice and ordered operable sites.

    The default 9 x 14 repetitions contain 504 atoms and 25 sites. Coordinates
    remain in z=0, with periodicity only in x/y. No calculator is attached.
    """
    cc = config.bond_length
    if config.orientation == "armchair":
        cell = (3 * cc, np.sqrt(3) * cc, config.vacuum)
        positions = [[1 / 6, 1 / 4, 0], [1 / 3, 3 / 4, 0],
                     [2 / 3, 3 / 4, 0], [5 / 6, 1 / 4, 0]]
    else:
        cell = (np.sqrt(3) * cc, 3 * cc, config.vacuum)
        positions = [[0, 1 / 6, 0], [0, 5 / 6, 0],
                     [1 / 2, 1 / 3, 0], [1 / 2, 2 / 3, 0]]
    unit = Atoms("C4", scaled_positions=positions, cell=cell, pbc=(True, True, False))
    nx = config.n_x_grid * config.n_armchair + 2 * config.buffer_cells
    ny = config.n_y_grid * config.n_zigzag + 2 * config.buffer_cells
    atoms = unit.repeat((nx, ny, 1))
    atoms.set_tags(0)
    atoms.set_array("original_index", np.arange(len(atoms), dtype=int))
    sites = []
    for gx in range(config.n_x_grid):
        for gy in range(config.n_y_grid):
            ix = config.buffer_cells + gx * config.n_armchair
            iy = config.buffer_cells + gy * config.n_zigzag
            target = unit.positions[0] + ix * unit.cell[0] + iy * unit.cell[1]
            index = int(np.argmin(np.linalg.norm(atoms.positions - target, axis=1)))
            sites.append(index)
            atoms[index].tag = 1
    if len(set(sites)) != len(sites):
        raise ValueError("Geometry produces duplicate operable sites")
    return atoms, sites


def ordered_neighbors(atoms: Atoms, site: int) -> np.ndarray:
    """Find three nearest neighbors and order by in-plane azimuth."""
    distances = atoms.get_distances(site, range(len(atoms)), mic=True)
    indices = np.argsort(distances)[1:4]
    vectors = atoms.get_distances(site, indices, mic=True, vector=True)
    return indices[np.argsort(np.arctan2(vectors[:, 1], vectors[:, 0]))]


def apply_actions(pristine: Atoms, sites: list[int], actions) -> Atoms:
    """Apply rotations from pristine coordinates, then delete vacancies.

    Actions 0/1/2 rotate a bond by 90 degrees; 3 removes the site atom; 4 leaves
    it unchanged. This preserves the source construction rule, including the
    use of pristine coordinates for each rotation. Counts describe operations,
    not a post-relaxation topological analysis of defects.
    """
    actions = list(actions)
    if len(actions) > len(sites):
        raise ValueError("More actions than sites")
    result = pristine.copy()
    remove = []
    for site, action in zip(sites, actions):
        if action not in range(5):
            raise ValueError(f"Invalid action {action}")
        if action < 3:
            neighbor = int(ordered_neighbors(pristine, site)[action])
            first = pristine.positions[site]
            displacement = pristine.get_distance(site, neighbor, mic=True, vector=True)
            second = first + displacement
            midpoint = (first + second) / 2
            vector = first - midpoint
            rotated = np.array([-vector[1], vector[0], 0.0])
            result.positions[site] = midpoint + rotated
            result.positions[neighbor] = midpoint - rotated
            result[site].tag = 2
            result[neighbor].tag = 2
        elif action == 3:
            remove.append(site)
    for site in sorted(remove, reverse=True):
        del result[site]
    result.info["relaxation"] = "none; planar geometric construction"
    return result
