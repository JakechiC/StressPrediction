"""Explicit training and candidate generation; functions run only when called."""

from dataclasses import asdict
import csv
from pathlib import Path
import random
import time

import numpy as np
import torch

from .agent import AgentConfig, DQNAgent, read_policy_checkpoint
from .environment import Constraints, GrapheneEnv, threshold_in_ev
from .geometry import GeometryConfig
from .io import CandidateWriter, new_output, positive_integer, sha256_file, write_json


def make_environment(config, predictor=None):
    model = config["model"]
    if predictor is None:
        from stress_prediction.checkpoints import load_predictor
        predictor = load_predictor(model["checkpoint"], device=model["device"],
                                   allow_legacy_pickle=model["allow_legacy_pickle"])
        model["checkpoint_sha256"] = sha256_file(model["checkpoint"])
        model["predictor_source"] = "checkpoint"
    else:
        model["checkpoint_sha256"] = None
        model["predictor_source"] = "injected_callable"
    env = config["environment"]
    geometry = GeometryConfig(**env["geometry"])
    constraints = Constraints(env["rotations"], env["vacancies"], geometry.n_x_grid * geometry.n_y_grid)
    return GrapheneEnv(predictor, geometry, constraints, env["top_k_atoms"],
                       threshold_in_ev(env["reward"]["threshold"]), env["reward"]["scale"])


def seed_runtime(seed, device):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def agent_config(config):
    try:
        result = AgentConfig(**config["agent"])
    except TypeError as exc:
        raise ValueError(f"Invalid agent configuration: {exc}") from exc
    config["agent"] = asdict(result)
    return result


def save_checkpoint(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def run_training(config, *, resume=None, predictor=None):
    """Run one-step masked DQN; this is the only command that optimizes a policy."""
    settings = config["training"]
    settings.setdefault("checkpoint_interval", 100)
    settings.setdefault("seed", 42)
    episodes = positive_integer(settings.get("episodes"), "training.episodes")
    interval = positive_integer(settings.get("checkpoint_interval", 100), "training.checkpoint_interval")
    seed = settings.get("seed", 42)
    device = config["model"]["device"]
    seed_runtime(seed, device)
    network_config = agent_config(config)
    env = make_environment(config, predictor)
    agent = DQNAgent(env.state_dim, network_config, device, seed)
    best_score = None
    checkpoint = None
    if resume is not None:
        checkpoint = read_policy_checkpoint(Path(resume), device, config["model"]["allow_legacy_pickle"])
        saved = checkpoint.get("effective_config", {})
        for section in ("environment", "agent"):
            if saved.get(section) != config[section]:
                raise ValueError(f"Resume requires identical {section} configuration")
        for key in ("checkpoint_sha256", "predictor_source"):
            if saved.get("model", {}).get(key) != config["model"].get(key):
                raise ValueError("Resume surrogate checkpoint contents differ from the saved run")
        agent.restore_training(checkpoint)
        best_score = checkpoint.get("best_training_hotspot_ev")
        if agent.completed_episodes >= episodes:
            raise ValueError("training.episodes must exceed checkpoint completed_episodes")
    output = new_output(settings["output_dir"])
    write_json(output / "effective_config.json", config)
    start_episode = agent.completed_episodes
    started = time.monotonic()
    last_checkpoint_episode = start_episode
    best_policy = checkpoint.get("best_policy") if checkpoint is not None else None

    def persist_training():
        state = agent.training_state()
        state.update(effective_config=config, best_training_hotspot_ev=best_score,
                     best_policy=best_policy,
                     resume_source=str(resume) if resume else None)
        save_checkpoint(output / "training_latest.pt", state)

    # Always retain a consistent complete-episode checkpoint, including episode 0.
    persist_training()
    if checkpoint is not None:
        inherited = checkpoint.get("best_policy")
        if inherited is not None:
            save_checkpoint(output / "policy_best.pt", inherited)
    status = "running"
    try:
        with (output / "training.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["completed_episodes", "frames", "reward", "hotspot_ev",
                                                        "loss", "epsilon", "rotations", "vacancies"])
            writer.writeheader()
            for _ in range(start_episode, episodes):
                state = env.reset()
                losses = []
                reward = 0.0
                while not env.done:
                    action = agent.select(state, env.action_mask(), training=True)
                    next_state, reward, done, info = env.step(action)
                    agent.observe(state, action, reward, next_state, done, env.action_mask())
                    loss = agent.update()
                    if loss is not None:
                        losses.append(loss)
                    state = next_state
                agent.completed_episodes += 1
                score = info["hotspot_ev"]
                if best_score is None or score < best_score:
                    best_score = score
                    best_policy = agent.policy_state()
                    # Clone tensors; later training must not mutate the retained best snapshot.
                    best_policy["policy_net_state_dict"] = {
                        key: value.detach().cpu().clone() for key, value in best_policy["policy_net_state_dict"].items()
                    }
                    best_policy.update(effective_config=config, selection_rule="lowest observed terminal training hotspot",
                                       training_hotspot_ev=score, generating_actions=list(env.actions))
                    save_checkpoint(output / "policy_best.pt", best_policy)
                writer.writerow({"completed_episodes": agent.completed_episodes, "frames": agent.frames,
                                 "reward": reward, "hotspot_ev": score,
                                 "loss": float(np.mean(losses)) if losses else "",
                                 "epsilon": agent.epsilon, "rotations": env.rotations, "vacancies": env.vacancies})
                handle.flush()
                if agent.completed_episodes % interval == 0 or agent.completed_episodes == episodes:
                    state = agent.training_state()
                    state.update(effective_config=config, best_training_hotspot_ev=best_score,
                                 best_policy=best_policy, resume_source=str(resume) if resume else None)
                    save_checkpoint(output / "training_latest.pt", state)
                    last_checkpoint_episode = agent.completed_episodes
                    print(f"episode={agent.completed_episodes} frames={agent.frames} H_eV={score:.6g} epsilon={agent.epsilon:.4g}")
            final_policy = agent.policy_state()
            final_policy.update(effective_config=config, selection_rule="final completed episode")
            save_checkpoint(output / "policy_final.pt", final_policy)
            status = "complete"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except Exception:
        status = "failed"
        raise
    finally:
        write_json(output / "run_summary.json", {
            "status": status, "requested_total_episodes": episodes, "initial_completed_episodes": start_episode,
            "actual_completed_episodes": agent.completed_episodes, "frames": agent.frames,
            "latest_checkpoint_completed_episodes": last_checkpoint_episode,
            "best_training_hotspot_ev": best_score, "elapsed_seconds": time.monotonic() - started,
            "training_hotspot_unit": "eV", "reward_threshold_ev": env.threshold_ev,
            "resume_source": str(resume) if resume else None,
            "note": "Checkpoint resumes only complete episodes. A training-best exploratory trajectory is not a held-out evaluation.",
        })
    return {"status": status, "completed_episodes": agent.completed_episodes, "output_dir": str(output)}


def run_evaluation(config, *, predictor=None):
    settings = config["evaluation"]
    requested = positive_integer(settings.get("episodes"), "evaluation.episodes")
    settings.setdefault("max_attempts", requested * 100)
    settings.setdefault("temperature", 0.0)
    settings.setdefault("deduplicate", True)
    settings.setdefault("seed", 43)
    max_attempts = positive_integer(settings.get("max_attempts", requested * 100), "evaluation.max_attempts")
    if max_attempts < requested:
        raise ValueError("evaluation.max_attempts must be at least evaluation.episodes")
    temperature = settings.get("temperature", 0.0)
    if not isinstance(temperature, (int, float)) or not np.isfinite(temperature) or temperature < 0:
        raise ValueError("evaluation.temperature must be finite and nonnegative")
    deduplicate = settings.get("deduplicate", True)
    if not isinstance(deduplicate, bool):
        raise ValueError("evaluation.deduplicate must be a boolean")
    seed = settings.get("seed", 43)
    seed_runtime(seed, config["model"]["device"])
    network_config = agent_config(config)
    env = make_environment(config, predictor)
    agent = DQNAgent(env.state_dim, network_config, config["model"]["device"], seed)
    checkpoint = read_policy_checkpoint(Path(settings["policy_checkpoint"]), config["model"]["device"],
                                        config["model"]["allow_legacy_pickle"])
    saved_env = checkpoint.get("effective_config", {}).get("environment")
    if saved_env is not None and saved_env != config["environment"]:
        raise ValueError("Evaluation environment differs from the policy checkpoint configuration")
    saved_model = checkpoint.get("effective_config", {}).get("model", {})
    if "checkpoint_sha256" in saved_model:
        for key in ("checkpoint_sha256", "predictor_source"):
            if saved_model.get(key) != config["model"].get(key):
                raise ValueError("Evaluation surrogate differs from the policy training surrogate")
    agent.load_policy(checkpoint)
    output = new_output(settings["output_dir"])
    writer = CandidateWriter(output, "rl", config, env.top_k_atoms)
    seen = set()
    attempts = 0
    note = None
    for attempts in range(1, max_attempts + 1):
        state = env.reset()
        while not env.done:
            action = agent.select(state, env.action_mask(), temperature=temperature)
            state, reward, _, _ = env.step(action)
        sequence = tuple(env.actions)
        if not deduplicate or sequence not in seen:
            writer.append(f"rl_{attempts:06d}", env, reward, attempts)
        seen.add(sequence)
        if len(writer.manifest["candidates"]) >= requested:
            break
        if deduplicate and temperature == 0:
            note = "Greedy evaluation from a fixed state produces one action sequence; repeated copies were not counted as distinct candidates."
            break
    status = "complete" if len(writer.manifest["candidates"]) == requested else "incomplete_unique_population"
    if status != "complete" and note is None:
        note = "max_attempts reached before the requested number of unique action sequences was obtained"
    return writer.finish(requested, attempts, status, note)


def run_random(config, *, predictor=None):
    """Fixed-count shuffled sequences, matching the original baseline sampling rule."""
    settings = config["random"]
    settings.setdefault("seed", 44)
    requested = positive_integer(settings.get("episodes"), "random.episodes")
    seed = settings.get("seed", 44)
    seed_runtime(seed, config["model"]["device"])
    rng = np.random.default_rng(seed)
    env = make_environment(config, predictor)
    output = new_output(settings["output_dir"])
    writer = CandidateWriter(output, "random_fixed_count_shuffle", config, env.top_k_atoms)
    for episode in range(1, requested + 1):
        env.reset()
        constraints = env.constraints
        actions = np.concatenate([
            rng.integers(0, 3, constraints.rotations), np.full(constraints.vacancies, 3),
            np.full(constraints.total_steps - constraints.rotations - constraints.vacancies, 4),
        ]).astype(int)
        rng.shuffle(actions)
        for action in actions:
            _, reward, _, _ = env.step(int(action))
        writer.append(f"random_{episode:06d}", env, reward, episode)
    return writer.finish(requested, requested, note="Fixed-count sequences are shuffled; this is not sequential uniform sampling over masked actions.")
