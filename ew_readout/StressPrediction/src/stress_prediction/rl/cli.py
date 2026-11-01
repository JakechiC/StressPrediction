"""Explicit Dueling DQN training and candidate generation commands."""
import argparse
from pathlib import Path

def build_parser():
    parser = argparse.ArgumentParser(description="Graphene policy training and candidate generation")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("train", "Train a masked Dueling DQN policy"),
        ("evaluate", "Generate candidate structures from a saved policy"),
        ("random", "Generate candidates using the fixed-count shuffled baseline"),
    ):
        sub = commands.add_parser(name, help=description)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--output", type=Path)
        sub.add_argument("--episodes", type=int)
        sub.add_argument("--allow-legacy-pickle", action="store_true")
        if name == "train":
            sub.add_argument("--resume", type=Path)
        elif name == "evaluate":
            sub.add_argument("--policy", type=Path)
            sub.add_argument("--temperature", type=float)
    return parser

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from .io import load_config
        config = load_config(args.config)
        section = {"train": "training", "evaluate": "evaluation", "random": "random"}[args.command]
        if args.output is not None:
            config[section]["output_dir"] = str(args.output.expanduser().resolve())
        if "output_dir" not in config[section]:
            raise ValueError(f"{section}.output_dir or --output is required")
        if args.episodes is not None:
            config[section]["episodes"] = args.episodes
        if args.allow_legacy_pickle:
            config["model"]["allow_legacy_pickle"] = True
        if args.command == "train":
            from .pipeline import run_training
            resume = args.resume.expanduser().resolve() if args.resume else None
            if resume is not None and not resume.is_file():
                raise FileNotFoundError(resume)
            result = run_training(config, resume=resume)
        elif args.command == "evaluate":
            from .pipeline import run_evaluation
            if args.policy is not None:
                config[section]["policy_checkpoint"] = str(args.policy.expanduser().resolve())
            if args.temperature is not None:
                config[section]["temperature"] = args.temperature
            result = run_evaluation(config)
        else:
            from .pipeline import run_random
            result = run_random(config)
        status = result.get("status", "complete")
        print(f"status={status}; output={config[section]['output_dir']}")
        return 0 if status == "complete" else 3
    except KeyboardInterrupt:
        print("Interrupted; completed candidate records and saved checkpoints are retained.")
        return 130
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, ImportError) as error:
        parser.error(str(error))

if __name__ == "__main__":
    raise SystemExit(main())
