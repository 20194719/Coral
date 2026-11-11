"""Run the three-dataset cardinality estimation method."""

import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import json
import os
from pathlib import Path
import sys
import traceback


def parser():
    """Define the command-line interface."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=("imdb", "biodex", "trex"))
    p.add_argument(
        "--models-config",
        type=Path,
        default=Path(__file__).resolve().with_name("models.json"),
        help="Deployment configuration (default: models.json beside run.py)",
    )
    p.add_argument(
        "data",
        type=Path,
        help="Dataset directory containing CSV tables and queries.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Empty output directory (default: configured output_root/DATASET)",
    )
    p.add_argument(
        "--embedding-model",
        help="Override the configured local embedding model directory",
    )
    p.add_argument("--embedding-device")
    p.add_argument("--embedding-batch", type=int)
    p.add_argument(
        "--filter-device", help="Override the configured MLP training device"
    )
    p.add_argument("--endpoints", nargs="+")
    p.add_argument("--llm-model", help="Qwen model name served by the endpoints")
    p.add_argument("--workers", type=int)
    p.add_argument("--threads", type=int)
    p.add_argument(
        "--seeds", nargs=3, type=int, default=[0, 1, 2], metavar=("S0", "S1", "S2")
    )
    p.add_argument("--filter-model", choices=("tree", "mlp"))
    p.add_argument(
        "--filter-cv",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Select filter parameters with two grouped five-fold protocols",
    )
    p.add_argument("--tree-params", action="append", metavar="TABLE=JSON")
    p.add_argument("--mlp-params", action="append", metavar="TABLE=JSON")
    p.add_argument("--mlp-epochs", type=int)
    p.add_argument("--filter-prior", type=float)
    p.add_argument("--join-prior", type=float)
    p.add_argument("--join-update", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--filter-verification",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Review, deduplicate, and check coverage of mined predicates",
    )
    p.add_argument(
        "--filter-rewrites",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Augment training predicates with reviewed paraphrases",
    )
    p.add_argument(
        "--filter-normalization",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Normalize query predicates with equivalence review",
    )
    p.add_argument("--filter-predicates", type=int)
    p.add_argument("--filter-documents", type=int)
    p.add_argument("--budget", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--groups", type=int)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue this output with identical inputs and settings",
    )
    p.add_argument(
        "--show-config",
        action="store_true",
        help="Print resolved settings and exit without running",
    )
    return p


def resolve(args):
    """Resolve dataset defaults and command-line overrides."""
    from src.config import configuration, deployment, table_parameters, validate

    runtime_keys = (
        "llm_model",
        "endpoints",
        "workers",
        "embedding_model",
        "embedding_device",
        "embedding_batch",
        "filter_device",
        "threads",
    )
    runtime = deployment(
        args.dataset,
        args.models_config,
        {key: getattr(args, key) for key in runtime_keys},
    )
    for key in runtime_keys:
        setattr(args, key, runtime[key])

    mapping = dict(
        filter_prior="filter_prior_strength",
        join_prior="join_prior_strength",
        join_update="join_calibration",
        filter_predicates="filter_predicate_count",
        filter_documents="filter_calib_docs",
    )
    names = [
        "llm_model",
        "filter_model",
        "filter_cv",
        "filter_verification",
        "filter_rewrites",
        "filter_normalization",
        "budget",
        "batch",
        "groups",
        "mlp_epochs",
        *mapping,
    ]
    cfg = configuration(
        args.dataset, {mapping.get(k, k): getattr(args, k) for k in names}
    )
    cfg["embedding_dimension"] = runtime["embedding_dimension"]
    cfg["runtime"] = runtime
    if cfg["filter_rewrites"]:
        cfg["filter_original_weight"] = 0.5
    if cfg["filter_cv"] and (
        args.tree_params or args.mlp_params or args.mlp_epochs is not None
    ):
        raise ValueError("Fixed filter parameter overrides require --no-filter-cv")
    table_parameters(cfg, "tree_parameters", args.tree_params)
    table_parameters(cfg, "mlp_parameters", args.mlp_params)
    validate(cfg)
    if len(set(args.seeds)) != 3 or any(seed < 0 for seed in args.seeds):
        raise ValueError("Provide three distinct nonnegative seeds")
    if min(args.workers, args.threads, args.embedding_batch) <= 0:
        raise ValueError("Workers, threads, and embedding batch must be positive")
    if args.dataset == "biodex" and args.filter_verification:
        raise ValueError(
            "BioDEX uses oracle calibration predicates; predicate mining verification is inapplicable"
        )
    return cfg


@contextmanager
def exclusive_run(output):
    """Prevent concurrent writers to one run directory."""
    import fcntl

    with (Path(output) / ".run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "This output directory already has an active runner"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def run_log(path):
    """Capture pipeline output while keeping final metrics on the console."""
    with Path(path).open("a", buffering=1) as stream:
        sys.stdout.flush()
        sys.stderr.flush()
        original = [os.dup(1), os.dup(2)]
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            with redirect_stdout(stream), redirect_stderr(stream):
                yield
        finally:
            stream.flush()
            for fd, saved in zip((1, 2), original):
                os.dup2(saved, fd)
                os.close(saved)


def main():
    """Run one dataset and print only final Q-error summaries."""
    args = parser().parse_args()
    cfg = resolve(args)
    if args.show_config:
        print(json.dumps(cfg, indent=2))
        return
    if args.dataset == "biodex":
        from src.data.tables import dataset_directory
        from src.inference.oracle import validate_oracle_files

        validate_oracle_files(dataset_directory(args.data, args.dataset), cfg["views"])
    if not Path(args.embedding_model).is_dir():
        raise ValueError(
            "Set embedding.model_path in models.json to an existing local model directory"
        )
    output = args.output or Path(cfg["runtime"]["output_root"]) / args.dataset
    if args.resume:
        if not (output / "manifest.json").exists():
            raise ValueError("Resume requires an existing manifest.json")
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f"Use an empty output directory or --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with exclusive_run(output), run_log(output / "run.log"):
        try:
            from src import prompts

            prompts.configure(args.dataset)
            from src.inference.embeddings import Encoder
            from src.runner import run
            from threadpoolctl import threadpool_limits

            encoder = Encoder(
                args.embedding_model,
                args.embedding_device,
                args.embedding_batch,
                cfg["embedding_dimension"],
            )
            with threadpool_limits(args.threads):
                summary = run(
                    cfg,
                    args.data,
                    output,
                    args.seeds,
                    encoder,
                    args.endpoints,
                    args.workers,
                    device=args.filter_device,
                    resume=args.resume,
                )
        except BaseException:
            traceback.print_exc()
            raise
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
