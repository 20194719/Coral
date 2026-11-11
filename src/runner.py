"""Run offline learning and online cardinality estimation."""

from pathlib import Path
import hashlib
from importlib.metadata import version
import json
import time
from src.data.storage import atomic_json
from src.data.tables import dataset_directory, load_dataset
from src.inference.oracle import validate_oracle_files
from src.inference.backend import Backend
from src.offline.pipeline import PairScores
from src.offline.filtering import Filters
from src.online.engine import TreeEngine
from src.online.checkpoints import CheckpointProbe
from src.evaluation import summarize


def sha256(path):
    """Hash a run input."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signatures(cfg, data_root):
    """Fingerprint source and input files for continuation."""
    package = Path(__file__).resolve().parent.parent
    source = {
        str(p.relative_to(package)): sha256(p)
        for p in [
            package / "run.py",
            *sorted((package / "src").rglob("*.py")),
            package / "src/defaults.json",
        ]
    }
    data = dataset_directory(data_root, cfg["dataset"])
    files = [data / "queries.json", *[data / f"{name}.csv" for name in cfg["tables"]]]
    if cfg["dataset"] == "biodex":
        files.extend(validate_oracle_files(data, cfg["views"]))
    return dict(source=source, inputs={str(p.resolve()): sha256(p) for p in files})


def metric_summary(rows, truth, seeds):
    """Summarize end-to-end Q-errors for three seeds."""
    keys = ("median", "p90", "p99", "max")
    aggregate = summarize(rows, truth, tuple(seeds))
    result = dict(
        queries=len(truth),
        seeds=list(seeds),
        per_seed={},
        aggregate={k: aggregate[k] for k in keys},
        aggregation=aggregate["aggregation"],
    )
    for seed in seeds:
        metrics = summarize([r for r in rows if r["seed"] == seed], truth, (seed,))
        result["per_seed"][str(seed)] = {k: metrics[k] for k in keys}
    return result


def run(
    cfg,
    data_root,
    output,
    seeds,
    encoder,
    endpoints,
    workers,
    device="cpu",
    resume=False,
):
    """Execute a complete three-seed benchmark."""
    output = Path(output)
    target = output / "manifest.json"
    data_root = dataset_directory(data_root, cfg["dataset"])
    dataset = load_dataset(data_root, cfg["dataset"])
    identity = dict(
        configuration=cfg,
        seeds=list(seeds),
        endpoints=endpoints,
        embedding=dict(
            model=encoder.path, device=encoder.device, batch=encoder.batch_size
        ),
        filter_device=device,
        libraries={
            name: version(name) for name in ("numpy", "scipy", "scikit-learn", "torch")
        },
        fingerprints=signatures(cfg, data_root),
    )
    if resume:
        previous = json.loads(target.read_text())
        if previous["identity"] != identity:
            raise ValueError(
                "Resume requires identical code, data, configuration, and model services"
            )
        if previous["status"] == "complete":
            return json.loads((output / "summary.json").read_text())
    llm = Backend(
        cfg["llm_model"],
        endpoints,
        workers,
        output,
        resume=resume,
        oracle=cfg["dataset"] == "biodex",
        options=cfg["runtime"]["llm_options"],
    )
    meta = dict(identity=identity, status="running", started=time.time())
    atomic_json(target, meta)
    rows = (
        json.loads((output / "results.json").read_text())
        if resume and (output / "results.json").exists()
        else []
    )
    try:
        views, shared = [], {}
        for spec in cfg["views"]:
            tables = [dataset.tables[t] for t in spec["tables"]]
            paths, positives, members = llm.inputs(data_root, spec, tables)
            destination = output / spec["name"]
            destination.mkdir(exist_ok=True)
            pair = PairScores(
                cfg,
                spec,
                paths,
                tables,
                encoder,
                llm,
                destination,
                positives,
                resume=resume,
            )
            filters = Filters(
                cfg,
                spec,
                paths,
                tables,
                pair.emb,
                encoder,
                llm,
                destination,
                shared,
                device=device,
                resume=resume,
            )
            views.append(
                dict(
                    spec=spec,
                    paths=paths,
                    tables=tables,
                    pair=pair,
                    filters=filters,
                    positives=positives,
                    members=members,
                )
            )
        prepared = []
        for query in dataset.queries:
            selected = [
                v for v in views if set(v["spec"]["tables"]).issubset(query.tables)
            ]
            if len(selected) != len(query.tables) - 1:
                raise ValueError("Every query must form a connected join tree")
            baselines, edges, probes, predictions = [], [], [], {}
            for view in selected:
                condition = (
                    query.condition
                    if len(selected) == 1
                    else view["paths"]["condition"]
                )
                baselines.append(view["pair"].baseline(condition))
                edges.append(
                    tuple(query.tables.index(t) for t in view["spec"]["tables"])
                )
                probes.append(llm.probe(view, condition, query.filters))
                for name in view["spec"]["tables"]:
                    if name not in predictions:
                        predictions[name] = view["filters"].prediction(
                            name, query.filters.get(name)
                        )
            prepared.append(
                (
                    query,
                    baselines,
                    edges,
                    probes,
                    [predictions[t] for t in query.tables],
                )
            )
        completed = {(r["qid"], r["seed"]) for r in rows}
        if len(completed) != len(rows):
            raise ValueError("Duplicate saved query results")
        for query, baselines, edges, probes, predictions in prepared:
            for seed in seeds:
                if (query.id, seed) in completed:
                    continue
                checkpoint = output / "online" / f"q{query.id}-seed{seed}"
                judges = [
                    CheckpointProbe(probe, checkpoint / f"edge-{i}")
                    for i, probe in enumerate(probes)
                ]
                engine = TreeEngine(
                    baselines,
                    edges,
                    predictions,
                    groups=cfg["groups"],
                    join_prior_strength=cfg["join_prior_strength"],
                    filter_prior_strength=cfg["filter_prior_strength"],
                    join_calibration=cfg["join_calibration"],
                )
                result = engine.run(
                    judges, cfg["budget"], seed=seed, batch=cfg["batch"]
                )
                atomic_json(checkpoint / "audit.json", result.refinement)
                rows.append(
                    dict(
                        qid=query.id,
                        seed=seed,
                        estimate=result.ce,
                        judgments=result.calls,
                    )
                )
                atomic_json(output / "results.json", rows)
        summary = dict(
            dataset=cfg["dataset"], **metric_summary(rows, dataset.ground_truth, seeds)
        )
        atomic_json(output / "summary.json", summary)
        meta.update(status="complete", finished=time.time())
        return summary
    except BaseException as exc:
        meta.update(status="failed", error=repr(exc))
        raise
    finally:
        meta["llm"] = llm.statistics()
        atomic_json(target, meta)
        llm.close()
