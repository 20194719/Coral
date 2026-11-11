"""Dataset presets and runtime overrides."""

from pathlib import Path
import json
import math
from urllib.parse import urlparse


MODELS_PATH = Path(__file__).resolve().parents[1] / "models.json"


def deployment(dataset, path=MODELS_PATH, overrides=None):
    """Load deployment settings and optional command-line overrides."""
    path = Path(path).expanduser().resolve()
    settings = json.loads(path.read_text())
    expected = {
        "llm": {
            "model",
            "endpoints",
            "api_key_env",
            "workers",
            "timeout_seconds",
            "max_attempts",
            "temperature",
            "top_p",
            "enable_thinking",
        },
        "embedding": {
            "model_path",
            "dimension",
            "device",
            "batch_size",
            "batch_size_by_dataset",
        },
        "runtime": {"filter_device", "threads", "output_root"},
    }
    if set(settings) != set(expected):
        raise ValueError(f"{path}: expected sections {sorted(expected)}")
    for section, keys in expected.items():
        if not isinstance(settings[section], dict) or set(settings[section]) != keys:
            raise ValueError(f"{path}: {section} requires keys {sorted(keys)}")
    llm, emb, runtime = (settings[k] for k in ("llm", "embedding", "runtime"))
    batches = emb["batch_size_by_dataset"]
    if not isinstance(batches, dict) or set(batches) - {"imdb", "trex", "biodex"}:
        raise ValueError(
            "embedding.batch_size_by_dataset accepts imdb, trex, and biodex"
        )
    if any(
        type(v) is not int or v <= 0 for v in [emb["batch_size"], *batches.values()]
    ):
        raise ValueError("Embedding batch sizes must be positive integers")
    values = dict(
        llm_model=llm["model"],
        endpoints=llm["endpoints"],
        workers=llm["workers"],
        llm_options={
            k: v for k, v in llm.items() if k not in ("model", "endpoints", "workers")
        },
        embedding_model=emb["model_path"],
        embedding_dimension=emb["dimension"],
        embedding_device=emb["device"],
        embedding_batch=batches.get(dataset, emb["batch_size"]),
        filter_device=runtime["filter_device"],
        threads=runtime["threads"],
        output_root=runtime["output_root"],
    )
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    values.update(overrides)
    for key in ("embedding_model", "output_root"):
        value = values[key]
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ValueError(f"{key} must be a nonempty path")
        resolved = Path(value).expanduser()
        base = Path.cwd() if key in overrides else path.parent
        values[key] = str((base / resolved).resolve())
    for key in ("workers", "threads", "embedding_batch", "embedding_dimension"):
        if type(values[key]) is not int or values[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("llm_model", "embedding_device", "filter_device"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    endpoints = values["endpoints"]
    if (
        not isinstance(endpoints, list)
        or not endpoints
        or any(
            not isinstance(url, str)
            or urlparse(url).scheme not in ("http", "https")
            or not urlparse(url).netloc
            for url in endpoints
        )
    ):
        raise ValueError("llm.endpoints must contain HTTP(S) service URLs")
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("llm.endpoints must be distinct")
    if not isinstance(llm["api_key_env"], str) or not llm["api_key_env"].strip():
        raise ValueError("llm.api_key_env must name an environment variable")
    for key in ("timeout_seconds", "temperature", "top_p"):
        if type(llm[key]) not in (int, float) or not math.isfinite(llm[key]):
            raise ValueError(f"llm.{key} must be finite")
    if (
        llm["timeout_seconds"] <= 0
        or not 0 <= llm["temperature"] <= 2
        or not 0 < llm["top_p"] <= 1
    ):
        raise ValueError("Invalid LLM timeout or sampling settings")
    if type(llm["max_attempts"]) is not int or llm["max_attempts"] <= 0:
        raise ValueError("llm.max_attempts must be a positive integer")
    if type(llm["enable_thinking"]) is not bool:
        raise ValueError("llm.enable_thinking must be boolean")
    return values


def presets():
    """Load dataset defaults."""
    return json.loads(Path(__file__).with_name("defaults.json").read_text())


def configuration(name, overrides=None):
    """Load a dataset's reference configuration."""
    cfg = presets()[name]
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return cfg


def table_parameters(cfg, option, values):
    """Apply per-table model parameter overrides."""
    for value in values or []:
        table, separator, payload = value.partition("=")
        if not separator or table not in cfg["tables"]:
            raise ValueError(f"Expected TABLE=JSON for one of {cfg['tables']}")
        parameters = json.loads(payload)
        allowed = set(cfg[option][table])
        if not isinstance(parameters, dict) or set(parameters) - allowed:
            raise ValueError(f"Allowed {option} keys: {sorted(allowed)}")
        cfg[option][table].update(parameters)


def validate(cfg):
    """Validate the runtime configuration."""
    for key in (
        "budget",
        "batch",
        "groups",
        "filter_calib_docs",
        "filter_predicate_count",
        "mlp_epochs",
    ):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("filter_prior_strength", "join_prior_strength"):
        value = cfg[key]
        if not isinstance(value, (float, int)) or not 0 < value < float("inf"):
            raise ValueError(f"{key} must be finite and positive")
    if not 0 < cfg["filter_original_weight"] <= 1:
        raise ValueError("filter_original_weight must lie in (0, 1]")
