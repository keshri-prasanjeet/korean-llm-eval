"""Explicit provider settings and a dotenv reader that never executes shell code."""

import os
import re
from pathlib import Path
from urllib.parse import urlparse

from .dataset import require
from .storage import read_json

KEYS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY", "kimi": "MOONSHOT_API_KEY", "compatible": "LOCAL_API_KEY", "nvidia": "NVIDIA_API_KEY"}


def load_env(path):
    path = Path(path)
    if not path.exists():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        key, sep, value = line.partition("=")
        require(bool(sep) and bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)), f"Invalid .env entry on line {number}")
        value = value.strip()
        if value.startswith(('"', "'")):
            require(len(value) >= 2 and value[-1] == value[0], f"Unclosed .env quote on line {number}")
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        if value:  # Blank process variables should not hide a locally supplied key.
            if not os.environ.get(key):
                os.environ[key] = value


def load_config(path, *, dry_run=False):
    path = Path(path).resolve()
    config = read_json(path)
    require(set(config) <= {"dataset_files", "models", "output_dir", "concurrency", "max_attempts", "timeout_seconds", "retry_backoff_seconds"}, "Unknown config field")
    require(bool(config.get("dataset_files")) and bool(config.get("models")), "Config requires dataset_files and models")
    config["dataset_files"] = [str((path.parent / p).resolve()) for p in config["dataset_files"]]
    config["output_dir"] = str((path.parent / config.get("output_dir", "../outputs/benchmark")).resolve())
    config.setdefault("concurrency", 2)
    config.setdefault("max_attempts", 3)
    config.setdefault("timeout_seconds", 120)
    config.setdefault("retry_backoff_seconds", 2)
    require(type(config["concurrency"]) is int and config["concurrency"] > 0, "concurrency must be positive")
    require(type(config["max_attempts"]) is int and config["max_attempts"] > 0, "max_attempts must be positive")
    require(config["timeout_seconds"] > 0 and config["retry_backoff_seconds"] >= 0, "Invalid timeout/backoff")
    names = set()
    for model in config["models"]:
        require(set(model) <= {"name", "provider", "model", "model_env", "api_key_env", "base_url", "parameters", "responses"}, "Unknown model config field")
        require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", model.get("name", ""))), "Model name must be a safe filename")
        require(model["name"] not in names, "Duplicate model name")
        names.add(model["name"])
        provider = model["provider"]
        require(provider in {"mock", *KEYS}, f"Unknown provider: {provider}")
        require(not ("model" in model and "model_env" in model), "Choose model or model_env, not both")
        if "model_env" in model:
            model["model"] = os.environ.get(model["model_env"], "")
        require(bool(model.get("model")) or dry_run, f"Set {model.get('model_env', 'model')} for {model['name']}")
        if provider != "mock":
            model.setdefault("api_key_env", KEYS[provider])
        parameters = model.setdefault("parameters", {})
        require(set(parameters) <= {"max_output_tokens", "temperature", "top_p", "seed", "reasoning_effort", "enable_thinking"}, "Unknown generation parameter")
        if "enable_thinking" in parameters:
            require(provider == "nvidia" and type(parameters["enable_thinking"]) is bool, "enable_thinking is a boolean NVIDIA option")
        if provider == "nvidia":
            require("reasoning_effort" not in parameters, "Use enable_thinking for NVIDIA")
        parameters.setdefault("max_output_tokens", 2048)
        require(type(parameters["max_output_tokens"]) is int and parameters["max_output_tokens"] > 0, "Invalid output token limit")
        if provider in {"anthropic", "google"}:
            require(not {"seed", "reasoning_effort"} & parameters.keys(), f"seed/reasoning_effort unsupported by {provider} adapter")
        if provider == "kimi":
            require("seed" not in parameters, "seed is unsupported by the Kimi adapter")
            if "reasoning_effort" in parameters:
                require(parameters["reasoning_effort"] in {"low", "high", "max"}, "Kimi reasoning_effort must be low, high or max")
        if provider == "compatible":
            require("reasoning_effort" not in parameters, "Configure compatible server reasoning outside this adapter")
            require(bool(model.get("base_url")), "compatible provider needs base_url ending in /v1")
        if "base_url" in model:
            parsed = urlparse(model["base_url"])
            require(parsed.scheme in {"http", "https"} and parsed.hostname and not (parsed.username or parsed.password or parsed.query or parsed.fragment), "base_url must not contain credentials, query parameters or fragments")
        if provider == "mock":
            require(bool(model.get("responses")) and all(isinstance(x, str) for x in model["responses"]), "mock requires a nonempty response list")
    return config


def missing_access(config):
    missing = []
    for model in config["models"]:
        if not model.get("model"):
            missing.append(model.get("model_env", f"model ID for {model['name']}"))
        if model["provider"] not in {"mock", "compatible"} and not os.environ.get(model["api_key_env"]):
            missing.append(model["api_key_env"])
    return sorted(set(missing))
