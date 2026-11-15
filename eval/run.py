"""Generate reproducible multi-turn runs: python -m eval.run --help."""

import argparse
import asyncio
import contextlib
import fcntl
import random
import subprocess
import sys
import uuid
from pathlib import Path

from .config import load_config, load_env, missing_access
from .dataset import load_dataset
from .models import make_adapter
from .models.base import ProviderError
from .storage import append_jsonl, digest, load_generations, now, read_json, read_jsonl, text_digest, write_json

ROOT = Path(__file__).resolve().parent.parent


def code_fingerprint():
    return digest({str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sorted((ROOT / "eval").rglob("*")) if p.suffix in {".py", ".html"}})


def git_revision():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@contextlib.contextmanager
def run_lock(run_dir):
    """Kernel lock is released even if the process crashes; no stale PID guessing."""
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process is writing this run") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def generation_path(run_dir, model_name, dialogue_id, turn_id):
    return run_dir / "generations" / model_name / f"{dialogue_id}__{turn_id:03}.json"


def validate_history(saved, messages, manifest, model, dialogue, turn):
    expected = (model["name"], dialogue["id"], turn["turn_id"])
    actual = (saved["model_key"], saved["dialogue_id"], saved["turn_id"])
    if actual != expected or saved["run_id"] != manifest["run_id"]:
        raise ValueError("Saved generation identity does not match run")
    if saved["request_messages"] != messages or saved["dialogue_sha256"] != digest(dialogue):
        raise ValueError("Saved conversation history does not match the frozen script")
    if saved["response_sha256"] != text_digest(saved["raw_response"]):
        raise ValueError("Saved response hash mismatch")


async def execute(config, dialogues, run_dir, *, resume=False, retry_unknown=False, adapters=None,
                  max_requests=None, continue_after_incomplete=False):
    fingerprint = {
        "config_sha256": digest(config), "dataset_sha256": digest(dialogues),
        "code_sha256": code_fingerprint(),
    }
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        if not resume:
            raise ValueError("Run already exists; use --resume or choose another --output")
        manifest = read_json(manifest_path)
        if any(manifest.get(k) != v for k, v in fingerprint.items()):
            raise ValueError("Dataset, configuration or code changed; start a new run directory")
        if read_json(run_dir / "dataset.json") != dialogues:
            raise ValueError("Frozen dataset snapshot changed")
    else:
        if resume:
            raise ValueError("Cannot resume: no manifest in this directory")
        if any(p.name != ".lock" for p in run_dir.iterdir()):
            raise ValueError("New run directory must be empty")
        manifest = {
            "run_id": uuid.uuid4().hex, "created_at": now(),
            "schema_version": "1.2.0", "runner_version": "0.1.0", "git_revision": git_revision(),
            "synthetic": any(m["provider"] == "mock" for m in config["models"]),
            "config": config, **fingerprint,
            "planned_replies": sum(len(d["turns"]) for d in dialogues) * len(config["models"]),
        }
        write_json(run_dir / "dataset.json", dialogues)
        write_json(manifest_path, manifest)

    attempts_path = run_dir / "attempts.jsonl"
    events = read_jsonl(attempts_path)
    unfinished = {e["attempt_id"]: e for e in events if e["event"] == "REQUEST_STARTED"}
    for event in events:
        if event["event"] in {"DELIVERED", "ERROR", "RETRY_UNKNOWN_ACKNOWLEDGED"}:
            unfinished.pop(event["attempt_id"], None)
    delivered_ids = {g["attempt_id"] for g in load_generations(run_dir) if g["transport_status"] == "DELIVERED"}
    ambiguous = [event for aid, event in unfinished.items() if aid not in delivered_ids]
    if ambiguous and not retry_unknown:
        raise ValueError("An interrupted request has unknown delivery. Inspect attempts.jsonl; use --retry-unknown to explicitly retry it (may bill twice)")
    for event in ambiguous:
        append_jsonl(attempts_path, {**event, "event": "RETRY_UNKNOWN_ACKNOWLEDGED", "at": now()})

    semaphore = asyncio.Semaphore(config["concurrency"])
    disabled = set()
    errors = []
    request_count = 0
    limit_reached = False
    halted = {}

    def incomplete(record):
        return not record.get("raw_response", "").strip() or record.get("generation_limit_reached", False)

    async def conversation(model, dialogue):
        nonlocal request_count, limit_reached
        async with semaphore:
            adapter = adapters[model["name"]] if adapters else make_adapter(model, config["timeout_seconds"])
            messages = [{"role": "system", "content": dialogue["system_prompt"]}]
            for turn in dialogue["turns"]:
                messages.append({"role": "user", "content": turn["user"]})
                path = generation_path(run_dir, model["name"], dialogue["id"], turn["turn_id"])
                if path.exists():
                    saved = read_json(path)
                    if saved["transport_status"] == "DELIVERED":
                        validate_history(saved, messages, manifest, model, dialogue, turn)
                        if incomplete(saved) and not continue_after_incomplete:
                            disabled.add(model["name"])
                            halted[model["name"]] = "Saved empty/limited output: inspect it before using --continue-after-incomplete"
                            return
                        messages.append(saved.get("history_message") or {"role": "assistant", "content": saved["raw_response"]})
                        continue
                if model["name"] in disabled:
                    return
                endpoint, payload = adapter.prepare(messages)
                base = {
                    "record_kind": "generation", "run_id": manifest["run_id"], "model_key": model["name"],
                    "model_id": model["model"], "provider": model["provider"], "synthetic": model["provider"] == "mock",
                    "dialogue_id": dialogue["id"], "dialogue_sha256": digest(dialogue), "condition": dialogue["condition"],
                    "turn_id": turn["turn_id"], "request_messages": list(messages),
                    "request_endpoint": endpoint, "request_payload": payload,
                }
                for number in range(config["max_attempts"]):
                    if model["name"] in disabled:
                        return
                    if max_requests is not None and request_count >= max_requests:
                        limit_reached = True
                        return
                    request_count += 1
                    attempt_id = uuid.uuid4().hex
                    started = now()
                    identity = {"attempt_id": attempt_id, "model_key": model["name"], "dialogue_id": dialogue["id"], "turn_id": turn["turn_id"]}
                    append_jsonl(attempts_path, {**identity, "event": "REQUEST_STARTED", "at": started})
                    try:
                        response = await adapter.generate(messages)
                    except ProviderError as exc:
                        failure = {**identity, "event": "DELIVERY_UNKNOWN" if exc.delivery_unknown else "ERROR", "at": now(),
                                   "error": str(exc), "retryable": exc.retryable, "http_status": exc.status,
                                   "retry_after_seconds": exc.retry_after}
                        append_jsonl(attempts_path, failure)
                        delay = max(exc.retry_after or 0, config["retry_backoff_seconds"] * 2 ** number)
                        if exc.retryable and not exc.delivery_unknown and delay <= 30 and number + 1 < config["max_attempts"]:
                            await asyncio.sleep(delay + random.uniform(0, min(1, delay / 4)))
                            continue
                        write_json(path, {**base, "transport_status": "UNKNOWN" if exc.delivery_unknown else "ERROR", "raw_response": None,
                                          "response_sha256": None, "attempt_id": attempt_id, "error": str(exc), "queried_at": started, "completed_at": now()})
                        errors.append(identity)
                        disabled.add(model["name"])
                        halted[model["name"]] = str(exc) + (f"; wait at least {delay:g}s before retrying" if delay > 30 else "")
                        return
                    record = {
                        **base, "transport_status": "DELIVERED", "raw_response": response.text,
                        "response_sha256": text_digest(response.text), "model_id": response.model_id,
                        "finish_reason": response.finish_reason, "generation_limit_reached": response.generation_limit_reached,
                        "usage": response.usage, "provider_request_id": response.request_id,
                        "provider_response": response.provider_response, "attempt_id": attempt_id,
                        "queried_at": started, "completed_at": now(),
                    }
                    if response.history_message is not None:
                        record["history_message"] = response.history_message
                    write_json(path, record)
                    append_jsonl(attempts_path, {**identity, "event": "DELIVERED", "at": record["completed_at"]})
                    if incomplete(record) and not continue_after_incomplete:
                        disabled.add(model["name"])
                        halted[model["name"]] = "Empty or limited output saved; inspect it before using --continue-after-incomplete"
                        return
                    messages.append(response.history_message or {"role": "assistant", "content": response.text})
                    print(f"{model['name']} {dialogue['id']} {turn['turn_id']}/{len(dialogue['turns'])}", flush=True)
                    break

    await asyncio.gather(*(conversation(model, dialogue) for model in config["models"] for dialogue in dialogues))
    generations = load_generations(run_dir)
    delivered = sum(g["transport_status"] == "DELIVERED" for g in generations)
    status = {"run_id": manifest["run_id"], "updated_at": now(), "planned_replies": manifest["planned_replies"],
              "delivered_replies": delivered, "complete": delivered == manifest["planned_replies"] and not halted,
              "failed_this_invocation": len(errors), "disabled_models": sorted(disabled), "halt_reasons": halted,
              "requests_this_invocation": request_count, "request_limit_reached": limit_reached}
    write_json(run_dir / "status.json", status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, help="Override run directory")
    parser.add_argument("--dataset", nargs="+", type=Path, help="Override configured dataset files")
    parser.add_argument("--max-dialogues", type=int, help="Run only the first N conversations")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list access requirements; no API calls or run files")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-unknown", action="store_true", help="Explicitly retry interrupted requests whose delivery is unknown")
    parser.add_argument("--max-requests", type=int, help="Maximum new HTTP attempts in this invocation, including retries")
    parser.add_argument("--full-run", action="store_true", help="Explicitly permit an uncapped final-corpus run")
    parser.add_argument("--continue-after-incomplete", action="store_true", help="Use saved empty/limited outputs in later history after inspection")
    args = parser.parse_args(argv)
    try:
        load_env(args.env_file)
        config = load_config(args.config, dry_run=args.dry_run)
        if args.output:
            config["output_dir"] = str(args.output.resolve())
        if args.dataset:
            config["dataset_files"] = [str(path.resolve()) for path in args.dataset]
        dialogues, sources = load_dataset(config["dataset_files"])
        if args.max_dialogues is not None:
            if args.max_dialogues < 1:
                raise ValueError("--max-dialogues must be positive")
            dialogues = dialogues[:args.max_dialogues]
        config["dataset_sources"] = sources
        final = any(s.get("release_status") == "FROZEN" for s in sources)
        if args.max_requests is not None and args.max_requests < 1:
            raise ValueError("--max-requests must be positive")
        if args.full_run and args.max_requests is not None:
            raise ValueError("Choose --max-requests or --full-run")
        for model in config["models"]:
            if model.get("model"):
                make_adapter(model).prepare([{"role": "system", "content": "preflight"}, {"role": "user", "content": "preflight"}])
        missing = missing_access(config)
        count = sum(len(d["turns"]) for d in dialogues) * len(config["models"])
        print(f"{len(dialogues)} conversations × {len(config['models'])} model(s); {count} planned replies")
        print("Dataset version: " + ", ".join(s["dataset_version"] for s in sources))
        print("Models: " + ", ".join(f"{m['provider']}:{m.get('model') or '(unset)'}; output cap {m['parameters']['max_output_tokens']}" for m in config["models"]))
        print(f"New-request limit: {args.max_requests if args.max_requests is not None else 'uncapped'}")
        print("Access: " + ("set " + ", ".join(missing) if missing else "configured"))
        if args.dry_run:
            return 0
        if missing:
            raise ValueError("Missing access configuration: " + ", ".join(missing))
        if final and any(m["provider"] != "mock" for m in config["models"]) and args.max_requests is None and not args.full_run:
            raise ValueError("Final-corpus live runs require --max-requests N or --full-run. Start with --max-requests 1.")
        run_dir = Path(config["output_dir"])
        with run_lock(run_dir):
            status = asyncio.run(execute(config, dialogues, run_dir, resume=args.resume, retry_unknown=args.retry_unknown,
                                        max_requests=args.max_requests, continue_after_incomplete=args.continue_after_incomplete))
        print(f"Saved {status['delivered_replies']}/{count} replies to {run_dir}")
        for model, reason in status["halt_reasons"].items():
            print(f"Stopped {model}: {reason}")
        if status["request_limit_reached"]:
            print("Paused at the request limit. Resume this exact directory to reuse saved replies.")
        return 0 if status["complete"] or (status["request_limit_reached"] and not status["halt_reasons"]) else 2
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
