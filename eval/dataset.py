"""Load and validate the published Korean benchmark."""

import argparse
from pathlib import Path

from .storage import digest, read_json

CONDITIONS = ("MAINTAIN", "SWITCH-EXPLICIT", "SWITCH-RELATIONAL", "IMPLICIT", "REFERENT")
LEVELS = {"HAPSYO", "HAEYO", "HAE", "HAERA"}
SENTENCE_INSTRUCTION = "각 답변에는 완결된 문장을 적어도 한 문장 포함하세요."


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_dialogues(dialogues):
    require(isinstance(dialogues, list) and bool(dialogues), "Dataset must be a nonempty JSON array")
    ids = set()
    for dialogue in dialogues:
        require(isinstance(dialogue, dict), "Each dialogue must be an object")
        required = {"id", "condition", "title", "system_prompt", "switch_turn", "turns"}
        require(required <= dialogue.keys(), "Missing dialogue fields")
        did = dialogue["id"]
        require(isinstance(did, str) and did and all(c.isalnum() or c in "_-" for c in did), "Unsafe dialogue ID")
        require(did not in ids, f"Duplicate dialogue ID: {did}")
        ids.add(did)
        require(dialogue["condition"] in CONDITIONS, f"{did}: unknown condition")
        require(isinstance(dialogue["system_prompt"], str) and SENTENCE_INSTRUCTION in dialogue["system_prompt"], f"{did}: missing visible sentence requirement")
        turns = dialogue["turns"]
        require(isinstance(turns, list) and len(turns) >= 1, f"{did}: no turns")
        require([t["turn_id"] for t in turns] == list(range(1, len(turns) + 1)), f"{did}: turns must be consecutive from 1")
        switch = dialogue["switch_turn"]
        switching = dialogue["condition"].startswith("SWITCH-")
        require((type(switch) is int and 1 < switch <= len(turns)) if switching else switch is None, f"{did}: invalid switch boundary")
        entities = [e["id"] for e in dialogue.get("entities", [])]
        require(len(set(entities)) == len(entities), f"{did}: duplicate entities")
        for turn in turns:
            loc = f"{did} turn {turn['turn_id']}"
            require(isinstance(turn.get("user"), str) and turn["user"].strip(), f"{loc}: empty user message")
            require(isinstance(turn.get("context_rationale"), str) and turn["context_rationale"].strip(), f"{loc}: missing rationale")
            triggers = {e["turn_index"] for e in dialogue.get("switch_events", [])} | ({switch} if switching else set())
            phase = ("PRE_SWITCH" if turn["turn_id"] < switch else "SWITCH_TURN" if turn["turn_id"] in triggers else "POST_SWITCH") if switching else "STABLE"
            require(turn["phase"] == phase, f"{loc}: phase/boundary mismatch")
            expected = turn["expected"]
            require(set(expected) == {"speech_levels", "address_fit", "grammar", "completion", "min_complete_sentences", "content_requirements", "referent_targets"}, f"{loc}: invalid expected fields")
            levels = expected["speech_levels"]
            require(isinstance(levels, list) and bool(levels) and len(set(levels)) == len(levels) and set(levels) <= LEVELS, f"{loc}: invalid speech levels")
            require(expected["address_fit"] == ["ACCEPTABLE"] and set(expected["grammar"]) == {"G_OK", "G_MINOR"} and expected["completion"] == ["C_PASS"], f"{loc}: unsupported expected checks")
            require(type(expected["min_complete_sentences"]) is int and expected["min_complete_sentences"] >= 1, f"{loc}: minimum complete sentences must be a positive integer")
            propositions = expected["content_requirements"]
            pids = [p["id"] for p in propositions]
            require(bool(propositions) and len(set(pids)) == len(pids) and all(isinstance(p.get("meaning"), str) and p["meaning"].strip() for p in propositions), f"{loc}: invalid content requirements")
            targets = expected["referent_targets"]
            require(bool(targets) == (dialogue["condition"] == "REFERENT"), f"{loc}: referent targets/condition mismatch")
            require(len({r["id"] for r in targets}) == len(targets), f"{loc}: duplicate referent targets")
            for target in targets:
                require(target["subject_entity"] in entities and target["proposition_id"] in pids, f"{loc}: broken referent links")
                probes = target["canonical_feature_probes"]
                require(bool(probes) and len({p["id"] for p in probes}) == len(probes), f"{loc}: invalid probes")
                features = {p["feature"]: p["expectation"] for p in probes}
                require(set(features) <= {"subject_honorification", "subject_honorific_lexeme"} and set(features.values()) <= {"REQUIRED", "FORBIDDEN", "OPTIONAL"}, f"{loc}: invalid feature")
                require(bool(target["accepted_strategies"]), f"{loc}: no accepted strategy")
                require(all(s["feature_constraints"] == features for s in target["accepted_strategies"]), f"{loc}: strategy/probe mismatch")
        for turn in turns:
            active = [e for e in dialogue.get("switch_events", []) if e["turn_index"] <= turn["turn_id"]]
            target = active[-1]["target_levels_after"] if active else (
                dialogue["post_switch_target_levels"] if switching and turn["turn_id"] >= switch
                else dialogue["initial_target_levels"])
            target = dialogue.get("turn_target_overrides", {}).get(str(turn["turn_id"]), target)
            require(turn["expected"]["speech_levels"] == target, f"{did}: target schedule mismatch")
    return dialogues


def load_dataset(paths):
    dialogues, sources = [], []
    for path in paths:
        path = Path(path).resolve()
        data = read_json(path)
        require(isinstance(data, dict) and data.get("dataset_version") == "1.0.0",
                "Expected a version 1.0.0 benchmark dataset")
        rows = validate_dialogues(data["conversations"])
        require(len(rows) == data["conversation_count"], "Conversation count mismatch")
        require(sum(len(d["turns"]) for d in rows) == data["expected_response_slot_count"],
                "Response-slot count mismatch")
        sources.append({"path": str(path), "sha256": digest(path.read_bytes()),
                        "dataset_id": data["dataset_id"], "dataset_version": data["dataset_version"],
                        "release_status": "FROZEN"})
        dialogues.extend(rows)
    validate_dialogues(dialogues)
    return dialogues, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    default = Path(__file__).resolve().parents[1] / "data/dataset.json"
    dialogues, _ = load_dataset(args.files or [default])
    print(f"Valid: {len(dialogues)} conversations, {sum(len(d['turns']) for d in dialogues)} turns")


if __name__ == "__main__":
    main()
