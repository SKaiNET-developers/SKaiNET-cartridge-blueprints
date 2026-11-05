#!/usr/bin/env python3
"""Validate saved C ABI execution against frozen HF references; never runs inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SHA256 = "9bba3c56cd1ef374c4260fa42f72bea292a26de0fcdfec870c7941422dd61782"
REQUEST_SHA256 = {
    "arithmetic": "247ede37b4c49cf1ff73641f1cbaa2a54e0378d20f4c5a81b63d26507f0de87c",
    "system": "d96fd56ea374abe2f2c4fb7e7bbaeef43d83160d320b04c70f0f120bf88fe9f8",
    "multiturn": "8a9d24ffbdc3e9208797dfbf55611ecfc4cb252384d9a3117befcae0bc070b5d",
}

class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def references():
    data = (ROOT / "samples/golden-transcripts.json").read_bytes()
    require(sha(data) == GOLDEN_SHA256, "Golden transcript fixture changed")
    return json.loads(data)["cases"]


def check_events(events, generated, expected_text, label):
    require(isinstance(events, list) and len(events) >= 3, f"{label}: missing Ready/Delta/Final")
    require(events[0] == {"type": "ready"}, f"{label}: Ready must occur once, first")
    final = events[-1]
    require(isinstance(final, dict) and set(final) == {"type", "text", "finish_reason", "tokens"}, f"{label}: malformed Final")
    require(final == {"type": "final", "text": expected_text, "finish_reason": "STOP", "tokens": generated}, f"{label}: Final differs")
    deltas = events[1:-1]
    require(all(isinstance(e, dict) and set(e) == {"type", "text"} and e["type"] == "delta" and
                isinstance(e["text"], str) and e["text"] for e in deltas), f"{label}: invalid Delta")
    require("".join(e["text"] for e in deltas) == expected_text, f"{label}: concatenated text differs")
    return len(deltas)


def check(report):
    require(report.get("status") == "passed" and report.get("scope") == "native_runtime_abi_independent_consumer", "Not a passing ABI execution")
    for field in ("sealed_cartridge", "full_spec_conformance", "performance_benchmark", "task_quality_qualified"):
        require(report.get(field) is False, f"Overstated scope: {field}")
    require(report.get("consumer", {}).get("sha256") == sha((ROOT / "scripts/consume_native_cartridge.py").read_bytes()), "Consumer source identity changed")
    require(report.get("preflight", {}).get("status") == 0 and report["preflight"].get("ok") == 1, "Preflight failed")
    for field in ("one_cartridge_open_for_all_sessions", "result_survives_session_end", "result_survives_cartridge_close"):
        require(report.get(field) is True, f"Missing lifetime assertion: {field}")
    require(report.get("fresh_session_after_cancel") == report.get("cancel_before_flush") == "passed", "Cancellation lifecycle coverage missing")
    cancel = report.get("cross_thread_cancel", {})
    require(cancel.get("status") == "passed" and cancel.get("terminal_status") == -6, "Cancellation did not return ABI failure status")
    call = cancel.get("cancel_call", {})
    require(call.get("status") == 0 and call["end"] >= call["start"], "Cancel call failed")
    require(any(p["start"] <= call["start"] <= p["end"] and p["status"] == -6 for p in cancel.get("owner_pull_calls", [])), "No saved in-flight cancellation overlap")
    negatives = report.get("negative_cases", [])
    expected_negatives = {"malformed-json", "invalid-utf8", "model-knob", "unsupported-role", "unsupported-sampling", "duplicate-key"}
    require(len(negatives) == len(expected_negatives) and {x.get("id") for x in negatives} == expected_negatives, "Negative-case coverage differs")
    require(all(x.get("status") == -1 and isinstance(x.get("detail"), str) and x["detail"] for x in negatives), "Invalid input reported success")
    cases = report.get("cases", [])
    require([x.get("id") for x in cases] == ["arithmetic", "system", "multiturn", "after-cancel"], "Reference/recovery coverage differs")
    refs = references()
    results = []
    for case in cases:
        label = case["id"]
        ref_label = "arithmetic" if label == "after-cancel" else label
        reference = refs[ref_label]
        path = ROOT / f"samples/requests/{ref_label}.json"
        source_bytes = path.read_bytes()
        require(sha(source_bytes) == REQUEST_SHA256[ref_label], f"{label}: original request changed")
        original = json.loads(source_bytes)
        request = {"messages": original["messages"], "max_output_tokens": original["max_new_tokens"]}
        require(request["messages"] == reference["messages"] and case.get("request") == request, f"{label}: request differs")
        request_bytes = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
        require(case.get("request_sha256") == sha(request_bytes), f"{label}: request identity differs")
        if label != "after-cancel":
            require(case.get("original_request", {}).get("sha256") == sha(source_bytes), f"{label}: source request identity differs")
        require(case.get("borrowed_input_overwritten_after_each_push") is True and case.get("drained_twice") is True, f"{label}: ownership/drain coverage missing")
        events = case.get("events", [])
        require(len(events) >= 3 and all(set(e) == {"kind", "payload"} for e in events), f"{label}: missing native events")
        require([e["kind"] for e in events] == [1] * (len(events) - 1) + [2], f"{label}: native result kinds differ")
        require(events[0]["payload"] == {"type": "ready", "prompt_ids": reference["prompt_ids"]}, f"{label}: prompt IDs differ")
        profile_events = [{"type": "ready"}] + [e["payload"] for e in events[1:]]
        generated = reference["expected_generated_ids"]
        expected_text = reference["expected_decoded_completion_skip_special_tokens"]
        count = check_events(profile_events, generated, expected_text, label)
        results.append({"id": label, "prompt_tokens": len(reference["prompt_ids"]),
                        "generated_tokens": len(generated), "text": expected_text,
                        "delta_events": count, "exact_reference_trace": True})
    return {"status": "passed", "scope": "saved_native_abi_reference_validation", "cases": results,
            "unique_reference_conversations": 3, "recovery_replays": 1,
            "prompt_token_comparisons": sum(r["prompt_tokens"] for r in results),
            "generated_token_comparisons": sum(r["generated_tokens"] for r in results),
            "native_streaming_and_ownership": True, "in_flight_cross_thread_cancellation": True,
            "performance_benchmark": False, "task_quality_qualified": False, "sealed_cartridge": False,
            "full_spec_conformance": False,
            "limits": ["Saved observations require separate source/artifact and deployment provenance.",
                       "This is a custom ABI evaluation harness, not upstream ctg-conform.",
                       "No full-logit, long-context, sustained-load or accelerator qualification is added.",
                       "The matching multi-turn answer 3 remains incorrect."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output must be new")
    raw = args.report.read_bytes()
    try:
        report = check(json.loads(raw))
    except (ValidationError, ValueError, TypeError, KeyError, IndexError, OSError) as error:
        report = {"status": "failed", "scope": "saved_native_abi_reference_validation", "reason": str(error)}
    report["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["input"] = {"file": str(args.report), "sha256": sha(raw)}
    report["validator_sources"] = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in
        (Path(__file__).resolve(), ROOT / "samples/golden-transcripts.json")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
