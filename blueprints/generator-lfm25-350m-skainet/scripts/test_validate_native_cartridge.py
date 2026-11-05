import copy
import json
import unittest

from validate_native_cartridge import ValidationError
from validate_native_cartridge import ROOT, check, references, sha


class NativeReportValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        refs = references()
        cases = []
        for label in ("arithmetic", "system", "multiturn", "after-cancel"):
            key = "arithmetic" if label == "after-cancel" else label
            ref = refs[key]
            original = (ROOT / f"samples/requests/{key}.json").read_bytes()
            source = json.loads(original)
            request = {"messages": source["messages"], "max_output_tokens": source["max_new_tokens"]}
            text = ref["expected_decoded_completion_skip_special_tokens"]
            cases.append({"id": label, "request": request,
                          "request_sha256": sha(json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()),
                          "original_request": {"sha256": sha(original)},
                          "borrowed_input_overwritten_after_each_push": True, "drained_twice": True,
                          "events": [
                              {"kind": 1, "payload": {"type": "ready", "prompt_ids": ref["prompt_ids"]}},
                              {"kind": 1, "payload": {"type": "delta", "text": text}},
                              {"kind": 2, "payload": {"type": "final", "text": text, "finish_reason": "STOP",
                                                       "tokens": ref["expected_generated_ids"]}},
                          ]})
        cls.valid = {
            "status": "passed", "scope": "native_runtime_abi_independent_consumer",
            "sealed_cartridge": False, "full_spec_conformance": False,
            "performance_benchmark": False, "task_quality_qualified": False,
            "consumer": {"sha256": sha((ROOT / "scripts/consume_native_cartridge.py").read_bytes())},
            "preflight": {"status": 0, "ok": 1}, "one_cartridge_open_for_all_sessions": True,
            "result_survives_session_end": True, "result_survives_cartridge_close": True,
            "fresh_session_after_cancel": "passed", "cancel_before_flush": "passed",
            "cross_thread_cancel": {"status": "passed", "terminal_status": -6,
                                    "cancel_call": {"status": 0, "start": 1.1, "end": 1.2},
                                    "owner_pull_calls": [{"start": 1.0, "end": 1.3, "status": -6}]},
            "negative_cases": [{"id": key, "status": -1, "detail": "Rejected"} for key in
                               ("malformed-json", "invalid-utf8", "model-knob", "unsupported-role",
                                "unsupported-sampling", "duplicate-key")], "cases": cases,
        }

    def test_reference_and_recovery_trace(self):
        self.assertEqual(check(copy.deepcopy(self.valid))["unique_reference_conversations"], 3)

    def test_wrong_token_rejected_even_with_matching_text(self):
        report = copy.deepcopy(self.valid)
        report["cases"][0]["events"][-1]["payload"]["tokens"][0] += 1
        with self.assertRaises(ValidationError):
            check(report)

    def test_false_native_final_kind_rejected(self):
        report = copy.deepcopy(self.valid)
        report["cases"][0]["events"][-1]["kind"] = 1
        with self.assertRaises(ValidationError):
            check(report)

    def test_cancellation_outside_call_is_not_inflight(self):
        report = copy.deepcopy(self.valid)
        report["cross_thread_cancel"]["cancel_call"]["start"] = 1.4
        report["cross_thread_cancel"]["cancel_call"]["end"] = 1.5
        with self.assertRaises(ValidationError):
            check(report)

    def test_missing_recovery_or_ownership_rejected(self):
        for mutation in (lambda r: r["cases"].pop(), lambda r: r.update(result_survives_cartridge_close=False)):
            report = copy.deepcopy(self.valid)
            mutation(report)
            with self.assertRaises(ValidationError):
                check(report)

    def test_success_for_invalid_input_rejected(self):
        report = copy.deepcopy(self.valid)
        report["negative_cases"][0]["status"] = 0
        with self.assertRaises(ValidationError):
            check(report)


if __name__ == "__main__":
    unittest.main()
