"""Load-identity guards for the independent ABI consumer; no library/model execution."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import consume_native_cartridge as consumer


class NativeConsumerIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.pack = Path(self.temporary.name).resolve()
        self.library = self.pack / "artifacts/runtime/libtest.so"
        self.library.parent.mkdir(parents=True)
        self.library.write_bytes(b"identity fixture, not a shared library")
        self.descriptor = {"id": "identity-test", "version": "0.1.0-eval", "io": {"mode": "streaming"}}
        self.write_pack(consumer.canonical(self.descriptor))

    def write_pack(self, raw):
        (self.pack / "descriptor.json").write_bytes(raw)
        self.manifest = {"kind": "cartridge", "digest_alg": "sha256", "id": self.descriptor["id"],
                         "version": self.descriptor["version"],
                         "descriptor_digest": "sha256:" + hashlib.sha256(consumer.canonical(self.descriptor)).hexdigest(),
                         "artifacts": []}
        for role, path in (("descriptor", self.pack / "descriptor.json"), ("runtime", self.library)):
            record = consumer.identity(path)
            self.manifest["artifacts"].append({"role": role, "path": path.relative_to(self.pack).as_posix(),
                                               "digest": "sha256:" + record["sha256"], "size": record["size_bytes"]})
        self.save_manifest()

    def save_manifest(self):
        (self.pack / "manifest.json").write_bytes(consumer.canonical(self.manifest))

    def check(self):
        return consumer.check_package_identity(self.pack, self.library)

    def test_original_and_pretty_descriptor_packs_preserve_identity(self):
        for raw in (consumer.canonical(self.descriptor), json.dumps(self.descriptor, indent=2).encode() + b"\n"):
            self.write_pack(raw)
            canonical, record = self.check()
            self.assertEqual(consumer.canonical(self.descriptor), canonical)
            self.assertTrue(consumer.check_embedded_descriptor(canonical, canonical)["embedded_descriptor_canonical_equivalent"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), record["descriptor"]["sha256"])
            self.assertFalse(record["package_identity"]["signature_verification"])
            self.assertFalse(record["package_identity"]["full_artifact_inventory_checked"])

    def test_raw_descriptor_tamper_fails_even_when_json_is_equivalent(self):
        (self.pack / "descriptor.json").write_bytes(consumer.canonical(self.descriptor) + b"\n")
        with self.assertRaisesRegex(AssertionError, "raw artifact digest"):
            self.check()

    def test_canonical_digest_and_manifest_identity_must_match(self):
        for field, value in (("descriptor_digest", "sha256:" + "0" * 64), ("id", "other"), ("version", "0.2.0")):
            with self.subTest(field=field):
                before = self.manifest[field]
                self.manifest[field] = value
                self.save_manifest()
                with self.assertRaises(AssertionError): self.check()
                self.manifest[field] = before

    def test_loaded_runtime_path_bytes_and_size_are_manifest_bound(self):
        for field, value in (("path", "artifacts/runtime/different.so"), ("digest", "sha256:" + "0" * 64), ("size", 0)):
            with self.subTest(field=field):
                before = self.manifest["artifacts"][1][field]
                self.manifest["artifacts"][1][field] = value
                self.save_manifest()
                with self.assertRaises(AssertionError): self.check()
                self.manifest["artifacts"][1][field] = before

    def test_ambiguous_inventory_and_duplicate_json_are_rejected(self):
        self.manifest["artifacts"].append(dict(self.manifest["artifacts"][1]))
        self.save_manifest()
        with self.assertRaisesRegex(AssertionError, "Duplicate"): self.check()
        self.manifest["artifacts"].pop()
        self.manifest["artifacts"][1]["role"] = "other"
        self.save_manifest()
        with self.assertRaisesRegex(AssertionError, "one native runtime"): self.check()
        (self.pack / "manifest.json").write_bytes(b'{"id":"a","id":"a"}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"): self.check()

    def test_abi_getter_must_match_canonical_identity_without_duplicate_keys(self):
        expected = consumer.canonical(self.descriptor)
        for embedded in (consumer.canonical(dict(self.descriptor, id="different")), expected + b"\n"):
            with self.subTest(embedded=embedded):
                with self.assertRaises(AssertionError): consumer.check_embedded_descriptor(embedded, expected)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            consumer.check_embedded_descriptor(expected[:-1] + b',"id":"identity-test"}', expected)


if __name__ == "__main__":
    unittest.main()
