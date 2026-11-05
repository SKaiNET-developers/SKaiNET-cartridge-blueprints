import json
from pathlib import Path
import tempfile
import unittest
import prepare


class GpuPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def json(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def runtime_build(self, **overrides):
        directory = self.root / 'iree-out'
        (directory / 'licenses').mkdir(parents=True, exist_ok=True)
        (directory / 'licenses/LICENSE.iree').write_bytes(b'Apache-2.0 WITH LLVM-exception text')
        (directory / 'licenses/LICENSE.flatcc').write_bytes(b'Apache-2.0 text')
        files = []
        for name in (prepare.IREE_RUN_MODULE, 'iree-benchmark-module-linux-arm64'):
            tool = directory / name
            tool.write_bytes(b'aarch64-elf:' + name.encode())
            files.append({'file': name, **{'sha256': prepare.identity(tool)['sha256'],
                                           'size_bytes': prepare.identity(tool)['size_bytes']}})
        report = {'schema_version': 1, 'builder_image': 'skainet/iree-linux-arm64',
                  'iree_version': prepare.IREE_VERSION, 'iree_git_commit': prepare.IREE_SOURCE_COMMIT,
                  'target_triple': 'aarch64-linux-gnu', 'vulkan': True,
                  'hal_drivers': ['local-task', 'local-sync', 'vulkan'],
                  'license': prepare.IREE_LICENSE, 'license_files': 'licenses/', 'files': files}
        report.update(overrides)
        (directory / prepare.IREE_BUILD_REPORT).write_text(json.dumps(report))
        return directory

    def test_native_runtime_is_hash_verified_and_staged_with_licenses(self):
        directory = self.runtime_build()
        prepare.stage_iree_runtime(directory, self.root / 'out')
        staged = self.root / 'out' / prepare.IREE_RUN_MODULE
        self.assertTrue(staged.is_file())
        self.assertEqual(b'Apache-2.0 WITH LLVM-exception text', (self.root / 'out/licenses/LICENSE.iree').read_bytes())
        self.assertTrue((self.root / 'out' / prepare.IREE_BUILD_REPORT).is_file())

    def test_native_runtime_tamper_or_overwrite_is_rejected(self):
        directory = self.runtime_build()
        (directory / prepare.IREE_RUN_MODULE).write_bytes(b'tampered')
        with self.assertRaises(ValueError): prepare.stage_iree_runtime(directory, self.root / 'out')
        directory = self.runtime_build()
        existing = self.root / 'out' / prepare.IREE_RUN_MODULE
        existing.parent.mkdir(parents=True, exist_ok=True); existing.write_bytes(b'old')
        with self.assertRaises(FileExistsError): prepare.stage_iree_runtime(directory, self.root / 'out')
        self.assertEqual(b'old', existing.read_bytes())

    def test_native_runtime_report_pins_are_enforced(self):
        for overrides in ({'iree_version': '3.12.0'}, {'iree_git_commit': 'f' * 40},
                          {'target_triple': 'aarch64-linux-android'}, {'vulkan': False},
                          {'builder_image': 'other/image'}, {'files': []}):
            directory = self.runtime_build(**overrides)
            with self.assertRaises(ValueError): prepare.stage_iree_runtime(directory, self.root / ('out-' + next(iter(overrides))))
        directory = self.runtime_build()
        (directory / 'licenses/LICENSE.iree').unlink()
        with self.assertRaises(ValueError): prepare.stage_iree_runtime(directory, self.root / 'out-license')

    def test_private_settings_reject_missing_or_cpu_device_selection(self):
        data={'model_directory':str(self.root),'device_uri':'vulkan://test'}
        path=self.json('settings.json',data)
        self.assertEqual('vulkan://test',prepare.settings(path)['device_uri'])
        for uri in ('','vulkan://','local-task://0'):
            data['device_uri']=uri;path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):prepare.settings(path)
        data['device_uri']='vulkan://test';data['untrusted_model']='x';path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):prepare.settings(path)

    def test_descriptor_requires_exact_encoding_and_compiled_capacity(self):
        value={'license':'NOASSERTION','performance':prepare.PERFORMANCE,'target':{'accelerator':'vulkan'},'attributes':{'context_window_tokens':64}}
        raw=self.json('descriptor.json',value);compact=self.root/'canonical.json';compact.write_bytes(prepare.canonical(value))
        prepare.descriptor_pair(raw,compact)
        value['attributes']['context_window_tokens']=2048;raw.write_text(json.dumps(value));compact.write_bytes(prepare.canonical(value))
        with self.assertRaises(ValueError):prepare.descriptor_pair(raw,compact)

    def test_partial_or_cpu_bundle_is_not_a_full_generation_candidate(self):
        self.json('runtime-manifest.json',{'backend':'vulkan-spirv','dtype':'float32','capacity':64,'function_name':'attention','outputs':[]})
        with self.assertRaises(ValueError):prepare.verify_bundle(self.root)
        self.json('runtime-manifest.json',{'backend':'llvm-cpu','dtype':'float32','capacity':64})
        with self.assertRaises(ValueError):prepare.verify_bundle(self.root)

    def test_compilation_provenance_and_parameter_bytes_are_rechecked(self):
        contract={'backend':'vulkan-spirv','dtype':'float32','capacity':64,'function_name':'lfm2_cached_step',
                  'outputs':[{'name':str(i)} for i in range(23)],'embedding_shard_rows':[8192]*8}
        self.json('generation-contract.json',contract)
        exported={'provenance':{'checkpoint_sha256':prepare.MODEL_FILES['model.safetensors'][1]}}
        self.json('source-export-manifest.json',exported)
        (self.root/'model.vmfb').write_bytes(b'synthetic-module');(self.root/'model.irpa').write_bytes(b'synthetic-parameters')
        (self.root/'compiler-entrypoint.sh').write_bytes(b'wrapper');(self.root/'compile-driver.py').write_bytes(b'driver')
        runtime=dict(contract)
        for key,name in [('module','model.vmfb'),('parameters','model.irpa')]:runtime[key]={'file':name,**prepare.identity(self.root/name)}
        runtime.update(export_manifest_sha256=prepare.sha(self.root/'source-export-manifest.json'),
                       source_generation_contract_sha256=prepare.sha(self.root/'generation-contract.json'))
        (self.root/'verify_vulkan_ranges.py').write_bytes(b'auditor')
        audit=self.json('vulkan-range-audit.json',{'status':'passed','findings':[],'unresolved':[],
            'inputs':{'vmfb':prepare.identity(self.root/'model.vmfb'),'auditor':prepare.identity(self.root/'verify_vulkan_ranges.py'),'dumps':[]},
            'policy':{'max_binding_bytes':67108864,'max_allocation_bytes':8589934592,'max_workgroup_counts':[65535]*3}})
        runtime['vulkan_range_audit']={'file':audit.name,**prepare.identity(audit)}
        self.json('runtime-manifest.json',runtime)
        self.json('compilation-report.json',{'status':'compiled_not_executed','cpu_model_fallback':False,
            'runtime_manifest_sha256':prepare.sha(self.root/'runtime-manifest.json'),
            'compiler_wrapper_sha256':prepare.sha(self.root/'compiler-entrypoint.sh'),
            'compiler_driver_sha256':prepare.sha(self.root/'compile-driver.py'),
            'range_auditor_sha256':prepare.sha(self.root/'verify_vulkan_ranges.py'),'vulkan_range_audit':runtime['vulkan_range_audit'],
            'export_manifest_sha256':prepare.sha(self.root/'source-export-manifest.json'),
            'generation_contract_sha256':prepare.sha(self.root/'generation-contract.json')})
        self.assertEqual(64,prepare.verify_bundle(self.root)['capacity'])
        (self.root/'model.irpa').write_bytes(b'tampered')
        with self.assertRaises(ValueError):prepare.verify_bundle(self.root)

    def test_claim_preserves_failed_run_and_build_products_are_copied(self):
        out=self.root/'run';prepare.claim_directory(out)
        with self.assertRaises(FileExistsError):prepare.claim_directory(out)
        source=self.root/'worker.jar';source.write_bytes(b'original')
        prepare.place(source,out/'worker.jar',prepare.identity(source));source.write_bytes(b'rebuilt')
        self.assertEqual(b'original',(out/'worker.jar').read_bytes())

    def test_worker_uses_native_backend_context64_and_no_python(self):
        value=prepare.worker_configuration('vulkan://explicit')
        self.assertEqual('iree-native-vulkan',value['backend']);self.assertEqual(64,value['context_window_tokens'])
        self.assertEqual('artifacts/iree/'+prepare.IREE_RUN_MODULE,value['iree_run_module'])
        self.assertNotIn('python_executable_env',value);self.assertNotIn('python_package_dir',value)
        self.assertNotIn('model_dir',value)
        self.assertEqual(6,len(value['metadata_sha256']))


if __name__ == '__main__': unittest.main()
