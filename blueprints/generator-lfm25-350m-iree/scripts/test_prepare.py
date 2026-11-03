import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch
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

    def wheel(self, entries):
        path = self.root / 'test.whl'
        with zipfile.ZipFile(path, 'w') as archive:
            for name, content in entries: archive.writestr(name, content)
        pins = self.json('pins.json', {'python':'3.12','platform':'linux-aarch64',
            'wheels':[{'source':'test','file':path.name,**prepare.identity(path)}]})
        return path, pins

    def test_wheels_are_hash_verified_and_extracted_without_installers(self):
        _, pins = self.wheel([('namespace/pkg/module.py', b'value = 1\n'),('package.dist-info/LICENSE',b'MIT')])
        prepare.extract_wheels(self.root,pins,self.root/'out/packages')
        self.assertEqual(b'value = 1\n',(self.root/'out/packages/namespace/pkg/module.py').read_bytes())
        self.assertTrue((self.root/'out/python-wheels.json').is_file())

    def test_wheel_escape_or_collision_never_overwrites_files(self):
        _,pins=self.wheel([('../escape',b'bad')])
        with self.assertRaises(ValueError):prepare.extract_wheels(self.root,pins,self.root/'out')
        self.assertFalse((self.root/'escape').exists())
        _,pins=self.wheel([('module.py',b'new')]); (self.root/'out').mkdir(exist_ok=True)
        existing=self.root/'out/module.py';existing.write_bytes(b'old')
        with self.assertRaises(FileExistsError):prepare.extract_wheels(self.root,pins,self.root/'out')
        self.assertEqual(b'old',existing.read_bytes())

    def test_wheel_hash_or_wrong_python_abi_is_rejected(self):
        wheel,pins=self.wheel([('module.py',b'x')]);wheel.write_bytes(wheel.read_bytes()+b'changed')
        with self.assertRaises(ValueError):prepare.extract_wheels(self.root,pins,self.root/'out')
        data=prepare.read(pins);data['python']='3.11';pins.write_text(json.dumps(data))
        with self.assertRaises(ValueError):prepare.extract_wheels(self.root,pins,self.root/'out')

    def test_private_settings_reject_missing_or_cpu_device_selection(self):
        data={'model_directory':str(self.root),'wheel_directory':str(self.root),'device_uri':'vulkan://test'}
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

    def test_worker_uses_package_backend_context64_and_external_python(self):
        value=prepare.worker_configuration('vulkan://explicit')
        self.assertEqual('iree-vulkan',value['backend']);self.assertEqual(64,value['context_window_tokens'])
        self.assertEqual('LFM2_PYTHON_EXECUTABLE',value['python_executable_env']);self.assertNotIn('model_dir',value)
        self.assertEqual(6,len(value['metadata_sha256']))


if __name__ == '__main__': unittest.main()
