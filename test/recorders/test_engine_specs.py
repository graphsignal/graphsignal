import unittest

from graphsignal.launchers import sglang_launcher, vllm_launcher
from graphsignal.recorders.engine_specs import engine_spec


class EngineSpecTest(unittest.TestCase):
    def test_vllm(self):
        spec = engine_spec('vllm')
        self.assertEqual(spec.name, 'vllm')
        self.assertEqual(spec.info_path, '/version')
        self.assertEqual(spec.config_line_marker, ' with config: ')

    def test_sglang(self):
        spec = engine_spec('sglang')
        self.assertEqual(spec.name, 'sglang')
        self.assertEqual(spec.info_path, '/get_server_info')
        self.assertIsNone(spec.config_line_marker)

    def test_none_and_unlisted(self):
        self.assertIsNone(engine_spec(None))
        self.assertIsNone(engine_spec(''))
        self.assertIsNone(engine_spec('trtllm'))

    def test_every_launcher_names_a_listed_engine(self):
        # A launcher's key with no spec would silently drop the engine resource.
        for launcher in (vllm_launcher, sglang_launcher):
            self.assertEqual(engine_spec(launcher.ENGINE).name, launcher.ENGINE)
