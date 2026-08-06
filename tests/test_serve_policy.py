import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _load_server_module():
    policy_module = types.ModuleType("deploy.lingbot_vla_v2_policy")
    policy_module.LingbotVLAv2Server = object
    server_module = types.ModuleType("deploy.websocket_policy_server")
    server_module.WebsocketPolicyServer = object
    sys.modules[policy_module.__name__] = policy_module
    sys.modules[server_module.__name__] = server_module

    script = Path(__file__).resolve().parents[1] / "scripts" / "serve_policy.py"
    spec = importlib.util.spec_from_file_location("serve_policy", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ServerMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = _load_server_module()

    def test_xtrainer_metadata_includes_reset_pose(self) -> None:
        metadata = self.server._build_server_metadata("xtrainer")

        self.assertEqual(metadata["model_type"], "lingbot-vla-2.0")
        self.assertEqual(metadata["robot"], "xtrainer")
        self.assertEqual(metadata["reset_pose"], list(self.server.XTRAINER_RESET_POSE))
        self.assertEqual(len(metadata["reset_pose"]), 14)

    def test_other_robots_do_not_receive_xtrainer_pose(self) -> None:
        metadata = self.server._build_server_metadata("robotwin")

        self.assertNotIn("reset_pose", metadata)


if __name__ == "__main__":
    unittest.main()
