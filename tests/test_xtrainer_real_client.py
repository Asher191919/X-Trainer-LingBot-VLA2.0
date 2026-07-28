import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


def _load_client_module():
    websocket_module = types.ModuleType("deploy.websocket_client_policy")
    websocket_module.WebsocketClientPolicy = object
    environment_module = types.ModuleType("deploy.xtrainer_real")
    environment_module.XTrainerRealEnvironment = object
    sys.modules[websocket_module.__name__] = websocket_module
    sys.modules[environment_module.__name__] = environment_module

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_xtrainer_real.py"
    spec = importlib.util.spec_from_file_location("run_xtrainer_real", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ActionChunkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = _load_client_module()

    def test_truncates_chunk_to_horizon(self) -> None:
        actions = np.zeros((50, 14), dtype=np.float32)
        self.assertEqual(self.client._extract_action_chunk({"action": actions}, 25).shape, (25, 14))

    def test_promotes_single_action(self) -> None:
        action = np.zeros(14, dtype=np.float32)
        self.assertEqual(self.client._extract_action_chunk({"action": action}, 25).shape, (1, 14))

    def test_rejects_unsafe_responses(self) -> None:
        invalid_actions = (
            np.zeros((2, 13)),
            np.empty((0, 14)),
            np.full((1, 14), np.nan),
        )
        for actions in invalid_actions:
            with self.subTest(shape=actions.shape):
                with self.assertRaises(ValueError):
                    self.client._extract_action_chunk({"action": actions}, 25)

        with self.assertRaises(KeyError):
            self.client._extract_action_chunk({}, 25)


if __name__ == "__main__":
    unittest.main()