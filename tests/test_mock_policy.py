import importlib.util
import unittest
from pathlib import Path

import numpy as np


def _load_mock_policy_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "serve_mock_policy.py"
    spec = importlib.util.spec_from_file_location("serve_mock_policy", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HoldCurrentPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_mock_policy_module()

    def test_repeats_current_state(self) -> None:
        state = np.linspace(-1.0, 1.0, 14, dtype=np.float32)
        result = self.module.HoldCurrentPolicy(horizon=3).infer({"observation.state": state})

        self.assertEqual(result["action"].shape, (3, 14))
        np.testing.assert_array_equal(result["action"], np.repeat(state[None, :], 3, axis=0))

    def test_rejects_invalid_state(self) -> None:
        policy = self.module.HoldCurrentPolicy()
        with self.assertRaises(KeyError):
            policy.infer({})
        with self.assertRaises(ValueError):
            policy.infer({"observation.state": np.zeros(13)})
        with self.assertRaises(ValueError):
            policy.infer({"observation.state": np.full(14, np.nan)})

    def test_rejects_invalid_horizon(self) -> None:
        with self.assertRaises(ValueError):
            self.module.HoldCurrentPolicy(horizon=0)


if __name__ == "__main__":
    unittest.main()
