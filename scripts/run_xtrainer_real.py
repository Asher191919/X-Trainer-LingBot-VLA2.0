import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.websocket_client_policy import WebsocketClientPolicy
from deploy.xtrainer_real import XTrainerRealEnvironment


def _servo_range(value: str) -> tuple[int, int]:
    try:
        minimum, maximum = (int(part) for part in value.split(",", maxsplit=1))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected MIN,MAX") from exc
    if minimum >= maximum:
        raise argparse.ArgumentTypeError("MIN must be less than MAX")
    return minimum, maximum


def _extract_action_chunk(response: dict, action_horizon: int) -> np.ndarray:
    if "action" not in response:
        raise KeyError(f"Missing 'action' in policy response: {tuple(response.keys())}")
    actions = np.asarray(response["action"], dtype=np.float64)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected action shape (H, 14), got {actions.shape}")
    if actions.shape[0] == 0:
        raise ValueError("Policy returned an empty action chunk")
    if not np.all(np.isfinite(actions)):
        raise ValueError("Policy returned non-finite actions")
    return actions[:action_horizon]


def _rate_limit_action(action: np.ndarray, last_action: np.ndarray | None, max_delta_per_step: float) -> np.ndarray:
    target = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    if last_action is None or max_delta_per_step <= 0:
        return target
    previous = np.asarray(last_action, dtype=np.float64).reshape(-1)
    return previous + np.clip(target - previous, -max_delta_per_step, max_delta_per_step)


def _smooth_chunk_boundary(
    chunk: np.ndarray,
    last_action: np.ndarray | None,
    *,
    max_switch_delta: float,
    blend_steps: int,
) -> np.ndarray:
    smoothed = np.asarray(chunk, dtype=np.float64).copy()
    if last_action is None or len(smoothed) == 0 or max_switch_delta <= 0:
        return smoothed

    previous = np.asarray(last_action, dtype=np.float64).reshape(-1)
    delta = float(np.max(np.abs(smoothed[0] - previous)))
    if delta <= max_switch_delta:
        return smoothed

    steps = min(max(blend_steps, 1), len(smoothed))
    for index in range(steps):
        weight = float(index + 1) / float(steps + 1)
        smoothed[index] = previous + weight * (smoothed[index] - previous)
    logging.warning(
        "Blended action chunk boundary: max delta %.4f exceeded threshold %.4f over %d steps",
        delta,
        max_switch_delta,
        steps,
    )
    return smoothed


def _infer_action_chunk(policy: WebsocketClientPolicy, observation: dict, action_horizon: int) -> np.ndarray:
    response = policy.infer(observation)
    return _extract_action_chunk(response, action_horizon)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LingBot-VLA 2.0 on an X-Trainer robot")
    parser.add_argument("--host", required=True, help="LingBot policy server address")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", default="pick up the object")
    parser.add_argument("--action-horizon", type=int, default=25)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--left-robot-ip", default="192.168.5.1")
    parser.add_argument("--right-robot-ip", default="192.168.5.2")
    parser.add_argument("--left-gripper-port", default="/dev/ttyUSB1")
    parser.add_argument("--right-gripper-port", default="/dev/ttyUSB0")
    parser.add_argument("--left-gripper-id", type=int, default=21)
    parser.add_argument("--right-gripper-id", type=int, default=22)
    parser.add_argument("--left-gripper-servo-pos", type=_servo_range, default=(2048, 3052), metavar="MIN,MAX")
    parser.add_argument("--right-gripper-servo-pos", type=_servo_range, default=(2048, 3052), metavar="MIN,MAX")
    parser.add_argument("--camera-top-serial", required=True)
    parser.add_argument("--camera-left-wrist-serial", required=True)
    parser.add_argument("--camera-right-wrist-serial", required=True)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--render-height", type=int, default=224)
    parser.add_argument("--render-width", type=int, default=224)
    parser.add_argument("--max-joint-delta", type=float, default=0.17)
    parser.add_argument("--ramp-step", type=float, default=0.01)
    parser.add_argument("--ramp-max-steps", type=int, default=100)
    parser.add_argument("--gripper-update-threshold", type=float, default=0.02)
    parser.add_argument("--servo-step-limit", type=float, default=0.9)
    parser.add_argument(
        "--prefetch-remaining",
        type=int,
        default=8,
        help="Start background inference when this many actions remain in the current chunk; 0 disables prefetch",
    )
    parser.add_argument(
        "--switch-blend-steps",
        type=int,
        default=5,
        help="Number of actions used to blend across a large chunk-boundary jump",
    )
    parser.add_argument(
        "--max-switch-delta",
        type=float,
        default=0.12,
        help="Blend the next chunk if its first action differs from the last sent action by more than this",
    )
    parser.add_argument(
        "--max-delta-per-step",
        type=float,
        default=0.0,
        help="Optional final per-control-step action delta limit; <=0 disables this client-side limiter",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "port": args.port,
        "action_horizon": args.action_horizon,
        "control_hz": args.control_hz,
        "max_steps": args.max_steps,
        "camera_fps": args.camera_fps,
        "render_height": args.render_height,
        "render_width": args.render_width,
        "ramp_step": args.ramp_step,
        "ramp_max_steps": args.ramp_max_steps,
        "servo_step_limit": args.servo_step_limit,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"Expected positive values for: {', '.join(invalid)}")
    if args.max_joint_delta < 0 or args.gripper_update_threshold < 0:
        raise ValueError("Action thresholds must be non-negative")
    non_negative_values = {
        "prefetch_remaining": args.prefetch_remaining,
        "switch_blend_steps": args.switch_blend_steps,
        "max_switch_delta": args.max_switch_delta,
        "max_delta_per_step": args.max_delta_per_step,
    }
    invalid = [name for name, value in non_negative_values.items() if value < 0]
    if invalid:
        raise ValueError(f"Expected non-negative values for: {', '.join(invalid)}")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    if metadata.get("model_type") not in (None, "lingbot-vla-2.0"):
        raise RuntimeError(f"Unexpected model type: {metadata.get('model_type')}")
    if metadata.get("robot") not in (None, "xtrainer"):
        raise RuntimeError(f"Unexpected robot config: {metadata.get('robot')}")
    if metadata.get("mock_policy"):
        logging.warning("Connected to a mock hold-current policy; no learned actions will be executed")

    environment = XTrainerRealEnvironment(
        left_robot_ip=args.left_robot_ip,
        right_robot_ip=args.right_robot_ip,
        left_gripper_port=args.left_gripper_port,
        right_gripper_port=args.right_gripper_port,
        left_gripper_id=args.left_gripper_id,
        right_gripper_id=args.right_gripper_id,
        left_gripper_servo_pos=args.left_gripper_servo_pos,
        right_gripper_servo_pos=args.right_gripper_servo_pos,
        camera_top_serial=args.camera_top_serial,
        camera_left_wrist_serial=args.camera_left_wrist_serial,
        camera_right_wrist_serial=args.camera_right_wrist_serial,
        camera_fps=args.camera_fps,
        render_height=args.render_height,
        render_width=args.render_width,
        task=args.task,
        reset_pose=metadata.get("reset_pose"),
        max_joint_delta=args.max_joint_delta,
        ramp_step=args.ramp_step,
        ramp_max_steps=args.ramp_max_steps,
        gripper_update_threshold=args.gripper_update_threshold,
        servo_step_limit=args.servo_step_limit,
    )

    action_chunk = np.empty((0, 14), dtype=np.float64)
    action_index = 0
    next_chunk: np.ndarray | None = None
    prefetch_future: Future | None = None
    last_sent_action: np.ndarray | None = None
    period = 1.0 / args.control_hz
    deadline = time.monotonic()
    try:
        environment.reset()
        with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
            for step in range(args.max_steps):
                if prefetch_future is not None and prefetch_future.done():
                    next_chunk = prefetch_future.result()
                    prefetch_future = None
                    logging.info("Prefetched %d actions at step %d", len(next_chunk), step)

                if action_index >= len(action_chunk):
                    if next_chunk is not None:
                        action_chunk = _smooth_chunk_boundary(
                            next_chunk,
                            last_sent_action,
                            max_switch_delta=args.max_switch_delta,
                            blend_steps=args.switch_blend_steps,
                        )
                        next_chunk = None
                    else:
                        action_chunk = _infer_action_chunk(policy, environment.get_observation(), args.action_horizon)
                    action_index = 0
                    logging.info("Received %d actions at step %d", len(action_chunk), step)

                if action_index < len(action_chunk):
                    remaining_actions = len(action_chunk) - action_index
                    if (
                        args.prefetch_remaining > 0
                        and remaining_actions <= args.prefetch_remaining
                        and prefetch_future is None
                        and next_chunk is None
                    ):
                        observation = environment.get_observation()
                        prefetch_future = prefetch_executor.submit(
                            _infer_action_chunk,
                            policy,
                            observation,
                            args.action_horizon,
                        )
                        logging.info(
                            "Started action prefetch at step %d with %d actions remaining",
                            step,
                            remaining_actions,
                        )

                    action = _rate_limit_action(action_chunk[action_index], last_sent_action, args.max_delta_per_step)
                    action_index += 1
                elif last_sent_action is not None and prefetch_future is not None:
                    logging.warning("Action chunk exhausted before prefetch finished; holding last action")
                    action = last_sent_action.copy()
                else:
                    action = _infer_action_chunk(policy, environment.get_observation(), args.action_horizon)[0]

                environment.apply_action(action)
                last_sent_action = action.copy()
                deadline += period
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                else:
                    deadline = time.monotonic()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        environment.close()
        policy.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
