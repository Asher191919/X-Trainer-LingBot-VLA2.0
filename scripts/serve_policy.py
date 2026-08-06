import argparse
import logging
import socket
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
from deploy.websocket_policy_server import WebsocketPolicyServer


XTRAINER_RESET_POSE = (
    -1.57,
    0.0,
    -1.57,
    0.0,
    1.57,
    1.57,
    1.0,
    1.57,
    0.0,
    1.57,
    0.0,
    -1.57,
    -1.57,
    1.0,
)


def _build_server_metadata(robot: str) -> dict:
    metadata = {"model_type": "lingbot-vla-2.0", "robot": robot}
    if robot == "xtrainer":
        metadata["reset_pose"] = list(XTRAINER_RESET_POSE)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a LingBot-VLA 2.0 checkpoint over WebSocket")
    parser.add_argument("--model-path", "--model_path", required=True, help="Checkpoint directory containing safetensors")
    parser.add_argument(
        "--robot",
        "--robo-name",
        "--robo_name",
        dest="robot",
        default="xtrainer",
        help="Robot config name under configs/robot_configs (default: xtrainer)",
    )
    parser.add_argument("--norm-path", "--norm_path", default=None, help="Override normalization statistics path")
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket listen address")
    parser.add_argument("--port", type=int, default=8000, help="WebSocket listen port")
    parser.add_argument("--use-length", "--use_length", type=int, default=50, help="Actions consumed per model call")
    parser.add_argument(
        "--step-mode",
        action="store_true",
        help="Return one action per request instead of the full action chunk",
    )
    parser.add_argument("--fp32", action="store_true", help="Use float32 instead of bfloat16")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robot_config = PROJECT_ROOT / "configs" / "robot_configs" / f"{args.robot}.yaml"
    if not robot_config.is_file():
        raise FileNotFoundError(f"Robot config not found: {robot_config}")
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.model_path}")
    if args.step_mode and args.use_length <= 0:
        raise ValueError("--use-length must be greater than zero in step mode")

    policy = LingbotVLAv2Server(
        path_to_pi_model=args.model_path,
        robot_norm_path=args.norm_path,
        use_length=args.use_length,
        chunk_ret=not args.step_mode,
        use_bf16=not args.fp32,
        use_fp32=args.fp32,
        use_compile=args.compile,
    )
    policy.reset(args.robot)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=_build_server_metadata(args.robot),
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
