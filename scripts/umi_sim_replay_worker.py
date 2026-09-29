# 将 UMI 官方权重在验证集窗口上的预测动作流式交给 ROS 2 仿真节点
import argparse
import contextlib
import json
import pathlib
import sys

import numpy as np
import torch

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from diffusion_policy.common.pytorch_util import dict_apply
from eval_offline_policy import build_eval_dataset, load_policy
from umi.common.pose_util import mat_to_pose, pose10d_to_mat


def parse_args():
    parser = argparse.ArgumentParser(description="输出 UMI 策略的仿真回放动作块")
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=ROOT_DIR / "data/pretrained/cup_wild_vit_l_1img.ckpt",
    )
    parser.add_argument(
        "--dataset",
        type=pathlib.Path,
        default=ROOT_DIR / "data/cup_in_the_wild.zarr.zip",
    )
    parser.add_argument("--cache-dir", type=pathlib.Path, default=ROOT_DIR / "data/cache")
    parser.add_argument("--episode-index", type=int, default=-1)
    parser.add_argument("--start-step", type=int, default=15)
    parser.add_argument("--max-chunks", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def emit_chunks(args):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"权重不存在：{args.checkpoint}")
    if not args.dataset.is_file():
        raise FileNotFoundError(f"数据集不存在：{args.dataset}")
    if args.max_chunks < 1:
        raise ValueError("max_chunks 必须大于 0")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    with contextlib.redirect_stdout(sys.stderr):
        cfg, policy = load_policy(
            str(args.checkpoint),
            device,
            dataset_path=str(args.dataset),
            cache_dir=str(args.cache_dir),
            skip_pretrained=True,
        )
        policy.num_inference_steps = cfg.policy.num_inference_steps
        policy.eval().to(device)
        dataset, eval_set, eval_mask = build_eval_dataset(cfg, split="val")

    episode_ids = np.flatnonzero(eval_mask)
    if episode_ids.size == 0:
        raise RuntimeError("验证集没有可回放的 episode")
    if args.episode_index < 0:
        episode_index = int(episode_ids[0])
    else:
        episode_index = args.episode_index
    if episode_index not in episode_ids:
        raise ValueError(f"episode {episode_index} 不在官方权重验证集内")

    episode_ends = dataset.replay_buffer.episode_ends[:]
    episode_start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
    selected_positions = []
    next_frame = episode_start + args.start_step
    action_steps = int(cfg.n_action_steps)
    frame_stride = action_steps * int(dataset.key_down_sample_steps["action"])

    for sample_position, entry in enumerate(eval_set.sampler.indices):
        current_frame, start_frame = int(entry[0]), int(entry[1])
        if start_frame != episode_start or current_frame < next_frame:
            continue
        selected_positions.append((sample_position, current_frame))
        next_frame = current_frame + frame_stride
        if len(selected_positions) >= args.max_chunks:
            break

    if not selected_positions:
        raise RuntimeError("所选 episode 没有可用的动作窗口")

    print(
        f"[加载] episode={episode_index}，回放动作块={len(selected_positions)}，device={device}",
        file=sys.stderr,
        flush=True,
    )

    for chunk_number, (sample_position, current_frame) in enumerate(selected_positions):
        sample = eval_set[sample_position]
        obs = dict_apply(
            sample["obs"],
            lambda value: value.unsqueeze(0).to(device),
        )
        with torch.no_grad():
            prediction = policy.predict_action(obs)["action"][0].float().cpu().numpy()

        action_chunk = prediction[:action_steps]
        relative_pose = mat_to_pose(pose10d_to_mat(action_chunk[:, :9]))
        robot_actions = np.concatenate(
            [relative_pose, action_chunk[:, 9:10]],
            axis=-1,
        )
        message = {
            "episode_index": episode_index,
            "frame_index": current_frame,
            "last_chunk": chunk_number + 1 == len(selected_positions),
            "actions": robot_actions.tolist(),
        }
        sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
        sys.stdout.flush()

        if chunk_number + 1 < len(selected_positions):
            if sys.stdin.readline().strip() != "continue":
                break


def main():
    args = parse_args()
    try:
        emit_chunks(args)
    except Exception as error:
        print(f"[错误] {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
