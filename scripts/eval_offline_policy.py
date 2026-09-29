# 离线推理：用留出的示教数据评估训练好的 UMI 策略
# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import argparse
import collections
import copy
import pathlib
import pickle
import time

import dill
import hydra
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.pose_util import pose10d_to_mat


# 终端语义化颜色，禁止使用 * - = 作分隔线
class C:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    GREY = '\033[90m'
    END = '\033[0m'


def log(tag, msg, color=C.CYAN):
    print(f"{color}[{tag}]{C.END} {msg}", flush=True)


def load_policy(ckpt_path, device):
    # 用 checkpoint 里保存的 cfg 重建 workspace，保证 shape_meta 与训练完全一致
    payload = torch.load(
        open(ckpt_path, 'rb'),
        map_location='cpu',
        pickle_module=dill
    )
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # 训练时 use_ema=True，评估统一用 EMA 权重
    if cfg.training.use_ema:
        policy = workspace.ema_model
        unused = workspace.model
    else:
        policy = workspace.model
        unused = workspace.ema_model
    # 释放未使用的模型，8GB 显存下避免无谓占用
    workspace.model = None
    workspace.ema_model = None
    del unused, workspace, payload
    return cfg, policy


def build_heldout_dataset(cfg):
    # 直接实例化训练用的 dataset，读到同一个 LMDB 缓存与 val_mask
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    n_episodes = dataset.replay_buffer.n_episodes
    val_ratio = cfg.task.dataset.get('val_ratio', 0.0)
    max_train_episodes = cfg.task.dataset.get('max_train_episodes', None)
    seed = cfg.task.dataset.get('seed', 42)

    # 复现训练时的 episode 选择：val_mask 留出，再对训练集下采样到 max_train_episodes
    val_mask = get_val_mask(
        n_episodes=n_episodes,
        val_ratio=val_ratio,
        seed=seed
    )
    train_mask = downsample_mask(
        mask=~val_mask,
        max_n=max_train_episodes,
        seed=seed
    )
    heldout_mask = ~train_mask

    # 仿照 get_validation_dataset，用留出 episode 重建一个 sampler
    heldout_sampler = SequenceSampler(
        shape_meta=dataset.shape_meta,
        replay_buffer=dataset.replay_buffer,
        rgb_keys=dataset.rgb_keys,
        lowdim_keys=dataset.sampler_lowdim_keys,
        key_horizon=dataset.key_horizon,
        key_latency_steps=dataset.key_latency_steps,
        key_down_sample_steps=dataset.key_down_sample_steps,
        episode_mask=heldout_mask,
        action_padding=dataset.action_padding,
        repeat_frame_prob=dataset.repeat_frame_prob,
        max_duration=dataset.max_duration
    )
    heldout_set = copy.copy(dataset)
    heldout_set.sampler = heldout_sampler
    heldout_set.val_mask = heldout_mask
    return dataset, heldout_set, heldout_mask


def pick_windows(heldout_set, dataset, num_episodes, windows_per_episode, seed):
    # 把留出样本按 episode 分组，再按 episode 抽样，避免同一段视频被重复采样
    episode_ends = dataset.replay_buffer.episode_ends[:]
    cur_idx = np.array([item[0] for item in heldout_set.sampler.indices])
    ep_ids = np.searchsorted(episode_ends, cur_idx, side='right')

    ep_to_pos = collections.defaultdict(list)
    for pos, ep in enumerate(ep_ids):
        ep_to_pos[int(ep)].append(pos)
    episodes = sorted(ep_to_pos.keys())

    rng = np.random.default_rng(seed)
    n_ep = min(num_episodes, len(episodes))
    chosen_eps = rng.choice(episodes, size=n_ep, replace=False)
    chosen_eps = sorted(int(e) for e in chosen_eps)

    windows = list()
    for ep in chosen_eps:
        positions = ep_to_pos[ep]
        n_win = min(windows_per_episode, len(positions))
        picks = rng.choice(positions, size=n_win, replace=False)
        for pos in picks:
            windows.append((int(ep), int(pos)))
    return windows


def infer_one(policy, sample, device):
    # dataset.__getitem__ 已按训练同款方式完成图像归一化、相对位姿与动作转换
    obs = dict_apply(
        sample['obs'],
        lambda x: x.unsqueeze(0).to(device)
    )
    with torch.no_grad():
        result = policy.predict_action(obs)
    return result['action_pred'][0].float().cpu().numpy()


def rot_error_deg(pred, gt):
    # 动作是 10 维（pos3 + rot6d6 + gripper1），取前 9 维还原旋转矩阵再算测地线夹角
    r_pred = pose10d_to_mat(pred[..., :9])[..., :3, :3]
    r_gt = pose10d_to_mat(gt[..., :9])[..., :3, :3]
    rel = np.matmul(r_pred, np.transpose(r_gt, (0, 2, 1)))
    cos = (np.trace(rel, axis1=-2, axis2=-1) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def compute_metrics(pred, gt):
    pos_err = np.linalg.norm(pred[:, :3] - gt[:, :3], axis=-1) * 100.0
    rot_err = rot_error_deg(pred, gt)
    grip_err = np.abs(pred[:, 9] - gt[:, 9])
    return pos_err, rot_err, grip_err


def save_plots(records, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(records)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3 * n), squeeze=False)
    for i, rec in enumerate(records):
        pred = rec['pred']
        gt = rec['gt']
        t = np.arange(pred.shape[0])
        for j, axis in enumerate(['x', 'y', 'z']):
            ax = axes[i][j]
            ax.plot(t, gt[:, j] * 100.0, color='tab:blue', label='GT')
            ax.plot(t, pred[:, j] * 100.0, color='tab:red', linestyle='--', label='Pred')
            ax.set_title(f"ep{rec['episode']} pos {axis} (cm)")
            ax.set_xlabel('step')
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend()
        ax = axes[i][3]
        ax.plot(t, gt[:, 9], color='tab:blue', label='GT')
        ax.plot(t, pred[:, 9], color='tab:red', linestyle='--', label='Pred')
        ax.set_title(f"ep{rec['episode']} gripper")
        ax.set_xlabel('step')
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='用留出的示教数据离线评估 UMI 策略'
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='data/outputs/2026.09.27/19.30.29_formal_umi/checkpoints/latest.ckpt',
        help='checkpoint 路径，或含 checkpoints/latest.ckpt 的目录'
    )
    parser.add_argument(
        '--normalizer',
        type=str,
        default=None,
        help='normalizer.pkl 路径，默认取 checkpoint 同级目录'
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_episodes', type=int, default=20)
    parser.add_argument('--windows_per_episode', type=int, default=5)
    parser.add_argument('--latency_warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='data/eval_offline')
    parser.add_argument('--smoke', action='store_true', help='冒烟模式：1 episode 1 窗口且不画图')
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args()

    if args.smoke:
        args.num_episodes = 1
        args.windows_per_episode = 1
        args.no_plots = True
        log('冒烟', '启用冒烟模式：1 个 episode，1 个窗口，不画图', C.YELLOW)

    ckpt_path = pathlib.Path(args.ckpt)
    if not ckpt_path.suffix == '.ckpt':
        ckpt_path = ckpt_path.joinpath('checkpoints', 'latest.ckpt')
    if not ckpt_path.is_file():
        log('错误', f'checkpoint 不存在：{ckpt_path}', C.RED)
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log('环境', f'device={device}，torch={torch.__version__}')

    # %%
    log('加载', f'checkpoint：{ckpt_path}')
    cfg = None
    policy = None
    cfg, policy = load_policy(str(ckpt_path), device)

    normalizer_path = args.normalizer
    if normalizer_path is None:
        normalizer_path = ckpt_path.parent.parent.joinpath('normalizer.pkl')
    normalizer = pickle.load(open(normalizer_path, 'rb'))
    policy.set_normalizer(normalizer)
    policy.num_inference_steps = cfg.policy.num_inference_steps
    policy.eval().to(device)
    log('加载', f'normalizer：{normalizer_path}，num_inference_steps={policy.num_inference_steps}', C.GREEN)

    # %%
    log('数据', '实例化数据集并复现训练 mask')
    dataset, heldout_set, heldout_mask = build_heldout_dataset(cfg)
    n_episodes = dataset.replay_buffer.n_episodes
    n_heldout = int(heldout_mask.sum())
    log(
        '数据',
        f'总 episode={n_episodes}，留出 episode={n_heldout}，'
        f'留出窗口={len(heldout_set.sampler)}',
        C.GREEN
    )
    if n_heldout == 0:
        log('错误', '没有留出 episode，无法评估', C.RED)
        sys.exit(1)

    windows = pick_windows(
        heldout_set,
        dataset,
        args.num_episodes,
        args.windows_per_episode,
        args.seed
    )
    log('采样', f'选中 {len(windows)} 个推理窗口')

    # %%
    log('预热', f'预热 {args.latency_warmup} 次')
    np.random.seed(args.seed)
    first_sample = heldout_set[windows[0][1]]
    for _ in range(args.latency_warmup):
        infer_one(policy, first_sample, device)
    log('预热', '完成', C.GREEN)

    # %%
    log('推理', '开始逐窗口推理')
    records = list()
    latencies = list()
    for ep, pos in windows:
        np.random.seed(args.seed + pos)
        sample = heldout_set[pos]
        t0 = time.perf_counter()
        pred = infer_one(policy, sample, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        latencies.append(dt)

        gt = sample['action'].numpy()
        pos_err, rot_err, grip_err = compute_metrics(pred, gt)
        records.append({
            'episode': ep,
            'pos': pos,
            'pred': pred,
            'gt': gt,
            'pos_err': pos_err,
            'rot_err': rot_err,
            'grip_err': grip_err
        })
        print(
            f"{C.GREY}  episode={ep} window={pos} "
            f"pos_err={pos_err.mean():.2f}cm rot_err={rot_err.mean():.2f}deg "
            f"grip_err={grip_err.mean():.4f} latency={dt * 1000:.1f}ms{C.END}",
            flush=True
        )

    # %%
    lat = np.array(latencies) * 1000.0
    all_pos = np.concatenate([r['pos_err'] for r in records])
    all_rot = np.concatenate([r['rot_err'] for r in records])
    all_grip = np.concatenate([r['grip_err'] for r in records])
    any_nan = any(
        not np.isfinite(r['pred']).all() for r in records
    )

    print()
    log('结果', f'预测有限值校验：{"通过" if not any_nan else "存在 nan/inf"}', C.GREEN if not any_nan else C.RED)
    log(
        '结果',
        f'预测耗时 ms：mean={lat.mean():.1f} median={np.median(lat):.1f} '
        f'max={lat.max():.1f}（含 {policy.num_inference_steps} 步 DDIM）',
        C.BLUE
    )
    log(
        '结果',
        f'位置误差 cm：mean={all_pos.mean():.2f} p95={np.percentile(all_pos, 95):.2f} max={all_pos.max():.2f}',
        C.BLUE
    )
    log(
        '结果',
        f'旋转误差 deg：mean={all_rot.mean():.2f} p95={np.percentile(all_rot, 95):.2f} max={all_rot.max():.2f}',
        C.BLUE
    )
    log(
        '结果',
        f'夹爪误差：mean={all_grip.mean():.4f} p95={np.percentile(all_grip, 95):.4f} max={all_grip.max():.4f}',
        C.BLUE
    )

    # %%
    out_dir = pathlib.Path(args.out_dir).joinpath(time.strftime('%Y.%m.%d/%H.%M.%S'))
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir.joinpath('metrics.npz'),
        latencies_ms=lat,
        pos_err_cm=all_pos,
        rot_err_deg=all_rot,
        grip_err=all_grip,
        episodes=np.array([r['episode'] for r in records]),
        windows=np.array([r['pos'] for r in records])
    )
    if not args.no_plots:
        save_plots(records[:8], out_dir.joinpath('pred_vs_gt.png'))
        log('产物', f'预测对比图：{out_dir.joinpath("pred_vs_gt.png")}')
    log('产物', f'指标数据：{out_dir.joinpath("metrics.npz")}', C.GREEN)


if __name__ == "__main__":
    main()
