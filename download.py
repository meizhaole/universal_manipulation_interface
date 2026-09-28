# 多线程分段下载 UMI 官方 cup_in_the_wild 数据集，并聚合显示总进度

import os
import sys
import time
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests

URL = "https://real.stanford.edu/umi/data/zarr_datasets/cup_in_the_wild.zarr.zip"
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT_DIR / "data" / "cup_in_the_wild.zarr.zip"
USER_AGENT = "umi-dataset-downloader/1.0"
# 每个 HTTP Range 请求的大小，也是写盘单位
READ_CHUNK = 8 * 1024 * 1024
MAX_RETRY = 5
TIMEOUT = 60
PROGRESS_INTERVAL = 0.3
BAR_WIDTH = 32


def human_size(num_bytes):
    # 把字节数转成易读单位
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}PiB"


def format_eta(seconds):
    # 把剩余秒数转成时分秒
    seconds = max(int(seconds), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}时{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def get_remote_size(session, url):
    # HEAD 拿总大小，并确认服务器支持 Range 分段
    resp = session.head(
        url,
        timeout=TIMEOUT,
        allow_redirects=True
    )
    resp.raise_for_status()
    size = int(resp.headers.get("Content-Length", 0))
    accept_ranges = resp.headers.get("Accept-Ranges", "")
    if size <= 0:
        raise RuntimeError("服务器未返回 Content-Length，无法确定文件大小")
    if "bytes" not in accept_ranges.lower():
        raise RuntimeError(f"服务器不支持 Range 分段下载，Accept-Ranges={accept_ranges!r}")
    return size


def fetch_range(url, start, end):
    # 下载闭区间 [start, end]，失败按退避重试
    headers = {"Range": f"bytes={start}-{end}"}
    last_error = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            session = requests.Session()
            session.headers.update({"User-Agent": USER_AGENT})
            try:
                resp = session.get(
                    url,
                    headers=headers,
                    timeout=TIMEOUT
                )
                resp.raise_for_status()
                # 起始偏移非 0 时必须返回 206，否则说明服务器忽略了 Range
                if start > 0 and resp.status_code != 206:
                    raise RuntimeError(f"服务器忽略了 Range，返回 {resp.status_code}")
                return resp.content
            finally:
                session.close()
        except Exception as err:
            last_error = err
            if attempt < MAX_RETRY:
                time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"区间 {start}-{end} 下载失败：{last_error}")


def download_range(url, start, end, filepath, state):
    # 循环下载 [start, end] 并就地写入文件对应偏移，区间互不重叠
    pos = start
    with open(filepath, "r+b") as f:
        while pos <= end:
            chunk_end = min(pos + READ_CHUNK - 1, end)
            data = fetch_range(
                url,
                pos,
                chunk_end
            )
            if not data:
                raise RuntimeError(f"区间 {pos}-{chunk_end} 返回空数据")
            f.seek(pos)
            f.write(data)
            pos += len(data)
            with state["lock"]:
                state["done"] += len(data)
    return None


def render(state, total, start_time, final=False):
    # 汇总所有线程已下载字节数，画一条总进度条
    done = state["done"]
    elapsed = max(time.time() - start_time, 1e-6)
    speed = done / elapsed
    ratio = done / total if total else 0.0
    eta = (total - done) / speed if speed > 0 else 0
    filled = min(int(BAR_WIDTH * ratio), BAR_WIDTH)
    bar = "#" * filled + "-" * (BAR_WIDTH - filled)
    line = (
        f"\r下载进度 [{bar}] {ratio * 100:5.1f}%  "
        f"{human_size(done)}/{human_size(total)}  "
        f"{human_size(speed)}/s  剩余 {format_eta(eta)}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()
    if final:
        sys.stdout.write("\n")


def split_ranges(total, n_threads):
    # 把总字节数尽量均分给每个线程
    ranges = []
    base, remainder = divmod(total, n_threads)
    cursor = 0
    for index in range(n_threads):
        length = base + (1 if index < remainder else 0)
        if length <= 0:
            continue
        ranges.append((cursor, cursor + length - 1))
        cursor += length
    return ranges


def main():
    parser = argparse.ArgumentParser(description="多线程下载 UMI 官方数据集")
    parser.add_argument("--url", default=URL, help="数据集下载地址")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT), help="输出文件路径")
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=0,
        help="线程数，默认取本机逻辑核数"
    )
    args = parser.parse_args()

    # 默认吃满本机逻辑核数
    if args.threads > 0:
        n_threads = args.threads
    elif hasattr(os, "sched_getaffinity"):
        n_threads = len(os.sched_getaffinity(0))
    else:
        n_threads = os.cpu_count() or 4

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    try:
        total = get_remote_size(session, args.url)
    finally:
        session.close()

    # 预分配文件，各线程随后就地写入自己的区间
    with open(out_path, "wb") as f:
        f.truncate(total)

    ranges = split_ranges(total, n_threads)
    state = {"done": 0, "lock": threading.Lock()}
    start_time = time.time()

    print(f"来源: {args.url}")
    print(f"大小: {human_size(total)}  线程: {len(ranges)}  输出: {out_path}")

    try:
        with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
            futures = [
                executor.submit(
                    download_range,
                    args.url,
                    start,
                    end,
                    out_path,
                    state
                )
                for start, end in ranges
            ]
            while not all(f.done() for f in futures):
                render(state, total, start_time)
                time.sleep(PROGRESS_INTERVAL)
            # 有任何线程抛错都会在这里重新抛出
            for future in futures:
                future.result()
        render(state, total, start_time, final=True)
    except KeyboardInterrupt:
        out_path.unlink(missing_ok=True)
        print(f"\n已中断，删除未完成文件 {out_path}")
        raise SystemExit(130)
    except Exception:
        out_path.unlink(missing_ok=True)
        print(f"\n下载失败，删除未完成文件 {out_path}")
        raise

    actual = out_path.stat().st_size
    if actual != total:
        raise SystemExit(f"文件大小不符：期望 {total}，实际 {actual}")
    with open(out_path, "rb") as f:
        magic = f.read(4)
    if magic != b"PK\x03\x04":
        raise SystemExit(f"文件头不是 zip：{magic!r}")

    print(f"下载完成: {out_path} ({human_size(actual)})")
    print(f"在 formal.yaml 中保持 task.dataset_path 为 data/cup_in_the_wild.zarr.zip 即可训练")


if __name__ == "__main__":
    main()
