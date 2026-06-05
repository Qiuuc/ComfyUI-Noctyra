# ComfyUI-Noctyra — 动作(视频转 3D)节点 · sidecar 运行器
# GPL-3.0 (见仓库 LICENSE)
"""
把重活交给私有 sidecar 环境(py3.10/torch2.3.1)跑，节点本身只在 ComfyUI 主环境
里当壳子。这里提供：

- plugin_root() / work_dir() : 插件根目录与中间产物目录
- sidecar_python()           : 解析 sidecar 的 python.exe(配置优先，其次插件内置 runtime/)
- run()                      : 同步跑子进程，解析 stdout 的百分比驱动 ComfyUI 进度条，
                               响应中断(取消工作流即杀子进程)，非零退出抛错。

设计对齐项目原 main.py 的 run_subprocess：PYTHONUNBUFFERED + 按 \\r/\\n 分块读，
强制清空代理变量(流量有限，且推理不走外网)。
"""
import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import folder_paths

logger = logging.getLogger("noctyra")

_PCT = re.compile(rb"(\d{1,3})\s*%")
_ANSI = re.compile(rb"\x1b\[[0-9;]*[a-zA-Z]")


def plugin_root() -> Path:
    # nodes/motion/_runtime.py -> 上溯三级到插件根
    return Path(__file__).resolve().parents[2]


def work_dir() -> Path:
    """中间产物(pt/npz/log)放 ComfyUI temp 下，跟随 temp 清理策略。"""
    d = Path(folder_paths.get_temp_directory()) / "mocap"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_file(src, dst):
    """把视频暂存到下游工具的输入目录：先试硬链接(同卷零拷贝)，失败再真拷。"""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def video_from_file(path):
    """把磁盘视频包成 ComfyUI 原生 VIDEO(惰性、保留音轨)。"""
    try:
        from comfy_api.latest import InputImpl
        return InputImpl.VideoFromFile(str(path))
    except Exception:
        from comfy_api.input_impl import VideoFromFile
        return VideoFromFile(str(path))


def video_to_mp4(video):
    """把 ComfyUI 原生 VIDEO 落成磁盘 mp4,返回 (路径, stem)。
    GVHMR/HaMeR 都需要真实存在的 mp4 文件,接 ComfyUI 自带『加载视频』节点即可。"""
    from comfy_api.util import VideoContainer
    stem = f"vid_{uuid.uuid4().hex[:8]}"
    out = work_dir() / f"{stem}.mp4"
    video.save_to(str(out), format=VideoContainer("mp4"), codec="auto", metadata=None)
    return out, stem


def sidecar_python() -> str:
    """解析 sidecar 解释器：环境变量 MOCAP_SIDECAR_PYTHON 优先，否则插件内置 runtime/。"""
    p = os.environ.get("MOCAP_SIDECAR_PYTHON", "").strip().strip('"')
    if p:
        return p
    # 插件内置私有环境(install.py 用 uv 建在这里)
    for cand in (
        plugin_root() / "runtime" / "Scripts" / "python.exe",   # Windows venv
        plugin_root() / "runtime" / "python.exe",
        plugin_root() / "runtime" / "bin" / "python",           # *nix venv
    ):
        if cand.exists():
            return str(cand)
    raise RuntimeError(
        "未找到 sidecar python：设置环境变量 MOCAP_SIDECAR_PYTHON 指向解释器，"
        "或运行 install.py 自动建 runtime/ 私有环境。"
    )


def run(cmd, cwd, log_name: str, label: str, extra_env: dict | None = None):
    """同步执行子进程并把进度映射到 ComfyUI 进度条。失败抛 RuntimeError。

    cmd      : 命令列表(元素会 str 化)
    cwd      : 工作目录(同时塞进 PYTHONPATH 头部，兼容隐式命名空间包)
    log_name : 日志文件名(落在 work_dir())
    label    : 进度/日志前缀(中文阶段名)
    """
    import comfy.utils
    import comfy.model_management as mm

    pb = comfy.utils.ProgressBar(100)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # 不走代理：推理不需要外网，且用户流量有限
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""
    env["NO_PROXY"] = "*"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (str(cwd) + os.pathsep + existing_pp) if existing_pp else str(cwd)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    log_path = work_dir() / log_name
    cmd = [str(c) for c in cmd]
    logger.info(f"{label} ▶ {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
    )

    buf = b""
    last_pct = -1
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
            while True:
                # 取消工作流时立刻抛出 → finally 杀子进程
                mm.throw_exception_if_processing_interrupted()
                chunk = proc.stdout.read1(4096)
                if not chunk:
                    if proc.poll() is not None:
                        break
                    continue
                buf += chunk
                while True:
                    r = buf.find(b"\r")
                    n = buf.find(b"\n")
                    if r < 0 and n < 0:
                        break
                    pos = r if n < 0 else (n if r < 0 else min(r, n))
                    raw = buf[:pos]
                    buf = buf[pos + 1:]

                    m = _PCT.search(raw)
                    if m:
                        p = max(0, min(100, int(m.group(1))))
                        if p != last_pct:
                            last_pct = p
                            pb.update_absolute(p, 100)

                    clean = _ANSI.sub(b"", raw).decode("utf-8", "replace")
                    if clean.strip():
                        lf.write(clean + "\n")
                        lf.flush()

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"{label} 失败 (rc={rc})，详见 {log_path}")
    finally:
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait()
            except Exception:
                pass
    return log_path
