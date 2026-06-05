# ComfyUI-Noctyra — 动作(视频转 3D)· SMPL-X → BVH 写出
# GPL-3.0 (见仓库 LICENSE)
"""
把合并后的 SMPL-X 动作(AMASS 风格 npz)写成 BVH。纯 numpy/scipy，跑在 ComfyUI
主环境即可,不进 sidecar。

输入 npz 字段(由合并节点产出)：
    poses (T, 165)  —— 55 关节 × 轴角(轴角=旋转向量)
    trans (T, 3)    —— 根位移
    mocap_frame_rate—— 帧率

骨架取 _skeleton 的 55 关节。轴角→ZXY 欧拉(BVH 通道顺序),根/各关节一致地留在
SMPL-X 原生坐标系(不做 Y/Z 互换),保证旋转与位移同一参考系、动作不串。
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from ._skeleton import SMPLX_NAMES, SMPLX_PARENTS, SMPLX_OFFSETS
except ImportError:  # 允许独立导入(测试用)
    from _skeleton import SMPLX_NAMES, SMPLX_PARENTS, SMPLX_OFFSETS

_OFFSET_SCALE = 10.0    # 示意骨长缩放
_TRANS_SCALE = 100.0    # 米 → 厘米


def _children(parent_idx):
    return [i for i, p in enumerate(SMPLX_PARENTS) if p == parent_idx]


def _aa_to_zxy_deg(aa):
    """轴角 → ZXY 欧拉角(度),返回 [z, x, y] 对应 BVH 的 Zrot Xrot Yrot。"""
    if not np.any(aa):
        return (0.0, 0.0, 0.0)
    z, x, y = R.from_rotvec(aa).as_euler("ZXY", degrees=True)
    return (z, x, y)


def _build_hierarchy():
    """递归生成 BVH HIERARCHY 段。根 6 通道,其余 3 通道。"""
    lines = ["HIERARCHY", "ROOT " + SMPLX_NAMES[0], "{",
             "\tOFFSET 0.000000 0.000000 0.000000",
             "\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"]

    def recurse(idx, depth):
        indent = "\t" * depth
        kids = _children(idx)
        if kids:
            for c in kids:
                o = [v * _OFFSET_SCALE for v in SMPLX_OFFSETS[c]]
                lines.append(f"{indent}JOINT {SMPLX_NAMES[c]}")
                lines.append(f"{indent}{{")
                lines.append(f"{indent}\tOFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
                lines.append(f"{indent}\tCHANNELS 3 Zrotation Xrotation Yrotation")
                recurse(c, depth + 1)
                lines.append(f"{indent}}}")
        else:
            # 叶子用一个短 End Site 收尾
            o = [v * _OFFSET_SCALE * 0.5 for v in SMPLX_OFFSETS[idx]]
            lines.append(f"{indent}End Site")
            lines.append(f"{indent}{{")
            lines.append(f"{indent}\tOFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
            lines.append(f"{indent}}}")

    recurse(0, 1)
    lines.append("}")
    return lines


def write_bvh(npz_path, out_path, fps=None):
    data = np.load(npz_path)
    poses = np.asarray(data["poses"], dtype=np.float64)      # (T, 165)
    T = poses.shape[0]
    poses = poses.reshape(T, -1, 3)                          # (T, J, 3)
    J = min(poses.shape[1], 55)
    trans = np.asarray(data["trans"], dtype=np.float64) if "trans" in data else np.zeros((T, 3))
    if fps is None:
        fps = int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else 30

    # HIERARCHY + 关节写出顺序(深度优先,与 hierarchy 递归一致)
    order = []

    def collect(idx):
        order.append(idx)
        for c in _children(idx):
            collect(c)
    collect(0)

    lines = _build_hierarchy()
    lines += ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.6f}"]

    for f in range(T):
        vals = []
        # 根:位移(原生坐标系,m→cm) + 旋转
        t = trans[f] * _TRANS_SCALE
        vals += [t[0], t[1], t[2]]
        for j in order:
            aa = poses[f, j] if j < J else np.zeros(3)
            vals += list(_aa_to_zxy_deg(aa))
        lines.append(" ".join(f"{v:.6f}" for v in vals))

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out_path, T, fps
