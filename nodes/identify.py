# ComfyUI-Noctyra
# Copyright (C) 2026 Qiuuc
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ---------------------------------------------------------------------------
# 本节点封装 vendored MIT 项目 remove-ai-watermarks 的 identify 模块
# (Copyright (c) 2025 wiltodelta)。详见仓库 THIRD_PARTY_NOTICES.md。
# ---------------------------------------------------------------------------

"""
AI 溯源鉴定节点

聚合 C2PA 内容凭证、IPTC "Made with AI" 标签、嵌入生成参数、SynthID 元数据
代理、可见 Gemini 闪光等信号，给出一张图的来源判定与水印清单。

两种输入(二选一)：
- 接 IMAGE：只能做**像素域检测**(可见水印)。ComfyUI 的 IMAGE 是解码后的像素，
  不含元数据，C2PA/EXIF/IPTC 这些已随解码丢失。
- 填**图片路径**：读原始文件做**完整溯源**(元数据 + 像素域)。只填文件名时
  自动到 ComfyUI input/ 目录找。
"""
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ._vendor import ensure as _ensure_vendor

logger = logging.getLogger("noctyra")


def _resolve_path(p):
    """解析图片路径：绝对/相对存在则用；只给文件名则到 ComfyUI input/ 找。"""
    p = p.strip().strip('"')
    if not p:
        return None
    if os.path.isfile(p):
        return p
    try:
        import folder_paths
        cand = os.path.join(folder_paths.get_input_directory(), p)
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    return None


class IdentifyAIProvenance:
    """AI 溯源鉴定（C2PA / 类型识别）

    优先用『图片路径』做完整溯源(元数据+像素域)；若只接了 IMAGE，则仅做像素域
    可见水印检测(元数据已随解码丢失)。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "图片路径": ("STRING", {"default": ""}),
                "图像": ("IMAGE",),
                "检测像素域": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("报告", "是否AI生成", "平台")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Noctyra/水印去除"

    def run(self, 图片路径="", 图像=None, 检测像素域=True):
        _ensure_vendor()
        try:
            from remove_ai_watermarks.identify import identify
        except ImportError as e:
            return (f"vendored 模块导入失败: {e}", "未知", "")

        path = _resolve_path(图片路径)
        note = ""
        tmp = None
        if path is None and 图像 is not None:
            # 只有 IMAGE：存临时文件，仅能做像素域检测(元数据已丢失)
            arr = np.clip(图像[0].cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            tmp = os.path.join(tempfile.gettempdir(), f"noctyra_idf_{int(time.time()*1000)}.png")
            Image.fromarray(arr).save(tmp)
            path = tmp
            note = "（输入为 IMAGE：仅像素域检测，元数据/C2PA 已随解码丢失，如需完整溯源请填图片路径）"
        if path is None:
            return ("未提供有效输入：请填『图片路径』或连接『图像』。", "未知", "")

        try:
            report = identify(Path(path), check_visible=检测像素域, check_invisible=检测像素域)
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

        verdict_map = {True: "是(AI生成)", False: "否(非AI)", None: "未知"}
        verdict = verdict_map[report.is_ai_generated]
        platform = report.platform or ""

        lines = [
            f"判定: {verdict}  (置信度: {report.confidence})",
            f"平台: {platform or '未确定'}",
        ]
        if note:
            lines.insert(0, note)
        if report.is_ai_generated is None:
            lines.append(
                "  未发现本地可读的 AI 信号。这≠『干净』——元数据常被重编码/截图/上传"
                "剥除，SynthID 类像素水印也无本地检测器。"
            )
        if report.integrity_clashes:
            lines.append("⚠ 完整性冲突(溯源信号互相矛盾):")
            lines += [f"  - {c}" for c in report.integrity_clashes]
        if report.watermarks:
            lines.append("水印 / 溯源标记:")
            lines += [f"  - {w}" for w in report.watermarks]
        else:
            lines.append("未发现水印或溯源标记。")
        if report.caveats:
            lines.append("说明:")
            lines += [f"  - {c}" for c in report.caveats]

        report_str = "\n".join(lines)
        try:
            print(f"[Noctyra 溯源鉴定] {path}\n{report_str}")
        except Exception:
            import sys
            enc = sys.stdout.encoding or "utf-8"
            print((f"[Noctyra 溯源鉴定]\n{report_str}").encode(enc, "replace").decode(enc))
        return (report_str, verdict, platform)


NODE_CLASS_MAPPINGS = {
    "IdentifyAIProvenance": IdentifyAIProvenance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IdentifyAIProvenance": "【鉴定】AI水印溯源（C2PA/类型识别）",
}
