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

"""
全屏(平铺)水印节点 —— 图片 / 视频

从图片水印、视频水印节点中抽出的『全屏』模式：把水印旋转后斜向平铺铺满整图。
图片版处理单/多帧；视频版另带『帧区间 + 淡入淡出』。
（另有 watermark_grid 的网格水印，是带随机偏移的网格平铺，风格不同，二者并存。）
"""
import logging

import torch

from .._utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_watermark_rgba,
    apply_watermark_frame,
)

logger = logging.getLogger("noctyra")


def _tile_frame(img_tensor, watermark_pil, size_ratio, opacity, angle, sx, sy, density):
    """对单帧做全屏平铺水印，返回 [1,H,W,3] 张量。"""
    pil = tensor_to_pil(img_tensor).convert("RGBA")
    result = apply_watermark_frame(
        pil, watermark_pil,
        size_ratio=size_ratio, opacity=opacity, position="全屏",
        margin_x=sx, margin_y=sy, fs_angle=angle, fs_density=density,
    )
    return pil_to_tensor(result)


class AddFullscreenWatermark:
    """全屏水印（图片）

    把水印旋转后斜向平铺铺满整图，适合防盗用的满版水印。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "水印大小比例": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 2.0, "step": 0.01}),
                "不透明度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "旋转角度": ("INT", {"default": -30, "min": -360, "max": 360}),
                "水平间距": ("INT", {"default": 10, "min": 0, "max": 4096}),
                "垂直间距": ("INT", {"default": 10, "min": 0, "max": 4096}),
                "密度": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1}),
                "反转遮罩": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "水印图像": ("IMAGE",),
                "水印遮罩": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "add_watermark"
    CATEGORY = "Noctyra/添加水印"

    def add_watermark(self, 图像, 水印大小比例=0.15, 不透明度=0.5, 旋转角度=-30,
                      水平间距=10, 垂直间距=10, 密度=1.0, 反转遮罩=False,
                      水印图像=None, 水印遮罩=None):
        if 图像 is None or len(图像) == 0:
            return (torch.zeros((0, 1, 1, 3)),)
        watermark_pil = prepare_watermark_rgba(水印图像, 水印遮罩, 反转遮罩)
        out = [
            _tile_frame(t, watermark_pil, 水印大小比例, 不透明度, 旋转角度, 水平间距, 垂直间距, 密度)
            for t in 图像
        ]
        return (torch.cat(out, dim=0),)


class AddVideoFullscreenWatermark:
    """全屏水印（视频）

    给视频帧序列铺满版水印，支持只在 [起始帧, 结束帧] 区间显示并做淡入淡出。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "水印大小比例": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 2.0, "step": 0.01}),
                "不透明度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "旋转角度": ("INT", {"default": -30, "min": -360, "max": 360}),
                "水平间距": ("INT", {"default": 10, "min": 0, "max": 4096}),
                "垂直间距": ("INT", {"default": 10, "min": 0, "max": 4096}),
                "密度": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1}),
                "起始帧": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "结束帧": ("INT", {"default": -1, "min": -1, "max": 1000000}),
                "淡入帧数": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "淡出帧数": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "反转遮罩": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "水印图像": ("IMAGE",),
                "水印遮罩": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "add_watermark"
    CATEGORY = "Noctyra/添加水印"

    def add_watermark(self, 图像, 水印大小比例=0.15, 不透明度=0.5, 旋转角度=-30,
                      水平间距=10, 垂直间距=10, 密度=1.0,
                      起始帧=0, 结束帧=-1, 淡入帧数=0, 淡出帧数=0, 反转遮罩=False,
                      水印图像=None, 水印遮罩=None):
        if 图像 is None or len(图像) == 0:
            return (torch.zeros((0, 1, 1, 3)),)

        n = len(图像)
        start = max(0, 起始帧)
        end = (n - 1) if 结束帧 < 0 else min(结束帧, n - 1)
        watermark_pil = prepare_watermark_rgba(水印图像, 水印遮罩, 反转遮罩)

        out = []
        for i, t in enumerate(图像):
            if i < start or i > end:
                out.append(t.unsqueeze(0) if t.dim() == 3 else t)
                continue
            factor = 1.0
            if 淡入帧数 > 0:
                factor = min(factor, (i - start + 1) / 淡入帧数)
            if 淡出帧数 > 0:
                factor = min(factor, (end - i + 1) / 淡出帧数)
            opacity = 不透明度 * max(0.0, min(1.0, factor))
            if opacity <= 0.0:
                out.append(t.unsqueeze(0) if t.dim() == 3 else t)
                continue
            out.append(_tile_frame(t, watermark_pil, 水印大小比例, opacity, 旋转角度, 水平间距, 垂直间距, 密度))
        return (torch.cat(out, dim=0),)


NODE_CLASS_MAPPINGS = {
    "AddFullscreenWatermark": AddFullscreenWatermark,
    "AddVideoFullscreenWatermark": AddVideoFullscreenWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddFullscreenWatermark": "全屏水印（图片·平铺）",
    "AddVideoFullscreenWatermark": "全屏水印（视频·帧区间+淡入淡出）",
}
