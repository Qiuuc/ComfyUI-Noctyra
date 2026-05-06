"""
全屏网格水印节点 - 独立控制行列密度
"""
import logging
import random
import torch
from PIL import Image

from ._utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_watermark_rgba,
    apply_opacity,
)

logger = logging.getLogger("noctyra")


class AddGridWatermark:
    """添加全屏网格水印（独立控制行列密度）

    分别控制水平方向和垂直方向的密度，可以独立调整行列数量。
    包围盒基于水印实际尺寸计算，水平和垂直互不干扰。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "水印大小比例": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 2.0, "step": 0.01}),
                "不透明度": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "水平密度": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 5.0, "step": 0.1, "description": "每行水印数量系数"}),
                "垂直密度": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 5.0, "step": 0.1, "description": "每列水印数量系数"}),
                "包围盒倍数": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1, "description": "1.0=紧贴，1.5=半间距，2.0=宽松"}),
                "最小包围盒比例": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01, "description": "相对于图片短边的最小包围盒大小"}),
                "最大随机偏移": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05, "description": "相对于剩余空间的偏移比例，1.0=可触及包围盒边缘"}),
                "旋转角度": ("INT", {"default": -30, "min": -360, "max": 360}),
                "随机种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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
    CATEGORY = "Noctyra/图片"

    def add_watermark(
        self,
        图像,
        水印大小比例=0.12,
        不透明度=0.5,
        水平密度=1.0,
        垂直密度=1.0,
        包围盒倍数=1.5,
        最小包围盒比例=0.0,
        最大随机偏移=0.8,
        旋转角度=-30,
        随机种子=0,
        反转遮罩=False,
        水印图像=None,
        水印遮罩=None,
    ):
        if 图像 is None or len(图像) == 0:
            return (torch.zeros((0, 1, 1, 3)),)

        watermark_pil = prepare_watermark_rgba(水印图像, 水印遮罩, 反转遮罩)

        # 按分辨率分组：同分辨率共享同一布局（位置序列）
        resolution_groups = {}
        for img_idx, img_tensor in enumerate(图像):
            h, w = img_tensor.shape[0], img_tensor.shape[1]
            resolution_groups.setdefault((w, h), []).append((img_idx, img_tensor))

        processed_images = [None] * len(图像)

        for (img_width, img_height), group in resolution_groups.items():
            # 种子：用户指定 > 0 时直接用；否则按尺寸派生（同尺寸 -> 同布局）
            if 随机种子 > 0:
                seed = 随机种子
            else:
                seed = ((img_width * 73856093) ^ (img_height * 19349663)) & 0xffffffffffffffff
            rng = random.Random(seed)

            # 缩放 -> 不透明度 -> 旋转
            base_size = min(img_width, img_height)
            target_size = int(base_size * 水印大小比例)
            wm_w, wm_h = watermark_pil.size
            ratio = min(target_size / wm_w, target_size / wm_h)
            resized_watermark = watermark_pil.resize(
                (max(1, int(wm_w * ratio)), max(1, int(wm_h * ratio))),
                Image.Resampling.LANCZOS,
            )
            resized_watermark = apply_opacity(resized_watermark, 不透明度)

            rotated_watermark = (
                resized_watermark.rotate(旋转角度, expand=True, resample=Image.Resampling.BICUBIC)
                if 旋转角度 != 0 else resized_watermark
            )
            rot_w, rot_h = rotated_watermark.size

            # 包围盒（长方形，水平/垂直独立）
            min_box = int(base_size * 最小包围盒比例)
            box_w = max(int(rot_w * 包围盒倍数), min_box)
            box_h = max(int(rot_h * 包围盒倍数), min_box)

            # 行列数（按密度 + 图片宽高比）
            base_cols = max(3, int(3 * 水平密度))
            base_rows = max(3, int(3 * 垂直密度))
            cols = max(2, int(base_cols * img_width / max(img_width, img_height)))
            rows = max(2, int(base_rows * img_height / max(img_width, img_height)))

            start_x = (img_width - cols * box_w) // 2
            start_y = (img_height - rows * box_h) // 2

            # 预计算所有水印位置
            positions = []
            for row in range(rows):
                for col in range(cols):
                    cx = start_x + col * box_w + box_w / 2
                    cy = start_y + row * box_h + box_h / 2
                    spare_x = (box_w - rot_w) / 2
                    spare_y = (box_h - rot_h) / 2
                    max_off_x = spare_x * min(最大随机偏移, 1.0)
                    max_off_y = spare_y * min(最大随机偏移, 1.0)
                    off_x = rng.uniform(-max_off_x, max_off_x) if max_off_x > 0 else 0
                    off_y = rng.uniform(-max_off_y, max_off_y) if max_off_y > 0 else 0
                    positions.append((
                        int(cx - rot_w / 2 + off_x),
                        int(cy - rot_h / 2 + off_y),
                    ))

            # 应用到该分辨率的所有图片
            for img_idx, img_tensor in group:
                pil_image = tensor_to_pil(img_tensor).convert("RGBA")
                for fx, fy in positions:
                    if fx + rot_w > 0 and fx < img_width and fy + rot_h > 0 and fy < img_height:
                        pil_image.paste(rotated_watermark, (fx, fy), rotated_watermark)
                processed_images[img_idx] = pil_to_tensor(pil_image.convert("RGB"))

        # 防御：group 处理若中途异常漏填，用原图 fallback 避免 torch.cat 崩溃
        for idx, item in enumerate(processed_images):
            if item is None:
                fallback = 图像[idx].unsqueeze(0) if 图像[idx].dim() == 3 else 图像[idx]
                processed_images[idx] = fallback

        return (torch.cat(processed_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "AddGridWatermark": AddGridWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddGridWatermark": "添加网格水印（独立行列密度）",
}
