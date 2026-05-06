"""
Noctyra 节点公共工具
"""
import numpy as np
import torch
from PIL import Image, ImageDraw


def tensor_to_pil(tensor):
    """单张 IMAGE 张量 -> PIL 图像（自动去 batch 维）"""
    arr = np.clip(255.0 * tensor.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(pil_image):
    """PIL 图像 -> ComfyUI IMAGE 张量（[1,H,W,C]）"""
    return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)


def mask_to_pil_l(mask_tensor, target_size):
    """MASK 张量 -> 与 target_size 匹配的 L 模式 PIL 图像"""
    m = mask_tensor
    if m.dim() == 3:
        m = m[0]
    arr = np.clip(m.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    mask_pil = Image.fromarray(arr, mode="L")
    if mask_pil.size != target_size:
        mask_pil = mask_pil.resize(target_size, Image.Resampling.LANCZOS)
    return mask_pil


def create_default_watermark():
    """占位水印（仅在用户未提供水印图像时使用）"""
    watermark_pil = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark_pil)
    draw.rectangle([0, 0, 100, 100], fill=(0, 0, 0, 100))
    draw.text((20, 40), "Watermark", fill=(255, 255, 255, 200))
    return watermark_pil


def prepare_watermark_rgba(watermark_image, watermark_mask, invert_mask):
    """
    准备 RGBA 水印图像。

    - 未提供 watermark_image：返回占位水印
    - 提供 watermark_image：转 RGBA；若同时提供 mask，则替换 alpha 通道
    - 提供但加载/处理失败：抛异常（避免静默用占位水印覆盖用户成品图）
    """
    if watermark_image is None:
        return create_default_watermark()

    watermark_pil = tensor_to_pil(watermark_image[0]).convert("RGBA")

    if watermark_mask is not None:
        mask_t = watermark_mask[0] if watermark_mask.dim() == 3 else watermark_mask
        mask_l = mask_to_pil_l(mask_t, watermark_pil.size)
        if invert_mask:
            mask_l = mask_l.point(lambda p: 255 - p)
        r, g, b, _ = watermark_pil.split()
        watermark_pil = Image.merge("RGBA", (r, g, b, mask_l))

    return watermark_pil


def apply_opacity(watermark_rgba, opacity):
    """把不透明度系数乘到 RGBA 水印的 alpha 通道（opacity >= 1.0 时直接返回原图）"""
    if opacity >= 1.0:
        return watermark_rgba
    r, g, b, a = watermark_rgba.split()
    a = a.point(lambda p: int(p * opacity))
    return Image.merge("RGBA", (r, g, b, a))
