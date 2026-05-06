"""
水印处理相关节点
"""
import logging
import torch
from PIL import Image

from ._utils import (
    tensor_to_pil,
    pil_to_tensor,
    prepare_watermark_rgba,
    apply_opacity,
)

logger = logging.getLogger("noctyra")


class AddImageWatermark:
    """添加图像水印节点（支持遮罩）

    在 ZML_添加图像水印的基础上新增 MASK 输入。
    当提供水印图像的遮罩时，遮罩会作为水印的 alpha 通道，
    使得水印图像的背景区域真正透明，而不是直接整张贴到背景图上。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "水印大小比例": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 2.0, "step": 0.01}),
                "不透明度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "位置": (["左上", "中上", "右上", "左中", "居中", "右中", "左下", "中下", "右下", "全屏"],),
                "水平边距": ("INT", {"default": 30, "min": 0, "max": 4096}),
                "垂直边距": ("INT", {"default": 30, "min": 0, "max": 4096}),
                "全屏水印旋转角度": ("INT", {"default": -30, "min": -360, "max": 360}),
                "全屏水印密度": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1}),
                "全屏水印间距": ("INT", {"default": 10, "min": 0, "max": 200}),
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
        水印大小比例=0.25,
        不透明度=1.0,
        位置="右下",
        水平边距=30,
        垂直边距=30,
        全屏水印旋转角度=-30,
        全屏水印密度=1.0,
        全屏水印间距=10,
        反转遮罩=False,
        水印图像=None,
        水印遮罩=None,
    ):
        watermark_pil = prepare_watermark_rgba(水印图像, 水印遮罩, 反转遮罩)

        processed_images = []
        for img_tensor in 图像:
            pil_image = tensor_to_pil(img_tensor).convert("RGBA")
            img_width, img_height = pil_image.size

            # 等比缩放水印
            base_size = min(img_width, img_height)
            target_size = int(base_size * 水印大小比例)
            wm_w, wm_h = watermark_pil.size
            ratio = min(target_size / wm_w, target_size / wm_h)
            new_w = max(1, int(wm_w * ratio))
            new_h = max(1, int(wm_h * ratio))
            resized = watermark_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized = apply_opacity(resized, 不透明度)
            wm_w, wm_h = resized.size

            if 位置 == "全屏":
                try:
                    rot_img = resized.rotate(
                        全屏水印旋转角度, expand=True, resample=Image.Resampling.BICUBIC
                    )
                    r_w, r_h = rot_img.size
                    sx = max(1, int((r_w + 水平边距) / 全屏水印密度))
                    sy = max(1, int((r_h + 垂直边距) / 全屏水印密度))
                    offset = sx // 2
                    row_idx = 0
                    for y in range(-r_h, img_height, sy):
                        start_x = -r_w + (offset if (row_idx % 2) else 0)
                        for x in range(start_x, img_width, sx):
                            pil_image.paste(rot_img, (x, y), rot_img)
                        row_idx += 1
                except Exception as e:
                    logger.error(f"全屏水印处理失败: {e}")
            else:
                x_pos = (
                    水平边距 if "左" in 位置
                    else (img_width - wm_w - 水平边距 if "右" in 位置
                          else (img_width - wm_w) // 2)
                )
                y_pos = (
                    垂直边距 if "上" in 位置
                    else (img_height - wm_h - 垂直边距 if "下" in 位置
                          else (img_height - wm_h) // 2)
                )
                x_pos = max(0, min(x_pos, img_width - 1))
                y_pos = max(0, min(y_pos, img_height - 1))
                pil_image.paste(resized, (x_pos, y_pos), resized)

            pil_image = pil_image.convert("RGB")
            processed_images.append(pil_to_tensor(pil_image))

        return (torch.cat(processed_images, dim=0),)


NODE_CLASS_MAPPINGS = {
    "AddImageWatermark": AddImageWatermark,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddImageWatermark": "添加图像水印（支持遮罩）",
}
