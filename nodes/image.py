"""
图片处理相关节点
"""
import os
import numpy as np
import torch
from PIL import Image
import folder_paths


class SaveImageNoMetadata:
    """保存图片（无元数据）节点
    
    保存图片到输出目录，不写入任何元数据信息
    """
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Clean_Image"}),
            },
            "optional": {
                "compress_level": ("INT", {"default": 4, "min": 0, "max": 9, "step": 1}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Noctyra/图片"

    def save_images(self, images, filename_prefix="Clean_Image", compress_level=4):
        if images is None or len(images) == 0:
            return {"ui": {"images": []}}

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                images[0].shape[1],
                images[0].shape[0]
            )
        )

        results = []
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            file = f"{filename}_{counter:05}_.png"

            # pnginfo=None 确保不写入元数据
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=None,
                compress_level=compress_level,
            )

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


# 节点映射
NODE_CLASS_MAPPINGS = {
    "SaveImageNoMetadata": SaveImageNoMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageNoMetadata": "Save Image (No Metadata)",
}
