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
视频处理相关节点
"""
import logging
import os

import folder_paths
from comfy_api.util import VideoCodec, VideoContainer

logger = logging.getLogger("noctyra")


class SaveVideoNoMetadata:
    """保存视频（无元数据）节点

    与 ComfyUI 自带 SaveVideo 等价，但强制不写入 prompt / workflow 元数据。
    无论是否启动 --disable-metadata，都不会向容器（mp4 udta / mkv tag）写任何 JSON。
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": "video/Clean_Video"}),
                "format": (VideoContainer.as_input(), {"default": "auto"}),
                "codec": (VideoCodec.as_input(), {"default": "auto"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "Noctyra/视频"

    def save_video(self, video, filename_prefix="video/Clean_Video", format="auto", codec="auto"):
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix,
                self.output_dir,
                width,
                height,
            )
        )

        ext = VideoContainer.get_extension(format) or "mp4"
        file = f"{filename}_{counter:05}_.{ext}"

        # metadata=None 强制跳过 prompt / extra_pnginfo 写入
        video.save_to(
            os.path.join(full_output_folder, file),
            format=VideoContainer(format),
            codec=codec,
            metadata=None,
        )

        return {
            "ui": {
                "images": [{"filename": file, "subfolder": subfolder, "type": self.type}],
                "animated": (True,),
            }
        }


NODE_CLASS_MAPPINGS = {
    "SaveVideoNoMetadata": SaveVideoNoMetadata,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveVideoNoMetadata": "Save Video (No Metadata)",
}
