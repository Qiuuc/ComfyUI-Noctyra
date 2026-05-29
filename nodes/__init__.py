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
Noctyra 节点模块

按功能分类：
- image: 图片处理节点
- video: 视频处理节点
- watermark: 水印处理节点
- watermark_grid: 网格水印节点（随机偏移防重叠）
- dewatermark: 去除可见水印节点（通用区域修复）
- dewatermark_gemini: 去除 Gemini 水印（反向 alpha 还原）
- dewatermark_doubao: 去除豆包 AIGC 文字条水印
- dewatermark_invisible: 去除隐形水印（SDXL 低强度 img2img 重生成）
- identify: AI 溯源鉴定（C2PA/IPTC/SynthID/可见水印）
- model_downloader: 隐形水印扩散模型自动下载
- easyai_api: AI 服务节点
"""
from . import (
    image, video, watermark, watermark_grid,
    dewatermark, dewatermark_gemini, dewatermark_doubao, dewatermark_invisible,
    identify, model_downloader, easyai_api,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod in (image, video, watermark, watermark_grid,
             dewatermark, dewatermark_gemini, dewatermark_doubao, dewatermark_invisible,
             identify, model_downloader, easyai_api):
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
