"""
Noctyra 节点模块

按功能分类：
- image: 图片处理节点
- watermark: 水印处理节点
- watermark_grid: 网格水印节点（随机偏移防重叠）
- easyai_api: AI 服务节点
"""

from .image import NODE_CLASS_MAPPINGS as ImageMappings
from .image import NODE_DISPLAY_NAME_MAPPINGS as ImageDisplayMappings
from .watermark import NODE_CLASS_MAPPINGS as WatermarkMappings
from .watermark import NODE_DISPLAY_NAME_MAPPINGS as WatermarkDisplayMappings
from .watermark_grid import NODE_CLASS_MAPPINGS as GridWatermarkMappings
from .watermark_grid import NODE_DISPLAY_NAME_MAPPINGS as GridWatermarkDisplayMappings
from .easyai_api import NODE_CLASS_MAPPINGS as AIMappings
from .easyai_api import NODE_DISPLAY_NAME_MAPPINGS as AIDisplayMappings

# 合并所有节点映射
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(ImageMappings)
NODE_CLASS_MAPPINGS.update(WatermarkMappings)
NODE_CLASS_MAPPINGS.update(GridWatermarkMappings)
NODE_CLASS_MAPPINGS.update(AIMappings)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(ImageDisplayMappings)
NODE_DISPLAY_NAME_MAPPINGS.update(WatermarkDisplayMappings)
NODE_DISPLAY_NAME_MAPPINGS.update(GridWatermarkDisplayMappings)
NODE_DISPLAY_NAME_MAPPINGS.update(AIDisplayMappings)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
