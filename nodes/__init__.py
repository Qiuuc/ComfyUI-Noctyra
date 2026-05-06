"""
Noctyra 节点模块

按功能分类：
- image: 图片处理节点
- video: 视频处理节点
- watermark: 水印处理节点
- watermark_grid: 网格水印节点（随机偏移防重叠）
- easyai_api: AI 服务节点
"""
from . import image, video, watermark, watermark_grid, easyai_api

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod in (image, video, watermark, watermark_grid, easyai_api):
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
