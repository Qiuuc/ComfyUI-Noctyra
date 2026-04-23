"""
ComfyUI-Noctyra 插件

自定义节点集合：
- 图像处理节点（无元数据保存）
- 水印节点（含网格水印，随机偏移防重叠）
- 51EasyAI API 节点（文生图、图生图）

配套的模型管理器已独立为 ComfyUI-Noctyra-Manager 插件。
"""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__version__ = "1.0.0"

print(f"\033[34m[ComfyUI-Noctyra]\033[0m v{__version__} \033[92mLoaded\033[0m ({len(NODE_CLASS_MAPPINGS)} nodes)")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
