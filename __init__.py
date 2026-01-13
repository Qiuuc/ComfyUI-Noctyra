from .nodes import SaveImageNoMetadata

# 映射类名
NODE_CLASS_MAPPINGS = {
    "SaveImageNoMetadata": SaveImageNoMetadata
}

# 映射显示名称 (UI上看到的名字)
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveImageNoMetadata": "Save Image (No Metadata)"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]