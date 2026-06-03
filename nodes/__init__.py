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

- image / video      : 无元数据保存
- easyai_api         : AI 服务节点
- watermark_add/     : 添加水印（图像/网格/视频/全屏/文字）
- watermark_remove/  : 去除水印 + 溯源 + 模型下载
"""
import logging
import sys

# 统一控制台输出：所有 logging.getLogger("noctyra") 的消息自动加 [Noctyra] 前缀。
_nlog = logging.getLogger("noctyra")
if not getattr(_nlog, "_noctyra_configured", False):
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[Noctyra] %(message)s"))
    _nlog.addHandler(_h)
    _nlog.setLevel(logging.INFO)
    _nlog.propagate = False
    _nlog._noctyra_configured = True

from . import image, video, easyai_api, watermark_add, watermark_remove

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for _mod in (image, video, easyai_api, watermark_add, watermark_remove):
    NODE_CLASS_MAPPINGS.update(getattr(_mod, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_mod, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
