# ComfyUI-Noctyra — 动作(视频转 3D)· 模型自检
# GPL-3.0 (见仓库 LICENSE)
"""
插件加载时扫一遍 mocap_models_manifest.json,缺哪个权重就在控制台明确打印:
缺什么、去哪下、放到哪个绝对路径。不阻断加载(动作节点照样注册,真跑时才需要权重)。
"""
import json
import logging

from ._runtime import plugin_root
from . import _paths

logger = logging.getLogger("noctyra")


def manifest_path():
    return plugin_root() / "mocap_models_manifest.json"


def check(verbose=True):
    mf = manifest_path()
    if not mf.exists():
        return []
    try:
        items = json.loads(mf.read_text(encoding="utf-8")).get("items", [])
    except Exception as e:
        logger.warning(f"Mocap 清单读取失败: {e}")
        return []

    home = _paths.mocap_home()
    missing = [it for it in items if not (home / it["path"]).exists()]
    if not missing:
        if verbose:
            logger.info("Mocap 模型: 全部就位 ✓")
        return []

    if verbose:
        lic = [m for m in missing if m.get("license")]
        non = [m for m in missing if not m.get("license")]
        logger.info(f"Mocap 模型: 缺 {len(missing)} 项(动作节点要用,放齐再跑)。根目录: {home}")
        for m in non:
            logger.info(f"  ✗ {m['path']}  ({m.get('size','')})  ← {m.get('source','')}")
        for m in lic:
            logger.info(f"  ✗ {m['path']}  ← [license,自行注册] {m.get('source','')}")
    return missing
