"""
AI 服务相关节点 - 51EasyAI API
"""
import logging
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image

logger = logging.getLogger("noctyra")


# ---------- 内部工具 ----------

# 复用 HTTP 连接（keep-alive），所有节点共用
_session = requests.Session()


def _normalize_base_url(url):
    """规范化 base_url：去掉末尾斜杠后确保以 /v1 结尾。

    避免用户填写 'http://x/v1/' 时被错判为不含 /v1 而拼成 '//v1'。
    """
    url = (url or "").strip().rstrip("/")
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def _build_headers(token, json_body=False):
    headers = {}
    if json_body:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _download_image(url, timeout=30):
    """下载图片并转换为 ComfyUI 张量格式"""
    try:
        response = _session.get(url, timeout=timeout)
        response.raise_for_status()

        # 校验响应是图片而非 JSON 错误页
        ctype = response.headers.get("Content-Type", "").lower()
        if ctype and not ctype.startswith(("image/", "application/octet-stream")):
            logger.error(f"下载图片失败: {url}, 返回非图片内容 Content-Type={ctype}")
            return None

        with BytesIO(response.content) as buf:
            img = Image.open(buf)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = np.array(img).astype(np.float32) / 255.0

        return torch.from_numpy(img_array).unsqueeze(0)

    except requests.Timeout:
        logger.error(f"下载图片超时（{timeout}s）: {url}")
    except requests.RequestException as e:
        logger.error(f"下载图片网络错误: {url}, {e}")
    except Exception as e:
        logger.error(f"下载图片失败: {url}, 错误: {e}")
    return None


def _parse_and_download(result, action_name):
    """解析 EasyAI 响应并下载所有 image 类型条目，返回 IMAGE 张量"""
    if result.get("status") != "success":
        logger.error(f"{action_name}失败: {result}")
        raise Exception(f"{action_name}图片失败")

    result_data = result.get("data", [])
    if not result_data:
        logger.error(f"无图片数据: {result}")
        raise Exception("无图片返回")

    images = []
    for item in result_data:
        if item.get("type") == "image" and "url" in item:
            img_tensor = _download_image(item["url"])
            if img_tensor is not None:
                images.append(img_tensor)

    if not images:
        raise Exception("下载图片失败")

    return torch.cat(images, dim=0)


def _add_optional_str(payload, **kwargs):
    """把非空字符串值写入 payload（None / 空白会被跳过）"""
    for key, val in kwargs.items():
        if val is not None and str(val).strip():
            payload[key] = str(val).strip()


# ---------- 选择器节点 ----------

class EasyAIAspectRatioSelector:
    """EasyAI 宽高比选择器

    用于选择图片宽高比，Auto 返回 None
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ([
                    "Auto",
                    "1:1",
                    "4:3",
                    "3:4",
                    "3:2",
                    "2:3",
                    "16:9",
                    "9:16",
                ],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("aspect_ratio",)
    FUNCTION = "get_aspect_ratio"
    CATEGORY = "Noctyra/Easy AI API"
    OUTPUT_NODE = False

    def get_aspect_ratio(self, aspect_ratio):
        """返回选中的宽高比，Auto 返回空字符串（表示不设置）"""
        if aspect_ratio == "Auto":
            return ("",)
        return (aspect_ratio,)


class EasyAIResolutionSelector:
    """EasyAI 分辨率选择器"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (["1K", "2K", "4K"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("resolution",)
    FUNCTION = "get_resolution"
    CATEGORY = "Noctyra/Easy AI API"
    OUTPUT_NODE = False

    def get_resolution(self, resolution):
        return (resolution,)


class EasyAIModelSelector:
    """EasyAI 模型选择器"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "即梦V4.0图像生成及编辑",
                    "Nano Banana 2",
                    "即梦 图片 4.0",
                ],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model",)
    FUNCTION = "get_model"
    CATEGORY = "Noctyra/Easy AI API"
    OUTPUT_NODE = False

    def get_model(self, model):
        return (model,)


class AIConfig:
    """AI API 配置节点

    用于配置 API 的 base_url 和 token
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": "http://127.0.0.1:3001"}),
                "token": ("STRING", {"default": ""}),
                "timeout": ("INT", {"default": 120, "min": 10, "max": 1800, "step": 10}),
            }
        }

    RETURN_TYPES = ("CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "create_config"
    CATEGORY = "Noctyra/Easy AI API"
    OUTPUT_NODE = False

    def create_config(self, base_url, token, timeout=120):
        return ({
            "base_url": base_url.strip(),
            "token": token.strip(),
            "timeout": timeout,
        },)


# ---------- 文生图 ----------

class AIImageGenerator:
    """AI 绘图节点 - 51EasyAI API"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("CONFIG",),
                "prompt": ("STRING", {"multiline": True, "default": "a beautiful cat"}),
                "model": ("STRING", {"forceInput": True}),
            },
            "optional": {
                "size": ("STRING", {"forceInput": True}),
                "aspect_ratio": ("STRING", {"forceInput": True}),
                "resolution": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_image"
    CATEGORY = "Noctyra/Easy AI API"
    OUTPUT_NODE = True

    def generate_image(self, config, prompt, model, size=None, aspect_ratio=None, resolution=None):
        base_url = _normalize_base_url(config.get("base_url", "http://127.0.0.1:3001"))
        token = config.get("token", "")
        timeout = config.get("timeout", 120)

        payload = {"prompt": prompt, "model": model}
        _add_optional_str(payload, size=size, aspect_ratio=aspect_ratio, resolution=resolution)

        logger.info(f"提交任务: {payload}")

        try:
            response = _session.post(
                f"{base_url}/images/generations",
                headers=_build_headers(token, json_body=True),
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            logger.debug(f"响应: {result}")
            return (_parse_and_download(result, "生成"),)

        except Exception as e:
            logger.error(f"错误: {e}")
            raise Exception(f"生成图片失败: {e}")


# ---------- 图生图 ----------

class EasyAIImageEditor:
    """EasyAI 图片编辑节点

    支持基于参考图片的编辑和重绘
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("CONFIG",),
                "model": ("STRING", {"forceInput": True}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "edit the image"}),
            },
            "optional": {
                "size": ("STRING", {"forceInput": True}),
                "aspect_ratio": ("STRING", {"forceInput": True}),
                "resolution": ("STRING", {"forceInput": True}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1,
                    "forceInput": True,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "input_image")
    FUNCTION = "edit_image"
    CATEGORY = "Noctyra/Easy AI API"
    OUTPUT_NODE = True

    def edit_image(self, config, image, prompt, model, size=None,
                   aspect_ratio=None, resolution=None, seed=-1):
        base_url = _normalize_base_url(config.get("base_url", "http://127.0.0.1:3001"))
        token = config.get("token", "")
        timeout = config.get("timeout", 120)

        # 处理输入图片 - 支持批量上传（最多 16 张）
        files = []
        if image is not None:
            batch_size = image.shape[0] if len(image.shape) == 4 else 1
            for i in range(min(batch_size, 16)):
                img_tensor = image[i] if len(image.shape) == 4 else image
                img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_array)
                # with 块确保 BytesIO 释放，大批量图片不累积句柄
                with BytesIO() as img_buffer:
                    img_pil.save(img_buffer, format="PNG")
                    files.append(("image[]", (f"image_{i}.png", img_buffer.getvalue(), "image/png")))

        data = {"prompt": prompt, "model": model}
        _add_optional_str(data, size=size, aspect_ratio=aspect_ratio, resolution=resolution)
        if seed is not None and seed >= 0:
            data["seed"] = str(seed)

        logger.info(f"提交编辑任务: model={model}, prompt={prompt[:50]}...")

        try:
            endpoint = f"{base_url}/images/edits"
            logger.debug(f"调用: {endpoint}")
            response = _session.post(
                endpoint,
                headers=_build_headers(token),
                data=data,
                files=files or None,
                timeout=timeout,
            )

            logger.debug(f"响应状态: {response.status_code}")
            logger.debug(f"响应内容: {response.text[:1000]}")

            if response.status_code not in (200, 201):
                raise Exception(f"API 错误 {response.status_code}: {response.text[:500]}")
            result = response.json()
            logger.debug(f"响应: {result}")

            images_tensor = _parse_and_download(result, "编辑")
            # 返回生成结果 + 输入图片（用于确认）
            return (images_tensor, image)

        except Exception as e:
            logger.error(f"错误: {e}")
            raise Exception(f"编辑图片失败: {e}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "EasyAIAspectRatioSelector": EasyAIAspectRatioSelector,
    "EasyAIResolutionSelector": EasyAIResolutionSelector,
    "EasyAIModelSelector": EasyAIModelSelector,
    "AIConfig": AIConfig,
    "AIImageGenerator": AIImageGenerator,
    "EasyAIImageEditor": EasyAIImageEditor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasyAIAspectRatioSelector": "EasyAI Aspect Ratio Selector",
    "EasyAIResolutionSelector": "EasyAI Resolution Selector",
    "EasyAIModelSelector": "EasyAI Model Selector",
    "AIConfig": "EasyAI API Config",
    "AIImageGenerator": "EasyAI Image Generator",
    "EasyAIImageEditor": "EasyAI Image Editor",
}
