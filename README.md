# ComfyUI-Noctyra

ComfyUI 自定义节点合集。

- **图像处理**：`SaveImageNoMetadata`（保存图片不写入任何元数据）
- **水印**：`AddWatermark`、`AddGridWatermark`（网格水印，随机偏移防重叠）
- **AI 服务**：51EasyAI API 文生图 / 图生图节点
- **视频转 3D 动作（Mocap）**：单目视频 → SMPL-X 骨骼动作 → BVH / GLB（见下）

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Qiuuc/ComfyUI-Noctyra.git
cd ComfyUI-Noctyra
pip install -r requirements.txt
```


## 视频转 3D 动作（Mocap）

单目视频里抽出 3D 骨骼动作，导出BVH 和带蒙皮 GLB。


### 安装（一次）

GVHMR/HaMeR 与 ComfyUI 主环境版本互斥，故跑在插件私有 sidecar（py3.10/torch2.3.1）。

```bash
python_embeded\python.exe ComfyUI\custom_nodes\ComfyUI-Noctyra\install.py
```

### 模型下载

 `ComfyUI-Noctyra/mocap/` 下对应目录。
**插件加载时会自检**，缺哪个就在控制台打印「缺什么 / 去哪下 / 放哪」。完整清单见
[`mocap_models_manifest.json`](mocap_models_manifest.json)。其中三件受 MPI license 限制、
**禁止再分发，须自行注册下载**：

| 文件 | 注册地址 | 放到 `mocap/` 下 |
|---|---|---|
| SMPL | https://smpl.is.tue.mpg.de | `GVHMR/inputs/checkpoints/body_models/smpl/` |
| SMPL-X | https://smpl-x.is.tue.mpg.de | `GVHMR/inputs/checkpoints/body_models/smplx/` |
| MANO | https://mano.is.tue.mpg.de | `hamer/_DATA/data/mano/` |

其余权重为 GVHMR / HaMeR 官方公开模型，可从其官方页或 hf-mirror.com 镜像获取。
GVHMR、HaMeR 代码本身亦放在 `mocap/GVHMR`、`mocap/hamer`。

---

## 许可证

[GPL-3.0-or-later](LICENSE)
