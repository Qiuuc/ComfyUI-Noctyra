# ComfyUI-Noctyra

ComfyUI 自定义节点合集。

- **图像处理**：`SaveImageNoMetadata`（保存图片不写入任何元数据）
- **水印**：`AddWatermark`、`AddGridWatermark`（网格水印，随机偏移防重叠）
- **AI 服务**：51EasyAI API 文生图 / 图生图节点

> 模型管理器已独立为 [ComfyUI-Noctyra-Manager](https://github.com/Qiuuc/ComfyUI-Noctyra-Manager)。

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Qiuuc/ComfyUI-Noctyra.git
cd ComfyUI-Noctyra
pip install -r requirements.txt
```

重启 ComfyUI，控制台出现 `[ComfyUI-Noctyra] v1.0.0 Loaded` 即成功。

---

## 使用

在 ComfyUI 节点菜单 `Noctyra/` 分类下查看全部节点。`workflows/` 目录提供示例工作流。

---

## 许可证

[MIT](LICENSE)
