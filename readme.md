# YOLO Instance Segmentation Backend for Label Studio
# YOLO 实例分割标注后端服务

## Description 描述

This repository contains a generic YOLO (You Only Look Once) backend implementation for Label Studio, specifically designed for instance segmentation tasks. It provides a machine learning backend that integrates YOLO instance segmentation capabilities with Label Studio's annotation platform, allowing for pixel-level annotation and prediction of individual object instances.

本仓库包含一个通用的 YOLO（You Only Look Once）后端实现，用于 Label Studio 平台的实例分割任务。它提供了一个机器学习后端，将 YOLO 实例分割功能与 Label Studio 标注平台进行集成，支持像素级别的对象实例标注和预测。

## Features 特性

- 🔄 Automatic model version control and status tracking
- 🎭 Instance segmentation with YOLO models (using polygon masks)
- 🤝 Seamless integration with Label Studio for pixel-level annotation
- 🚀 Docker support for easy deployment
- 📊 Automatic train/validation split
- ✨ Real-time inference capabilities
- 📋 Model registry for tracking segmentation model performance

---

- 🔄 自动模型版本控制和状态跟踪
- 🎭 使用 YOLO 模型进行实例分割（使用多边形掩码）
- 🤝 与 Label Studio 无缝集成, 实现像素级别标注和半自动化训练
- 🚀 支持 Docker 部署
- 📊 自动训练/验证集划分
- ✨ 实时推理能力
- 📋 模型注册表，用于跟踪分割模型性能

## Requirements 环境要求

- Python 3.8+
- Label Studio (Now only support 1.10.3)
- Docker (optional 可选)
- CUDA compatible GPU (recommended 推荐)

## Installation 安装

### Using Docker 使用 Docker

```bash
docker-compose up -d
```

### Manual Installation 手动安装

1. Install requirements 安装依赖:
```bash
pip install -r requirements.txt
```

2. Set environment variables 设置环境变量:
```bash
export LABEL_STUDIO_HOST=<your-label-studio-host>
export LABEL_STUDIO_API_KEY=<your-api-key>
```

## Project Structure 项目结构

```
├── yolo.py              # Main YOLO segmentation backend implementation 主要的YOLO分割后端实现
├── yolo_config.py       # Configuration settings 配置设置
├── _wsgi.py             # WSGI entry point WSGI入口点
├── requirements.txt     # Python dependencies Python依赖
├── Dockerfile           # Docker configuration Docker配置
└── docker-compose.yml   # Docker Compose configuration Docker Compose配置
```

## Configuration 配置

The model can be configured through `yolo_config.py`. Key settings include:
模型可以通过 `yolo_config.py` 进行配置。主要设置包括：

- Model version (must be a segmentation model, e.g., yolo11m-seg) 模型版本（必须是分割模型，例如 yolo11m-seg）
- Training parameters 训练参数
- Prediction configuration 预测配置
- Fine-tuning settings 微调设置

## Label Studio Configuration Label Studio 配置

For instance segmentation, you should use the PolygonLabels control in your Label Studio configuration. Example:
对于实例分割任务，您应该在 Label Studio 配置中使用 PolygonLabels 控件。

## Usage 使用方法

1. Start Label Studio and create a project with PolygonLabels configuration
   启动 Label Studio 并创建带有 PolygonLabels 配置的项目

2. Configure the ML Backend in Label Studio settings
   在 Label Studio 设置中配置机器学习后端

3. Start annotating and the model will automatically:
   开始标注，模型将自动：
   - Train on your polygon annotations 基于您的多边形标注进行训练
   - Provide segmentation predictions for new images 为新图像提供分割预测
   - Track model versions 跟踪模型版本


## API Integration API集成

The backend implements the Label Studio ML Backend API with three main endpoints:
后端实现了 Label Studio ML Backend API，包含三个主要端点：

- `/predict` - For segmentation predictions 用于分割预测
- `/train` - For model training 用于模型训练
- `/health` - For health checks 用于健康检查

## License 许可证

MIT

## Support 支持

For issues and feature requests, please create an issue in the repository.
如有问题和功能请求，请在仓库中创建 issue。
