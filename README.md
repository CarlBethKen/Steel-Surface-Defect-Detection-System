# 钢铁表面缺陷检测系统

基于 YOLOv8 深度学习模型的钢铁表面缺陷检测 Web 应用。

## 功能

- **实时监测** — 调用摄像头实时检测钢铁表面缺陷，检测到缺陷自动保存记录
- **图片检测** — 上传单张/批量图片进行缺陷检测，支持拖拽和文件夹上传
- **历史记录** — 查看、管理所有检测记录，支持批量删除和导出
- **统计分析** — 检测总数、缺陷分布、平均置信度等数据统计

## 项目结构

```
├── app.py                  # FastAPI 后端主应用
├── run.py                  # 一键启动脚本
├── requirements.txt        # Python 依赖
├── core/
│   ├── infer.py           # YOLOv8 模型推理
│   ├── preprocess.py      # 图像预处理
│   ├── draw.py            # 检测结果绘制
│   ├── database.py        # SQLite 数据库操作
│   └── storage.py         # CSV 备份存储
├── models/
│   └── best.pt            # YOLOv8 训练权重
├── templates/
│   └── system.html        # 前端页面
└── static/
    ├── uploads/           # 上传图片（运行时生成）
    └── results/           # 检测结果图（运行时生成）
```

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 放置模型

将训练好的 YOLOv8 权重文件 `best.pt` 放入 `models/` 目录。

### 3. 启动系统

```bash
python run.py
```

系统会自动启动服务并打开浏览器访问 http://localhost:8000/system

## 技术栈

- **后端**: FastAPI + Uvicorn
- **模型**: YOLOv8 (ultralytics)
- **数据库**: SQLite + SQLAlchemy
- **图像处理**: OpenCV
- **前端**: HTML5 + CSS3 + JavaScript
