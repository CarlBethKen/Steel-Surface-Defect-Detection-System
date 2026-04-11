钢铁表面缺陷检测系统 - 完整部署指南
=====================================

## 系统概述

本系统是一个基于深度学习的钢铁表面缺陷检测Web应用，支持以下功能：

### 核心功能
1. **检测操作主页** - 图像导入和检测操作
   - 支持上传带钢表面图像
   - 可选择模型 A 或模型 B 进行检测
   - 实时显示检测结果

2. **结果管理主页** - 查看检测统计和历史记录
   - 显示总记录数、缺陷总数、平均置信度等统计信息
   - 浏览历史检测记录
   - 按模型类型筛选
   - 支持记录详情查看

3. **缺陷检测** - 智能检测引擎
   - 集成两个已训练的模型（best.pt 和 model_b.pt）
   - 支持多类型缺陷检测

4. **检测结果展示** - 直观的结果呈现
   - 原图和标注图对比显示
   - 缺陷详情表格
   - 置信度可视化

5. **检测记录保存** - 持久化存储
   - 自动保存到 SQLite 数据库
   - 同时备份到 CSV 文件
   - 包含完整的元数据

6. **记录查看管理** - 灵活的查询管理
   - 查看历史检测记录
   - 按检测时间排序
   - 按模型类型筛选
   - 记录详情查看

7. **结果管理** - 操作接口
   - 查看单条记录详情
   - 删除已有记录
   - 批量管理功能

8. **统计分析** - 数据汇总
   - 总检测数
   - 按模型统计（A/B 使用次数）
   - 缺陷总数统计
   - 平均置信度
   - 缺陷类型分布

---

## 文件结构

```
steel_defect_web/
├── app.py                      # FastAPI 主应用
├── requirements.txt            # Python 依赖
├── detection_records.db        # SQLite 数据库（自动生成）
│
├── core/
│   ├── database.py            # 数据库 ORM 模型和操作
│   ├── infer.py               # 模型推理模块
│   ├── preprocess.py          # 图像预处理模块
│   ├── draw.py                # 检测结果绘制模块
│   └── storage.py             # CSV 存储模块
│
├── models/
│   ├── best.pt                # 模型 A（已训练）
│   └── model_b.pt             # 模型 B（已训练）
│
├── static/
│   ├── uploads/               # 上传的原始图像
│   ├── results/               # 检测结果标注图像
│   └── records.csv            # 检测记录备份
│
└── templates/
    ├── index.html             # 系统首页（可选）
    ├── detect.html            # 检测操作主页
    ├── result.html            # 检测结果展示页
    └── records.html           # 记录管理主页
```

---

## 安装与部署

### 1. 环境要求
- Python 3.8 或以上
- pip 包管理工具
- macOS/Linux/Windows

### 2. 安装依赖

```bash
cd /Volumes/KK/Final_project/steel_defect_web

# 安装 Python 依赖
pip install -r requirements.txt

# 或使用国内镜像（推荐，更快）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 配置模型

确保以下模型文件存在于 `models/` 目录：
- `best.pt` - 模型 A
- `model_b.pt` - 模型 B

如果模型路径不同，编辑 `app.py` 中的模型路径配置。

### 4. 运行应用

```bash
# 开发环境运行
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# 生产环境运行
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. 访问系统

打开浏览器，访问：
- **系统首页**: http://localhost:8000/
- **检测操作**: http://localhost:8000/ 或直接访问首页
- **记录管理**: http://localhost:8000/records

---

## API 接口文档

### 页面路由

| 方法 | URL | 说明 |
|------|-----|------|
| GET | `/` | 检测操作主页 |
| GET | `/records` | 检测记录管理页面 |

### 检测 API

#### POST /detect
上传图像并执行检测

**参数:**
- `image` (file): 图像文件
- `model_type` (string): 选择的模型 ("A" 或 "B")

**返回:** 检测结果页面 (HTML)

### 数据 API

#### GET /api/records
获取检测记录列表

**查询参数:**
- `model_type` (optional): 过滤模型类型 ("A" 或 "B")
- `skip` (optional): 跳过的记录数，默认 0
- `limit` (optional): 返回的最大记录数，默认 100

**返回:**
```json
{
  "total": 100,
  "records": [
    {
      "id": 1,
      "file_id": "uuid-string",
      "image_name": "test.jpg",
      "image_path": "/static/uploads/...",
      "result_image_path": "/static/results/...",
      "model_type": "A",
      "defect_count": 2,
      "defections": [
        {"label": "scratch", "score": 0.92, "bbox": [10, 20, 200, 180]}
      ],
      "confidence_avg": 0.92,
      "created_at": "2026-03-10 12:30:45"
    }
  ]
}
```

#### GET /api/records/{record_id}
获取单条记录详情

**返回:** 记录详情 JSON

#### DELETE /api/records/{record_id}
删除检测记录及相关文件

**返回:**
```json
{
  "status": "success",
  "message": "记录已删除"
}
```

#### GET /api/statistics
获取统计数据

**返回:**
```json
{
  "total_records": 100,
  "model_a_count": 60,
  "model_b_count": 40,
  "total_defects": 150,
  "average_confidence": 0.87,
  "defect_distribution": {
    "scratch": 80,
    "pit": 45,
    "patch": 25
  }
}
```

---

## 模型集成指南

### 自定义推理代码

编辑 `core/infer.py` 中的 `run_infer` 函数，根据您的模型类型进行调整：

#### 如果是 YOLOv5：

```python
def run_infer(models, img, model_type="A"):
    model = models.get(model_type)
    
    if model is None:
        return []
    
    with torch.no_grad():
        results = model(img)
        detections = results.pred  # [N, 6]
        
        detection_list = []
        for det in detections[0]:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            detection_list.append({
                "label": model.names[int(cls)],
                "score": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
        return detection_list
```

#### 如果是 Faster R-CNN：

```python
def run_infer(models, img, model_type="A"):
    model = models.get(model_type)
    
    if model is None:
        return []
    
    model.eval()
    with torch.no_grad():
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        outputs = model(img_tensor)
        
        detection_list = []
        boxes = outputs[0]['boxes']
        scores = outputs[0]['scores']
        labels = outputs[0]['labels']
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.cpu().numpy()
            detection_list.append({
                "label": f"class_{int(label)}",
                "score": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
        return detection_list
```

### 图像预处理

编辑 `core/preprocess.py` 进行自定义预处理：

```python
def preprocess_image(image_path: str, target_size: tuple = None):
    img = cv2.imread(image_path)
    
    # 自定义处理
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    
    return img
```

---

## 数据库管理

### SQLite 数据库

数据库自动创建于 `detection_records.db`，包含表：
- `detection_records` - 检测记录表

### 备份 CSV

每次检测自动备份到 `static/records.csv`

### 导出数据

```bash
# 查看 CSV 记录
cat static/records.csv

# 导出为 Excel（使用其他工具）
python -c "import pandas; df = pandas.read_csv('static/records.csv'); df.to_excel('records.xlsx')"
```

---

## 故障排除

### 问题 1: 模型加载失败

**错误**: `模型 A 文件不存在`

**解决**:
1. 检查 `models/best.pt` 是否存在
2. 检查文件权限
3. 检查绝对路径是否正确

### 问题 2: 图像上传失败

**错误**: `无法读取图片`

**解决**:
1. 检查图像格式（支持 jpg, png, bmp 等）
2. 检查 `static/uploads` 文件夹是否可写
3. 检查磁盘空间是否充足

### 问题 3: 数据库错误

**错误**: `数据库已被锁定`

**解决**:
1. 重启应用
2. 删除 `detection_records.db` 文件（会丢失历史记录）
3. 检查文件权限

### 问题 4: 推理速度慢

**优化方法**:
1. 增加 `--workers` 数量
2. 使用 GPU（安装 CUDA）
3. 优化图像尺寸
4. 使用模型 B（如果精度可接受）

---

## 性能优化

### 生产环境配置

```bash
# 使用 Gunicorn + Uvicorn
pip install gunicorn

gunicorn app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --worker-connections 1000
```

### 启用 GPU 加速

1. 安装 CUDA 和 cuDNN
2. 修改 `core/infer.py`:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### 缓存优化

启用浏览器缓存和 CDN：

```python
@app.middleware("http")
async def add_cache_headers(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "public, max-age=86400"
    return response
```

---

## 安全建议

1. **生产环境**:
   - 使用 HTTPS
   - 启用身份验证
   - 限制上传文件大小
   - 定期备份数据库

2. **数据隐私**:
   - 定期清理旧记录
   - 加密敏感数据
   - 限制 API 访问速率

3. **系统监控**:
   - 监控磁盘空间
   - 监控 CPU/内存使用
   - 记录访问日志

---

## 常见问题 (FAQ)

### Q: 如何修改系统的界面主题？

A: 编辑 `templates/detect.html`, `templates/records.html`, `templates/result.html` 中的 CSS 样式部分。

### Q: 可以同时使用多个模型吗？

A: 可以。系统支持模型 A 和 B 的切换和对比。

### Q: 如何备份数据？

A: 
- 数据库: 复制 `detection_records.db`
- CSV: 复制 `static/records.csv`
- 图像: 复制 `static/uploads` 和 `static/results`

### Q: 如何清理过期记录？

A: 使用删除 API 或数据库工具直接删除。建议定期清理以节省空间。

### Q: 支持批量检测吗？

A: 当前版本支持单张图像检测。批量检测功能可联系开发者定制。

---

## 技术栈

- **后端**: FastAPI + Uvicorn
- **数据库**: SQLite + SQLAlchemy
- **深度学习**: PyTorch
- **图像处理**: OpenCV
- **前端**: HTML5 + CSS3 + JavaScript (原生)
- **模板引擎**: Jinja2

---

## 许可证

本项目用于钢铁表面缺陷检测研究。

---

## 支持与反馈

如有问题或建议，请联系开发团队。

最后更新: 2026-03-10

