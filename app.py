"""
钢铁表面缺陷检测系统 - FastAPI 后端
"""
import os
import uuid
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, Depends, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from sqlalchemy.orm import Session
import zipfile
from datetime import datetime

from core.preprocess import preprocess_image
from core.infer import load_models, run_infer
from core.draw import draw_detections
from core.storage import save_record_csv
from core.database import (
    get_db,
    save_record,
    get_all_records,
    get_record_by_id,
    delete_record,
    get_statistics,
)

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "static", "results")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# FastAPI 应用初始化
app = FastAPI(title="钢铁表面缺陷检测系统", version="1.0.0")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# 模型加载
models = load_models()

# ==================== 前端页面路由 ====================

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """主页 - 检测操作界面"""
    return templates.TemplateResponse("detect.html", {"request": request})


@app.get("/records", response_class=HTMLResponse)
def records_page(request: Request):
    """历史记录管理页面"""
    return templates.TemplateResponse("records.html", {"request": request})


@app.get("/system", response_class=HTMLResponse)
def system_page(request: Request):
    """系统主页面"""
    return templates.TemplateResponse("system.html", {"request": request})


# ==================== 检测 API ====================

@app.post("/detect")
async def detect(
    request: Request,
    image: UploadFile = File(...),
    model_type: str = Form("yolov8m"),
    db: Session = Depends(get_db),
):
    """
    执行图像检测（返回 JSON），兼容前端 fetch 调用
    """
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(image.filename)[-1].lower() or ".jpg"
    upload_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    result_path = os.path.join(RESULT_DIR, f"{file_id}_result{ext}")

    # 保存上传的文件
    content = await image.read()
    with open(upload_path, "wb") as f:
        f.write(content)

    # 图像预处理
    img = preprocess_image(upload_path)

    # 运行推理
    detections = run_infer(models, img, model_type=model_type)

    # 绘制检测结果
    try:
        draw_detections(upload_path, result_path, detections)
    except Exception as e:
        print(f"绘制检测结果出错: {e}")

    # 保存记录到数据库（尽量不阻塞核心返回）
    try:
        record = save_record(
            db=db,
            file_id=file_id,
            image_name=image.filename,
            image_path=f"/static/uploads/{file_id}{ext}",
            result_image_path=f"/static/results/{file_id}_result{ext}",
            model_type=model_type,
            detections=detections,
        )
    except Exception as e:
        print(f"保存记录出错: {e}")
        record = None

    # 同时保存到 CSV（备份）
    try:
        save_record_csv(
            csv_path=os.path.join(BASE_DIR, "static", "records.csv"),
            image_name=image.filename,
            model_type=model_type,
            detections=detections,
            result_image=f"/static/results/{file_id}_result{ext}",
        )
    except Exception as e:
        print(f"保存 CSV 出错: {e}")

    # 返回 JSON，供前端处理
    resp = {
        "record_id": record.id if record is not None else None,
        "file_id": file_id,
        "orig_url": f"/static/uploads/{file_id}{ext}",
        "result_url": f"/static/results/{file_id}_result{ext}",
        "detections": detections,
        "model_type": model_type,
        "defect_count": len(detections),
    }

    return JSONResponse(resp)


@app.post("/detect/batch")
async def detect_batch(
    model_type: str = Form("yolov8m"),
    images: list = File(...),
    db: Session = Depends(get_db),
):
    """批量检测图像"""
    results = []

    if not images:
        return JSONResponse({"results": []})

    for image in images:
        try:
            file_id = str(uuid.uuid4())
            ext = os.path.splitext(image.filename)[-1].lower() or ".jpg"
            upload_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
            result_path = os.path.join(RESULT_DIR, f"{file_id}_result{ext}")

            content = await image.read()
            with open(upload_path, "wb") as f:
                f.write(content)

            img = preprocess_image(upload_path)
            detections = run_infer(models, img, model_type=model_type)

            try:
                draw_detections(upload_path, result_path, detections)
            except Exception as e:
                print(f"绘制检测结果出错: {e}")

            try:
                record = save_record(
                    db=db,
                    file_id=file_id,
                    image_name=image.filename,
                    image_path=f"/static/uploads/{file_id}{ext}",
                    result_image_path=f"/static/results/{file_id}_result{ext}",
                    model_type=model_type,
                    detections=detections,
                )
            except Exception as e:
                print(f"保存记录出错: {e}")
                record = None

            results.append({
                "record_id": record.id if record is not None else None,
                "file_id": file_id,
                "filename": image.filename,
                "result_url": f"/static/results/{file_id}_result{ext}",
                "defect_count": len(detections),
                "status": "success"
            })
        except Exception as e:
            print(f"处理错误: {e}")
            results.append({
                "filename": getattr(image, 'filename', 'unknown'),
                "status": "error",
                "message": str(e)
            })

    return JSONResponse({"results": results})


@app.post("/api/export/batch")
async def export_batch(db: Session = Depends(get_db)):
    """批量导出标注图片（ZIP格式）"""
    try:
        import tempfile

        # 创建临时 ZIP 文件
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_path = tmp.name

        with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 获取所有结果图片
            result_dir = RESULT_DIR
            for filename in os.listdir(result_dir):
                if "_result" in filename:
                    file_path = os.path.join(result_dir, filename)
                    arcname = f"results/{filename}"
                    zf.write(file_path, arcname)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return FileResponse(
            tmp_path,
            filename=f"detection_results_{timestamp}.zip",
            media_type="application/zip",
            background=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/records/batch")
async def delete_records_batch(request: Request, db: Session = Depends(get_db)):
    """批量删除记录"""
    try:
        body = await request.json()
        record_ids = body.get('record_ids', [])

        deleted_count = 0
        for record_id in record_ids:
            try:
                record = get_record_by_id(db, record_id)
                if record:
                    # 删除相关文件
                    try:
                        if record.image_path:
                            img_file = os.path.join(BASE_DIR, record.image_path.lstrip("/"))
                            if os.path.exists(img_file):
                                os.remove(img_file)
                        if record.result_image_path:
                            result_file = os.path.join(BASE_DIR, record.result_image_path.lstrip("/"))
                            if os.path.exists(result_file):
                                os.remove(result_file)
                    except Exception as e:
                        print(f"删除文件时出错: {e}")

                    # 删除数据库记录
                    if delete_record(db, record_id):
                        deleted_count += 1
            except Exception as e:
                print(f"删除记录 {record_id} 失败: {e}")

        return JSONResponse({
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"已删除 {deleted_count} 条记录"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ==================== 记录查询 API ====================

@app.get("/api/records")
def get_records(
    model_type: str = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    获取历史检测记录列表

    参数:
        - model_type: 过滤模型类型 (A 或 B)
        - skip: 跳过的记录数
        - limit: 返回的最大记录数

    返回:
        - JSON 格式的记录列表
    """
    result = get_all_records(db, model_type=model_type, skip=skip, limit=limit)
    return JSONResponse(result)


@app.get("/api/records/{record_id}")
def get_record(record_id: int, db: Session = Depends(get_db)):
    """获取单条记录详情"""
    record = get_record_by_id(db, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")
    return record.to_dict()


# ==================== 记录删除 API ====================

@app.delete("/api/records/{record_id}")
def delete_record_api(record_id: int, db: Session = Depends(get_db)):
    """
    删除检测记录（包括相关的图像文件）

    参数:
        - record_id: 记录 ID

    返回:
        - 删除结果
    """
    record = get_record_by_id(db, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="记录不存在")

    # 删除相关的图像文件
    try:
        # 删除原始图像
        if record.image_path:
            orig_file = os.path.join(BASE_DIR, record.image_path.lstrip("/"))
            if os.path.exists(orig_file):
                os.remove(orig_file)

        # 删除结果图像
        if record.result_image_path:
            result_file = os.path.join(BASE_DIR, record.result_image_path.lstrip("/"))
            if os.path.exists(result_file):
                os.remove(result_file)
    except Exception as e:
        print(f"删除文件时出错: {e}")

    # 删除数据库记录
    success = delete_record(db, record_id)

    if success:
        return JSONResponse({"status": "success", "message": "记录已删除"})
    else:
        raise HTTPException(status_code=500, detail="删除记录失败")


# ==================== 统计分析 API ====================

@app.get("/api/statistics")
def get_stats(db: Session = Depends(get_db)):
    """
    获取检测统计信息

    返回:
        - JSON 格式的统计数据，包括：
          - total_records: 总记录数
          - model_a_count: 模型 A 使用次数
          - model_b_count: 模型 B 使用次数
          - total_defects: 检出缺陷总数
          - average_confidence: 平均置信度
          - defect_distribution: 缺陷类型分布
    """
    stats = get_statistics(db)
    return JSONResponse(stats)
