"""
记录存储模块 - 保存检测记录到 CSV 文件
"""
import csv
import json
import os
from datetime import datetime


def save_record_csv(
    csv_path: str,
    image_name: str,
    model_type: str,
    detections: list,
    result_image: str,
):
    """
    保存检测记录到 CSV 文件

    Args:
        csv_path: CSV 文件路径
        image_name: 原始图像文件名
        model_type: 模型类型 (A 或 B)
        detections: 检测结果列表
        result_image: 结果图像相对路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 检查文件是否存在（判断是否需要写入表头）
    file_exists = os.path.exists(csv_path)

    # 计算统计信息
    defect_count = len(detections)
    confidence_avg = (
        sum(d.get("score", 0) for d in detections) / len(detections)
        if detections
        else 0.0
    )

    # 准备行数据
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": image_name,
        "model_type": model_type,
        "defect_count": defect_count,
        "confidence_avg": round(confidence_avg, 3),
        "detections_json": json.dumps(detections, ensure_ascii=False),
        "result_image": result_image,
    }

    # 写入 CSV
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "image_name",
                    "model_type",
                    "defect_count",
                    "confidence_avg",
                    "detections_json",
                    "result_image",
                ]
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)
    except Exception as e:
        print(f"保存 CSV 记录失败: {e}")

