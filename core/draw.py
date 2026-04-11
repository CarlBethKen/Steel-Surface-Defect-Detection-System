"""
绘制检测结果 - 在图片上标注缺陷
"""
import cv2
import os
from typing import List, Dict, Any


# 定义缺陷类型和对应颜色
DEFECT_COLORS = {
    "scratch": (0, 0, 255),        # 红色 - 划痕
    "pit": (0, 255, 0),             # 绿色 - 坑洞
    "patch": (255, 0, 0),           # 蓝色 - 补丁/斑点
    "roll_mark": (0, 255, 255),     # 黄色 - 轧痕
    "crack": (255, 0, 255),         # 洋红 - 裂纹
    "oxide": (255, 165, 0),         # 橙色 - 氧化
    "dent": (128, 0, 128),          # 紫色 - 凹陷
    "corrosion": (0, 128, 128),     # 青色 - 腐蚀
    "unknown": (128, 128, 128),     # 灰色 - 未知
}


def _get_defect_color(label: str) -> tuple:
    """根据缺陷标签获取颜色，支持 class_id 格式和文字标签"""
    # 直接查找
    if label in DEFECT_COLORS:
        return DEFECT_COLORS[label]

    # 如果是 class_0, class_1 格式
    if label.startswith("class_"):
        try:
            class_id = int(label.split("_")[1])
            colors = list(DEFECT_COLORS.values())
            return colors[class_id % len(colors)]
        except:
            pass

    return DEFECT_COLORS["unknown"]


def _auto_resize_image(img, max_width=2000, max_height=2000):
    """自适应调整图片大小，确保不超过最大尺寸"""
    h, w = img.shape[:2]

    if w <= max_width and h <= max_height:
        return img, 1.0  # 返回原图和缩放比例

    # 计算缩放比例
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)

    if scale == 1.0:
        return img, 1.0

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return resized, scale


def draw_detections(input_path: str, output_path: str, detections: List[Dict[str, Any]]):
    """
    在图片上绘制检测结果（支持自适应缩放）

    Args:
        input_path: 输入图片路径
        output_path: 输出图片路径
        detections: 检测结果列表，每个元素为 dict，包含 label/score/bbox
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"无法读取图片: {input_path}")

    # 自适应调整图片大小
    img, scale = _auto_resize_image(img)

    # 绘制每个检测结果
    for det in detections:
        label = det.get("label", "unknown")
        score = det.get("score", 0)
        bbox = det.get("bbox", [])

        if len(bbox) < 4:
            continue

        # 根据缩放比例调整 bbox 坐标
        x1 = int(bbox[0] * scale)
        y1 = int(bbox[1] * scale)
        x2 = int(bbox[2] * scale)
        y2 = int(bbox[3] * scale)

        # 获取颜色
        color = _get_defect_color(label)

        # 绘制矩形框（加粗以适应缩放后的大小）
        thickness = max(2, int(2 * scale) if scale > 1 else 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 绘制文本标签
        text = f"{label}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.4, 0.5 * scale)
        text_thickness = max(1, int(1 * scale) if scale > 1 else 1)
        text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]

        # 文本背景
        cv2.rectangle(
            img,
            (x1, y1 - text_size[1] - 6),
            (x1 + text_size[0] + 4, y1 - 2),
            color,
            -1
        )

        # 绘制文本
        cv2.putText(
            img,
            text,
            (x1 + 2, y1 - 4),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness
        )

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存结果图片
    cv2.imwrite(output_path, img)

