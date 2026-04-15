"""
绘制检测结果 - 在图片上标注缺陷
"""
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any


# 定义缺陷类型和对应颜色
DEFECT_COLORS = {
    "龟裂": (0, 0, 255),           # 红色
    "夹杂": (0, 255, 0),           # 绿色
    "斑块": (255, 0, 0),           # 蓝色
    "麻面": (0, 255, 255),         # 黄色
    "氧化铁皮": (255, 0, 255),     # 洋红
    "划痕": (255, 165, 0),         # 橙色
    "unknown": (128, 128, 128),     # 灰色
}


def _get_defect_color(label: str) -> tuple:
    """根据缺陷标签获取颜色"""
    if label in DEFECT_COLORS:
        return DEFECT_COLORS[label]
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


def _get_chinese_font(font_size: int):
    """
    获取中文字体，尝试多个常见字体路径
    """
    font_paths = [
        # macOS 字体路径
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        # Linux 字体路径
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # Windows 字体路径
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                continue
    
    # 如果找不到中文字体，返回默认字体
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def draw_detections(input_path: str, output_path: str, detections: List[Dict[str, Any]]):
    """
    在图片上绘制检测结果（支持中文标签）

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
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        # 尝试使用 PIL 绘制中文
        # 转换为 RGB（PIL 使用 RGB，OpenCV 使用 BGR）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # 获取中文字体
        font_size = max(16, int(20 * scale))
        font = _get_chinese_font(font_size)

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

            # 获取颜色（PIL 使用 RGB 格式）
            color_bgr = _get_defect_color(label)
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])  # BGR -> RGB

            # 绘制矩形框
            thickness = max(2, int(2 * scale) if scale > 1 else 2)
            for i in range(thickness):
                draw.rectangle(
                    [(x1 - i, y1 - i), (x2 + i, y2 + i)],
                    outline=color_rgb,
                    width=1
                )

            # 绘制文本标签
            text = f"{label}: {score:.2f}"
            
            # 获取文本边界框
            try:
                bbox_text = draw.textbbox((0, 0), text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except Exception:
                # 兼容旧版本 PIL
                try:
                    text_width, text_height = draw.textsize(text, font=font)
                except Exception:
                    text_width, text_height = 100, 20

            # 绘制文本背景
            text_bg_y1 = max(0, y1 - text_height - 8)
            text_bg_y2 = y1 - 2
            draw.rectangle(
                [(x1, text_bg_y1), (x1 + text_width + 8, text_bg_y2)],
                fill=color_rgb
            )

            # 绘制文本（白色）
            draw.text(
                (x1 + 4, text_bg_y1 + 2),
                text,
                fill=(255, 255, 255),
                font=font
            )

        # 转换回 OpenCV 格式（BGR）
        img_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"PIL 绘制失败，使用 OpenCV 回退方案: {e}")
        # 回退到 OpenCV 方案（不支持中文，但至少能显示）
        img_result = img.copy()
        
        for det in detections:
            label = det.get("label", "unknown")
            score = det.get("score", 0)
            bbox = det.get("bbox", [])

            if len(bbox) < 4:
                continue

            x1 = int(bbox[0] * scale)
            y1 = int(bbox[1] * scale)
            x2 = int(bbox[2] * scale)
            y2 = int(bbox[3] * scale)

            color = _get_defect_color(label)
            thickness = max(2, int(2 * scale) if scale > 1 else 2)
            
            # 绘制矩形框
            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制文本（可能显示为乱码，但至少有标记）
            text = f"{label}: {score:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, 0.5 * scale)
            text_thickness = max(1, int(1 * scale) if scale > 1 else 1)
            text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
            
            # 文本背景
            cv2.rectangle(
                img_result,
                (x1, y1 - text_size[1] - 6),
                (x1 + text_size[0] + 4, y1 - 2),
                color,
                -1
            )
            
            # 绘制文本
            cv2.putText(
                img_result,
                text,
                (x1 + 2, y1 - 4),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness
            )

    # 保存结果图片
    success = cv2.imwrite(output_path, img_result)
    if not success:
        raise ValueError(f"无法保存图片到: {output_path}")

