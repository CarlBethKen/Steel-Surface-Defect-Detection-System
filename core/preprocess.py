"""
图像预处理模块
"""
import cv2
import numpy as np


def preprocess_image(image_path: str, target_size: tuple = None):
    """
    预处理图像

    Args:
        image_path: 图像文件路径
        target_size: 目标尺寸 (height, width)，默认不缩放

    Returns:
        numpy array: 预处理后的图像
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 可选：转换为 RGB（OpenCV 默认是 BGR）
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 可选：缩放到指定尺寸（根据你的模型输入大小调整）
    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]))

    # 可选：图像增强（对比度调整、直方图均衡化等）
    # img = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    # 可选：去噪
    # img = cv2.bilateralFilter(img, 9, 75, 75)

    # 可选：归一化
    # img = img.astype(np.float32) / 255.0

    return img
