"""
推理模块 - YOLOv8 模型加载和缺陷检测
"""
import os
from typing import Dict, Any, List


def load_models(yolo_path: str = None) -> Dict[str, Any]:
    """
    加载 YOLOv8 模型，返回 {"yolov8m": model_or_None}
    """
    models = {"yolov8m": None}

    if yolo_path is None:
        yolo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best.pt')

    try:
        from ultralytics import YOLO

        if os.path.exists(yolo_path):
            try:
                models['yolov8m'] = YOLO(yolo_path)
                print(f"Loaded YOLOv8 model from: {yolo_path}")
            except Exception as e:
                print(f"Failed to initialize YOLO model from {yolo_path}: {e}")
        else:
            print(f"YOLO weight not found at {yolo_path}")
    except ImportError:
        print("ultralytics 未安装，请运行: pip install ultralytics")

    return models


def run_infer(models: Dict[str, Any], image_input: Any, model_type: str = "yolov8m") -> List[Dict[str, Any]]:
    """
    执行 YOLOv8 推理，返回检测结果列表
    """
    model = models.get('yolov8m')
    if model is None:
        print("YOLOv8 model is not loaded. Skipping inference.")
        return []

    try:
        results = model(image_input)
        detection_list = []

        for r in results:
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                continue
            for box in boxes:
                cls = getattr(box, 'cls', None)
                conf = getattr(box, 'conf', getattr(box, 'confidence', None))
                xyxy = getattr(box, 'xyxy', None)
                if xyxy is None:
                    continue
                try:
                    coords = [float(x) for x in xyxy[0].tolist()]
                except Exception:
                    try:
                        coords = [float(x) for x in xyxy.tolist()]
                    except Exception:
                        coords = [0, 0, 0, 0]
                label = f"class_{int(cls)}" if cls is not None else "object"
                detection_list.append({
                    "label": label,
                    "score": float(conf) if conf is not None else 0.0,
                    "bbox": coords,
                })
        return detection_list

    except Exception as e:
        print(f"Inference error: {e}")
        return []
