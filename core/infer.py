"""
推理模块 - 模型加载和缺陷检测（防御性实现）

目标：
- 避免在模块导入时强制依赖 ultralytics/mmcv/mmdet 等，防止启动期间抛出 ModuleNotFoundError
- 将 heavy 依赖的导入和模型初始化放在 load_models 内部并使用 try/except
- run_infer 在模型缺失时优雅返回空列表
"""
import os
from typing import Dict, Any, List, Tuple
import math


def load_models(yolo_path: str = None, fasterrcnn_path: str = None) -> Dict[str, Any]:
    """
    尝试加载可用的模型并返回字典：{"yolov8m": model_or_None, "fasterrcnn": model_or_None}
    导入 heavy 依赖时使用 try/except，若缺失则打印警告并返回 None 对应模型。
    """
    models = {"yolov8m": None, "fasterrcnn": None}

    # 加载 YOLOv8（ultralytics）
    if yolo_path is None:
        # 默认模型路径（相对于项目根）
        yolo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best.pt')

    try:
        try:
            from ultralytics import YOLO
        except Exception:
            # ultralytics 不可用
            raise

        if os.path.exists(yolo_path):
            try:
                models['yolov8m'] = YOLO(yolo_path)
                print(f"Loaded YOLOv8 model from: {yolo_path}")
            except Exception as e:
                print(f"Failed to initialize YOLO model from {yolo_path}: {e}")
                models['yolov8m'] = None
        else:
            print(f"YOLO weight not found at {yolo_path}")
    except Exception:
        # 不输出 full traceback，给出提示
        print("ultralytics (YOLOv8) 未安装或加载失败，跳过 YOLO 模型加载。")
        models['yolov8m'] = None

    # 加载 Faster R-CNN（mmdetection）
    if fasterrcnn_path is None:
        fasterrcnn_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_coco_bbox_mAP_epoch_10.pth')

    try:
        try:
            # import mmdetection 相关
            from mmdet.apis import init_detector
        except Exception:
            raise

        # config 文件路径（如果项目中没有，可以跳过）
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'faster_rcnn', 'faster-rcnn_r50_fpn_1x_coco.py')
        if os.path.exists(fasterrcnn_path) and os.path.exists(config_path):
            try:
                models['fasterrcnn'] = init_detector(config_path, fasterrcnn_path, device='cpu')
                print(f"Loaded Faster R-CNN model from: {fasterrcnn_path}")
            except Exception as e:
                print(f"Failed to initialize Faster R-CNN from {fasterrcnn_path}: {e}")
                models['fasterrcnn'] = None
        else:
            if not os.path.exists(fasterrcnn_path):
                print(f"Faster R-CNN checkpoint not found at {fasterrcnn_path}")
            if not os.path.exists(config_path):
                print(f"Faster R-CNN config not found at {config_path}")
            models['fasterrcnn'] = None
    except Exception:
        print("mmdetection (Faster R-CNN) 未安装或加载失败，跳过 Faster R-CNN 模型加载。")
        models['fasterrcnn'] = None

    return models


def _iou(box1: List[float], box2: List[float]) -> float:
    """计算两个 bbox 的 IoU，bbox 格式 [x1,y1,x2,y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def run_infer(models: Dict[str, Any], image_input: Any, model_type: str = "yolov8m") -> List[Dict[str, Any]]:
    """
    执行推理：支持 'yolov8m'、'fasterrcnn'、'both'
    """

    # 允许 both
    if model_type not in models and model_type != 'both':
        print(f"Unknown model_type: {model_type}")
        return []

    try:
        if model_type == 'both':
            # 分别调用两种模型（若加载）并合并
            results_all: List[Dict[str, Any]] = []
            # YOLO
            yolo_model = models.get('yolov8m')
            if yolo_model is not None:
                yolo_res = run_infer(models, image_input, model_type='yolov8m')
                results_all.extend(yolo_res)
            # Faster R-CNN
            fr_model = models.get('fasterrcnn')
            if fr_model is not None:
                fr_res = run_infer(models, image_input, model_type='fasterrcnn')
                results_all.extend(fr_res)

            # 如果只有一个模型有结果，直接返回
            if not results_all:
                return []

            # 合并：基于 IoU 去重，IoU>0.5 视为相同检测，保留置信度更高的
            merged: List[Dict[str, Any]] = []
            used = [False] * len(results_all)
            for i, det in enumerate(results_all):
                if used[i]:
                    continue
                best = det
                used[i] = True
                for j in range(i + 1, len(results_all)):
                    if used[j]:
                        continue
                    iou_val = _iou(best['bbox'], results_all[j]['bbox'])
                    if iou_val > 0.5:
                        # 保留置信度更高的
                        if results_all[j]['score'] > best['score']:
                            best = results_all[j]
                        used[j] = True
                merged.append(best)

            return merged

        # 原有单模型逻辑
        if model_type == 'yolov8m':
            # ultralytics YOLOv8: model accepts path or numpy array
            model = models.get('yolov8m')
            if model is None:
                print("Requested model 'yolov8m' is not loaded. Skipping inference.")
                return []
            results = model(image_input)
            detection_list = []
            # results may be iterable; extract boxes
            for r in results:
                # r.boxes 可以为空
                boxes = getattr(r, 'boxes', None)
                if boxes is None:
                    continue
                for box in boxes:
                    # box.cls, box.conf, box.xyxy
                    cls = getattr(box, 'cls', None)
                    conf = getattr(box, 'conf', getattr(box, 'confidence', None))
                    xyxy = getattr(box, 'xyxy', None)
                    if xyxy is None:
                        continue
                    # xyxy could be tensor; convert to list
                    try:
                        coords = [float(x) for x in xyxy[0].tolist()]
                    except Exception:
                        # fallback if structure differs
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

        elif model_type == 'fasterrcnn':
            model = models.get('fasterrcnn')
            if model is None:
                print("Requested model 'fasterrcnn' is not loaded. Skipping inference.")
                return []
            # mmdetection 推理：model.inference_detector
            result = model.inference_detector(image_input)
            # result 解析依赖于类别数量和输出格式，这里给出通用的占位解析
            detection_list = []
            # 如果 result 是 list of ndarray，每个数组表示对应类别的 bbox
            try:
                for cls_id, cls_bboxes in enumerate(result):
                    # cls_bboxes: numpy array [N,5] -> x1,y1,x2,y2,score
                    for row in cls_bboxes:
                        score = float(row[4])
                        if score < 0.01:
                            continue
                        bbox = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
                        label = f"class_{cls_id}"
                        detection_list.append({"label": label, "score": score, "bbox": bbox})
            except Exception:
                # 其他格式的 result，直接忽略并返回空
                pass
            return detection_list

        else:
            print(f"Unsupported model_type: {model_type}")
            return []

    except Exception as e:
        print(f"Inference error for model {model_type}: {e}")
        return []
