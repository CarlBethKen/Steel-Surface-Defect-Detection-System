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

    # 加载 Faster R-CNN（mmdetection 3.x）
    if fasterrcnn_path is None:
        fasterrcnn_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_coco_bbox_mAP_epoch_10.pth')

    try:
        try:
            from mmdet.apis import init_detector
            import torch as _torch
            from mmengine.config import Config as MmConfig
        except Exception:
            raise

        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'faster_rcnn', 'faster-rcnn_r50_fpn_1x_coco.py')
        if os.path.exists(fasterrcnn_path) and os.path.exists(config_path):
            try:
                # 从 checkpoint 中自动检测 num_classes 并修正 config
                cfg = MmConfig.fromfile(config_path)
                ckpt = _torch.load(fasterrcnn_path, map_location='cpu')
                # mmdet checkpoint 通常在 state_dict 或直接是 state_dict
                state_dict = ckpt.get('state_dict', ckpt)
                # 通过 fc_cls 权重的 shape 推断类别数（num_classes + 1 for background in some versions）
                fc_cls_key = None
                for k in state_dict:
                    if 'bbox_head.fc_cls' in k and 'weight' in k:
                        fc_cls_key = k
                        break
                if fc_cls_key is not None:
                    num_classes_from_ckpt = state_dict[fc_cls_key].shape[0]
                    # mmdet 3.x: fc_cls output = num_classes + 1 (background)
                    # 但如果用 sigmoid，则 output = num_classes
                    # Faster R-CNN 默认用 softmax，所以 output = num_classes + 1
                    inferred_num_classes = num_classes_from_ckpt - 1
                    if inferred_num_classes < 1:
                        inferred_num_classes = num_classes_from_ckpt
                    current_num_classes = cfg.model.roi_head.bbox_head.get('num_classes', 80)
                    if inferred_num_classes != current_num_classes:
                        print(f"Config num_classes={current_num_classes}, checkpoint num_classes={inferred_num_classes}, 自动修正。")
                        cfg.model.roi_head.bbox_head.num_classes = inferred_num_classes

                models['fasterrcnn'] = init_detector(cfg, fasterrcnn_path, device='cpu')
                print(f"Loaded Faster R-CNN model from: {fasterrcnn_path}")
            except Exception as e:
                print(f"Failed to initialize Faster R-CNN from {fasterrcnn_path}: {e}")
                import traceback
                traceback.print_exc()
                models['fasterrcnn'] = None
        else:
            if not os.path.exists(fasterrcnn_path):
                print(f"Faster R-CNN checkpoint not found at {fasterrcnn_path}")
            if not os.path.exists(config_path):
                print(f"Faster R-CNN config not found at {config_path}")
            models['fasterrcnn'] = None
    except Exception:
        print("mmdetection (Faster R-CNN) 未安装或加载失败，跳过 Faster R-CNN 模型加载。")
        print("请安装: pip install openmim && mim install mmengine mmcv mmdet")
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
            # mmdetection 3.x 推理
            try:
                from mmdet.apis import inference_detector
            except ImportError:
                print("mmdet.apis.inference_detector 不可用")
                return []

            result = inference_detector(model, image_input)
            detection_list = []

            # mmdet 3.x 返回 DetDataSample 对象
            try:
                pred_instances = result.pred_instances
                bboxes = pred_instances.bboxes.cpu().numpy()
                scores = pred_instances.scores.cpu().numpy()
                labels = pred_instances.labels.cpu().numpy()
                for i in range(len(scores)):
                    score = float(scores[i])
                    if score < 0.01:
                        continue
                    bbox = [float(bboxes[i][j]) for j in range(4)]
                    label = f"class_{int(labels[i])}"
                    detection_list.append({"label": label, "score": score, "bbox": bbox})
                return detection_list
            except (AttributeError, TypeError):
                pass

            # 兼容 mmdet 2.x: result 是 list of ndarray
            try:
                for cls_id, cls_bboxes in enumerate(result):
                    for row in cls_bboxes:
                        score = float(row[4])
                        if score < 0.01:
                            continue
                        bbox = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
                        label = f"class_{cls_id}"
                        detection_list.append({"label": label, "score": score, "bbox": bbox})
            except Exception:
                pass
            return detection_list

        else:
            print(f"Unsupported model_type: {model_type}")
            return []

    except Exception as e:
        print(f"Inference error for model {model_type}: {e}")
        return []
