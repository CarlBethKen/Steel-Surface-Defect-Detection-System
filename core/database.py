"""
数据库模块 - 用于保存和查询检测记录
"""
import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# 数据库配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "detection_records.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class DetectionRecord(Base):
    """检测记录数据库模型"""
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(String, unique=True, index=True)
    image_name = Column(String, index=True)
    image_path = Column(String)  # 原始图片路径
    result_image_path = Column(String)  # 标注图片路径
    model_type = Column(String, index=True)  # 模型类型: A 或 B
    defect_count = Column(Integer, default=0)  # 缺陷数量
    defections = Column(Text)  # JSON 格式的检测结果
    confidence_avg = Column(Float, default=0.0)  # 平均置信度
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "file_id": self.file_id,
            "image_name": self.image_name,
            "image_path": self.image_path,
            "result_image_path": self.result_image_path,
            "model_type": self.model_type,
            "defect_count": self.defect_count,
            "defections": json.loads(self.defections) if self.defections else [],
            "confidence_avg": round(self.confidence_avg, 3),
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else "",
        }


# 创建数据库表
Base.metadata.create_all(bind=engine)


def get_db():
    """获取数据库连接"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_record(
    db: Session,
    file_id: str,
    image_name: str,
    image_path: str,
    result_image_path: str,
    model_type: str,
    detections: list,
) -> DetectionRecord:
    """保存检测记录到数据库"""
    defect_count = len(detections)
    confidence_avg = (
        sum(d.get("score", 0) for d in detections) / len(detections)
        if detections
        else 0.0
    )

    record = DetectionRecord(
        file_id=file_id,
        image_name=image_name,
        image_path=image_path,
        result_image_path=result_image_path,
        model_type=model_type,
        defect_count=defect_count,
        defections=json.dumps(detections),
        confidence_avg=confidence_avg,
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_all_records(
    db: Session,
    model_type: str = None,
    skip: int = 0,
    limit: int = 100,
) -> list:
    """获取所有检测记录"""
    query = db.query(DetectionRecord).order_by(DetectionRecord.created_at.desc())

    if model_type:
        query = query.filter(DetectionRecord.model_type == model_type)

    total = query.count()
    records = query.offset(skip).limit(limit).all()

    return {
        "total": total,
        "records": [r.to_dict() for r in records],
    }


def get_record_by_id(db: Session, record_id: int) -> DetectionRecord:
    """根据 ID 获取检测记录"""
    return db.query(DetectionRecord).filter(DetectionRecord.id == record_id).first()


def get_record_by_file_id(db: Session, file_id: str) -> DetectionRecord:
    """根据 file_id 获取检测记录"""
    return db.query(DetectionRecord).filter(DetectionRecord.file_id == file_id).first()


def delete_record(db: Session, record_id: int) -> bool:
    """删除检测记录"""
    record = db.query(DetectionRecord).filter(DetectionRecord.id == record_id).first()
    if record:
        db.delete(record)
        db.commit()
        return True
    return False


def get_statistics(db: Session) -> dict:
    """获取检测统计信息"""
    total_records = db.query(DetectionRecord).count()

    # 按模型统计
    model_a_count = db.query(DetectionRecord).filter(
        DetectionRecord.model_type == "A"
    ).count()
    model_b_count = db.query(DetectionRecord).filter(
        DetectionRecord.model_type == "B"
    ).count()

    # 缺陷总数
    total_defects = db.query(DetectionRecord).with_entities(
        DetectionRecord.defect_count
    ).all()
    total_defects_count = sum(d[0] for d in total_defects)

    # 平均置信度
    avg_confidence = db.query(DetectionRecord).with_entities(
        DetectionRecord.confidence_avg
    ).all()
    avg_confidence_value = (
        sum(d[0] for d in avg_confidence) / len(avg_confidence)
        if avg_confidence
        else 0.0
    )

    # 各类型缺陷分布
    all_records = db.query(DetectionRecord).all()
    defect_types = {}
    for record in all_records:
        detections = json.loads(record.defections) if record.defections else []
        for det in detections:
            label = det.get("label", "unknown")
            defect_types[label] = defect_types.get(label, 0) + 1

    return {
        "total_records": total_records,
        "model_a_count": model_a_count,
        "model_b_count": model_b_count,
        "total_defects": total_defects_count,
        "average_confidence": round(avg_confidence_value, 3),
        "defect_distribution": defect_types,
    }

