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


def get_statistics(db: Session, date_from: str = None, date_to: str = None, source: str = None) -> dict:
    """获取检测统计信息，支持日期范围和来源筛选"""
    from sqlalchemy import func, cast, Date

    query = db.query(DetectionRecord)

    # 日期筛选
    if date_from:
        try:
            dt_from = datetime.strptime(date_from, "%Y-%m-%d")
            query = query.filter(DetectionRecord.created_at >= dt_from)
        except ValueError:
            pass
    if date_to:
        try:
            dt_to = datetime.strptime(date_to, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            query = query.filter(DetectionRecord.created_at <= dt_to)
        except ValueError:
            pass

    # 来源筛选
    if source == "realtime":
        query = query.filter(DetectionRecord.image_name.like("realtime_%"))
    elif source == "upload":
        query = query.filter(~DetectionRecord.image_name.like("realtime_%"))

    records = query.all()
    total_records = len(records)
    total_defects = sum(r.defect_count for r in records)
    avg_confidence = (
        sum(r.confidence_avg for r in records) / total_records
        if total_records else 0.0
    )
    defect_rate = (
        sum(1 for r in records if r.defect_count > 0) / total_records
        if total_records else 0.0
    )

    # 缺陷类型分布
    defect_types = {}
    for record in records:
        detections = json.loads(record.defections) if record.defections else []
        for det in detections:
            label = det.get("label", "unknown")
            defect_types[label] = defect_types.get(label, 0) + 1

    # 按日期分组统计
    daily_stats = {}
    for r in records:
        day = r.created_at.strftime("%Y-%m-%d") if r.created_at else "unknown"
        if day not in daily_stats:
            daily_stats[day] = {"count": 0, "defects": 0, "has_defect": 0}
        daily_stats[day]["count"] += 1
        daily_stats[day]["defects"] += r.defect_count
        if r.defect_count > 0:
            daily_stats[day]["has_defect"] += 1

    # 转为排序列表
    daily_list = []
    for day in sorted(daily_stats.keys(), reverse=True):
        s = daily_stats[day]
        daily_list.append({
            "date": day,
            "count": s["count"],
            "defects": s["defects"],
            "defect_rate": round(s["has_defect"] / s["count"], 3) if s["count"] else 0,
        })

    return {
        "total_records": total_records,
        "total_defects": total_defects,
        "average_confidence": round(avg_confidence, 3),
        "defect_rate": round(defect_rate, 3),
        "defect_distribution": defect_types,
        "daily_stats": daily_list,
    }

