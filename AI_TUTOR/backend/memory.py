"""Milestone 8: persistent session memory via PostgreSQL + SQLAlchemy.

Conversations survive server restarts. Gracefully degrades to in-memory
if PostgreSQL is unavailable.
"""
import os
from datetime import datetime

from sqlalchemy import Column, String, Text, Float, Integer, DateTime
from sqlalchemy.orm import declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://tutor:tutor@localhost:5432/ai_tutor")

Base = declarative_base()

_engine = None
_SessionLocal = None
_db_available = False


class Session(Base):
    __tablename__ = "sessions"
    id = Column(String(64), primary_key=True)
    student_level = Column(String(32), default="beginner")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True)
    role = Column(String(16))
    content = Column(Text)
    grounded_fraction = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class UploadedDoc(Base):
    __tablename__ = "uploaded_docs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=True)
    filename = Column(String(256))
    chunks = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    global _engine, _SessionLocal, _db_available
    if _engine is not None:
        return
    try:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        _engine = create_engine(
            DATABASE_URL, pool_pre_ping=False, future=True,
            connect_args={"connect_timeout": 3},
        )
        with _engine.connect() as conn:
            pass
        _SessionLocal = sessionmaker(bind=_engine)
        Base.metadata.create_all(_engine)
        _db_available = True
        print(f"[memory] PostgreSQL connected")
    except Exception as e:
        _db_available = False
        _engine = None
        print(f"[memory] PostgreSQL unavailable ({e.__class__.__name__}) — in-memory only")


def db_session():
    if not _db_available or _SessionLocal is None:
        return None
    return _SessionLocal()


def save_message(session_id: str, role: str, content: str, grounded_fraction: float | None = None):
    s = db_session()
    if not s:
        return
    try:
        msg = Message(session_id=session_id, role=role, content=content, grounded_fraction=grounded_fraction)
        s.add(msg)
        s.commit()
    except Exception:
        s.rollback()
    finally:
        s.close()


def get_history(session_id: str, limit: int = 50) -> list[dict]:
    s = db_session()
    if not s:
        return []
    try:
        rows = (
            s.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
            .all()
        )
        return [{"role": r.role, "content": r.content, "ts": r.created_at.isoformat()} for r in reversed(rows)]
    finally:
        s.close()
