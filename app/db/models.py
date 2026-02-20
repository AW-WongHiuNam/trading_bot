from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from app.db.session import Base


class AgentConfig(Base):
    __tablename__ = "agent_config"
    id = Column(Integer, primary_key=True, index=True)
    enabled_agents = Column(Text, nullable=False)
    model = Column(String(128), nullable=False)
    params_json = Column(Text, nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class PromptConfig(Base):
    __tablename__ = "prompt_config"
    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(64), unique=True, nullable=False)
    prompt_text = Column(Text, nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(16), nullable=False)
    target_date = Column(String(16), nullable=False)
    status = Column(String(32), nullable=False, default="queued")
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class JobState(Base):
    __tablename__ = "job_states"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, nullable=False, index=True)
    stage = Column(String(64), nullable=False)
    payload_json = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
