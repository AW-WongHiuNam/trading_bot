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


class BacktestingJob(Base):
    __tablename__ = "backtesting_jobs"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(16), nullable=False, index=True)
    start_date = Column(String(16), nullable=False)
    end_date = Column(String(16), nullable=False)
    decision_policy = Column(String(16), nullable=False, default="auto")
    status = Column(String(32), nullable=False, default="queued")
    progress = Column(Integer, nullable=False, default=0)
    report_path = Column(Text, nullable=True)
    summary_json = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class BacktestingState(Base):
    __tablename__ = "backtesting_states"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, nullable=False, index=True)
    stage = Column(String(64), nullable=False)
    payload_json = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class TradeHistory(Base):
    __tablename__ = "trade_history"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(16), nullable=False, index=True)
    trade_date = Column(String(16), nullable=False, index=True)
    side = Column(String(16), nullable=False)
    size = Column(String(32), nullable=False)
    entry_date = Column(String(16), nullable=True)
    exit_date = Column(String(16), nullable=True)
    entry_price = Column(String(32), nullable=True)
    exit_price = Column(String(32), nullable=True)
    net_return = Column(String(32), nullable=True)
    pnl = Column(String(32), nullable=True)
    source_type = Column(String(32), nullable=False, default="backtesting")
    source_job_id = Column(Integer, nullable=True, index=True)
    created_at = Column(DateTime, server_default=func.now())
