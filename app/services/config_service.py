import json
from sqlalchemy.orm import Session
from app.db.models import AgentConfig, PromptConfig

DEFAULT_CONFIG = {
    "enabled_agents": ["market", "social", "news", "fundamentals", "risk", "manager", "trader"],
    "model": "deepseek-r1:latest",
    "params": {},
}


def get_agent_config(db: Session) -> dict:
    row = db.query(AgentConfig).first()
    if not row:
        reset_config(db)
        row = db.query(AgentConfig).first()
    return {
        "enabled_agents": json.loads(row.enabled_agents),
        "model": row.model,
        "params": json.loads(row.params_json),
    }


def update_agent_config(db: Session, enabled_agents: list[str], model: str, params: dict) -> dict:
    row = db.query(AgentConfig).first()
    if not row:
        row = AgentConfig(
            enabled_agents=json.dumps(enabled_agents),
            model=model,
            params_json=json.dumps(params),
        )
        db.add(row)
    else:
        row.enabled_agents = json.dumps(enabled_agents)
        row.model = model
        row.params_json = json.dumps(params)
    db.commit()
    return {
        "enabled_agents": enabled_agents,
        "model": model,
        "params": params,
    }


def reset_config(db: Session) -> dict:
    db.query(AgentConfig).delete()
    row = AgentConfig(
        enabled_agents=json.dumps(DEFAULT_CONFIG["enabled_agents"]),
        model=DEFAULT_CONFIG["model"],
        params_json=json.dumps(DEFAULT_CONFIG["params"]),
    )
    db.add(row)
    db.commit()
    return DEFAULT_CONFIG


def upsert_prompt(db: Session, agent_name: str, prompt_text: str) -> dict:
    row = db.query(PromptConfig).filter(PromptConfig.agent_name == agent_name).first()
    if not row:
        row = PromptConfig(agent_name=agent_name, prompt_text=prompt_text)
        db.add(row)
    else:
        row.prompt_text = prompt_text
    db.commit()
    return {"agent_name": agent_name, "prompt_text": prompt_text}


def get_prompt(db: Session, agent_name: str) -> dict | None:
    row = db.query(PromptConfig).filter(PromptConfig.agent_name == agent_name).first()
    if not row:
        return None
    return {"agent_name": row.agent_name, "prompt_text": row.prompt_text}
