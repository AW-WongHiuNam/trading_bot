from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.schemas.config import AgentConfigIn, AgentConfigOut, PromptConfigIn, PromptConfigOut
from app.services.config_service import get_agent_config, update_agent_config, reset_config, upsert_prompt, get_prompt

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("/config", response_model=AgentConfigOut)
def api_get_config(db: Session = Depends(get_db)):
    return get_agent_config(db)


@router.put("/config", response_model=AgentConfigOut)
def api_put_config(payload: AgentConfigIn, db: Session = Depends(get_db)):
    return update_agent_config(db, payload.enabled_agents, payload.model, payload.params)


@router.post("/config/reset", response_model=AgentConfigOut)
def api_reset_config(db: Session = Depends(get_db)):
    return reset_config(db)


@router.post("/config/prompts", response_model=PromptConfigOut)
def api_prompts(payload: PromptConfigIn, db: Session = Depends(get_db)):
    if payload.prompt_text is None:
        result = get_prompt(db, payload.agent_name)
        if not result:
            return {"agent_name": payload.agent_name, "prompt_text": ""}
        return result
    return upsert_prompt(db, payload.agent_name, payload.prompt_text)


@router.put("/config/prompts", response_model=PromptConfigOut)
def api_prompts_put(payload: PromptConfigIn, db: Session = Depends(get_db)):
    return upsert_prompt(db, payload.agent_name, payload.prompt_text or "")
