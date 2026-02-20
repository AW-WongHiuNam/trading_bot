from pydantic import BaseModel
from typing import Any


class AgentConfigIn(BaseModel):
    enabled_agents: list[str]
    model: str
    params: dict[str, Any] = {}


class AgentConfigOut(BaseModel):
    enabled_agents: list[str]
    model: str
    params: dict[str, Any]


class PromptConfigIn(BaseModel):
    agent_name: str
    prompt_text: str | None = None


class PromptConfigOut(BaseModel):
    agent_name: str
    prompt_text: str
