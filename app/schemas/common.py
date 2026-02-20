from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    details: str | None = None
