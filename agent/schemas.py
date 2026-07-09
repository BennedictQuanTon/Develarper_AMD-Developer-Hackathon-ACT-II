from pydantic import BaseModel, Field


class Task(BaseModel):
    task_id: str = Field(..., alias="task_id")
    prompt: str

    class Config:
        populate_by_name = True
        extra = "allow"


class Result(BaseModel):
    task_id: str
    answer: str
