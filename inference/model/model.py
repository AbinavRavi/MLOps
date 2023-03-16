from pydantic import BaseModel

class ModelInput(BaseModel):
    tweet: str

class ModelOutput(BaseModel):
    inference: int
    sentiment: bool
    