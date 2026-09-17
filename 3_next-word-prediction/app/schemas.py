from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)

    num_words: int = Field(default=1, ge=1, le=20)


class PredictionResponse(BaseModel):
    input_text: str
    generated_text: str
