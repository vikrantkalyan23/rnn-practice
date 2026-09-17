from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    num_words: int = Field(default=1, ge=1, le=20)
    top_k: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=1.0, gt=0, le=3.0)
    sample: bool = False


class PredictionResponse(BaseModel):
    input_text: str
    generated_text: str
    predictions: list[dict[str, float | str]]
