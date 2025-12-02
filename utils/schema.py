from pydantic import BaseModel, Field


class Score(BaseModel):
    score: int = Field(..., ge=0, le=100, description="Score 0 to 100")

class Assessment(BaseModel):
    summary: str
    score: Score
    
class EvaluationResponse(BaseModel):
    novelty_assessment: Assessment
    s_and_t_assessment: Assessment
    evaluation: str
    proposal_ids: set
    
class GeneralSchema(BaseModel):
    proposal_id: str