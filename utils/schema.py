from pydantic import BaseModel, Field
from typing import Annotated

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
    
from pydantic import BaseModel

from pydantic import BaseModel, Field

class ProposalMetadata(BaseModel):
    proposal_id: Annotated[str, Field(..., description="Unique identifier for the research proposal")]
    title: Annotated[str, Field(..., description="Title of the research project or proposal")]
    pi_name: Annotated[str, Field(..., description="Name of the Principal Investigator leading the research")]
    institution: Annotated[str, Field(..., description="Organization or institute where the research will be conducted")]
    research_area: Annotated[str, Field(..., description="Broad domain or research field the proposal belongs to")]
    keywords: Annotated[str, Field(..., description="Comma-separated key terms representing the core topics of the research")]

    
    