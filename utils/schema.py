from pydantic import BaseModel, Field
from typing import Annotated,TypedDict,List,Any

class Score(BaseModel):
    score: int = Field(..., ge=0, le=10, description="Score 0 to 10")

class EvaluationScore(BaseModel):
    Budget:  int = Field(..., ge=0, le=10, description="Score 0 to 10")
    Technical_Novelty: int = Field(..., ge=0, le=10, description="Score 0 to 10")
    Technical_Feasibility: int = Field(..., ge=0, le=10, description="Score 0 to 10")
    Expertise: int = Field(..., ge=0, le=10, description="Score 0 to 10")
    Compliance_with_Guidelines: int = Field(..., ge=0, le=10, description="Score 0 to 10")
    Industry_Relevance: int = Field(..., ge=0, le=10, description="Score 0 to 10")
    Scalability: int = Field(..., ge=0, le=10, description="Score 0 to 10")
    Sustainability: int = Field(..., ge=0, le=10, description="Score 0 to 10")
    Impact: int = Field(..., ge=0, le=10, description="Score 0 to 10")
    Summary: str = Field(..., description= "Explanation behind the evaluation report")

class BudgetAnalysis(BaseModel):
    Equipment  :  int = Field(..., ge=0,  description="Amount of Money Spent on Equipment")
    Manpower : int = Field(..., ge=0,  description="Amount of Money spent on Manpower")
    Consumables : int = Field(..., ge=0, description="Amount of Money spent on Consumables")
    Travel : int = Field(..., ge=0,  description="Amount of Money spent on Travel")
    Contingency : int = Field(..., ge=0,  description="Amount of Money spent on Contingency")
    Overheads : int = Field(..., ge=0,  description="Amount of Money spent on Overheads")
    Summary : str = Field(..., description= "Explanation behind the budget report")
    Institution_Name : str = Field(..., description= "Give the name of the Institute")
    
class Assessment(BaseModel):
    summary: str
    score: Score
    
class EvaluationResponse(BaseModel):
    novelty_assessment: Assessment
    s_and_t_assessment: Assessment
    budget_assessment: BudgetAnalysis
    evaluation: EvaluationScore
    proposal_ids: set

class ProposalMetadata(BaseModel):
    proposal_id: Annotated[str, Field(..., description="Unique identifier for the research proposal")]
    title: Annotated[str, Field(..., description="Title of the research project or proposal")]
    pi_name: Annotated[str, Field(..., description="Name of the Principal Investigator leading the research")]
    institution: Annotated[str, Field(..., description="Organization or institute where the research will be conducted")]
    research_area: Annotated[str, Field(..., description="Broad domain or research field the proposal belongs to")]
    keywords: Annotated[str, Field(..., description="Comma-separated key terms representing the core topics of the research")]

class CoalRelevanceState(BaseModel):
    """State passed through the relevance checking workflow"""
    abstract: str
    messages: List[Any]
    is_relevant: bool
    explanation: str
    
    