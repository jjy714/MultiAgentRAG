from typing import Annotated, List
from pydantic import BaseModel, Field
import operator


# Schema for structured output to use in planning

### Pydantic model representing a single section of a research report
class Section(BaseModel):
    """
    args   : {
        "name (str)": "title name for this section of the report",
        "description (str)": "brief overview of topics and concepts to cover"
    }
    return : {
        "Section": "validated Pydantic model with report section metadata"
    }
    """
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )


### Pydantic model representing the full set of report sections produced by the planner
class Sections(BaseModel):
    """
    args   : {
        "sections (List[Section])": "ordered list of planned report sections"
    }
    return : {
        "Sections": "validated Pydantic model containing all planned sections"
    }
    """
    sections: List[Section] = Field(
        description="Sections of the report.",
    )


# Augment the LLM with schema for structured output
planner = llm.with_structured_output(Sections)