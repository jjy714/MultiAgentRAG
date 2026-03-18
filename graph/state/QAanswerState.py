from typing import TypedDict


### Structured output schema for the QuestionAnsweringAgent's response
class QAAnswerState(TypedDict):
    """
    args   : {
        "analysis (str)": "reasoning or evidence analysis",
        "answer (str)": "the final answer text",
        "success (str)": "indication of whether the answer is satisfactory",
        "rating (int)": "quality rating of the answer"
    }
    return : {
        "QAAnswerState": "typed dict with the QA agent's full structured response"
    }
    """
    analysis: str
    answer: str
    success: str
    rating: int
