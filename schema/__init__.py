from schema.ChatRequest import ChatRequest
from schema.Message import Message

NODE_TO_AGENT = {
    "discussion_panel": "DiscussionPanel",
    "direct_responder_advocate": "DirectResponderAdvocate",
    "contextual_analyst_advocate": "ContextualAnalystAdvocate",
    "deep_researcher_advocate": "DeepResearcherAdvocate",
    "moderator": "ModeratorAgent",
    "direct_responder": "DirectResponderAgent",
    "web_search": "WebSearchAgent",
    "easy_tier": "EasyTier",
    "retrieve": "RetrieverAgent",
    "extract": "ExtractorAgent",
    "answer": "QuestionAnsweringAgent",
    "medium_tier": "MediumTier",
    "planner": "PlannerAgent",
    "executor": "StepDefinerAgent",
    "complex_tier": "ComplexTier",
    "rag_execute": "RetrieverAgent",
    "web_execute": "WebSearchAgent",
}
