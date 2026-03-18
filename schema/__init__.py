
# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

from schema.ChatRequest import ChatRequest
from schema.Message import Message



# ---------------------------------------------------------------------------
# Agent name → display label mapping (for SSE events)
# ---------------------------------------------------------------------------

# Maps LangGraph node names to human-readable agent IDs used by the frontend


NODE_TO_AGENT = {
    # Discussion panel
    "discussion_panel":             "DiscussionPanel",
    "direct_responder_advocate":    "DirectResponderAdvocate",
    "contextual_analyst_advocate":  "ContextualAnalystAdvocate",
    "deep_researcher_advocate":     "DeepResearcherAdvocate",
    "moderator":                    "ModeratorAgent",
    # Easy tier
    "direct_responder":             "DirectResponderAgent",
    "web_search":                   "WebSearchAgent",
    "easy_tier":                    "EasyTier",
    # Medium tier
    "retrieve":                     "RetrieverAgent",
    "extract":                      "ExtractorAgent",
    "answer":                       "QuestionAnsweringAgent",
    "medium_tier":                  "MediumTier",
    # Complex tier
    "planner":                      "PlannerAgent",
    "executor":                     "StepDefinerAgent",
    "complex_tier":                 "ComplexTier",
    "rag_execute":                  "RetrieverAgent",
    "web_execute":                  "WebSearchAgent",
}