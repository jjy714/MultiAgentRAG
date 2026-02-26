from langgraph.graph import START, END, StateGraph
from graph.state import DiscussionState
from agents.DirectResponderAdvocate import DirectResponderAdvocate
from agents.ContextualAnalystAdvocate import ContextualAnalystAdvocate
from agents.DeepResearcherAdvocate import DeepResearcherAdvocate
from agents.ModeratorAgent import ModeratorAgent


def create_discussion_graph():
    """
    Three advocate agents run in parallel fan-out from START.
    All three fan back into a Moderator that applies strict majority vote.
    Outputs DiscussionState with 'complexity' field set.
    """
    graph = StateGraph(DiscussionState)

    graph.add_node("direct_responder_advocate",   DirectResponderAdvocate)
    graph.add_node("contextual_analyst_advocate", ContextualAnalystAdvocate)
    graph.add_node("deep_researcher_advocate",    DeepResearcherAdvocate)
    graph.add_node("moderator",                   ModeratorAgent)

    # Parallel fan-out: START → all three advocates simultaneously
    graph.add_edge(START, "direct_responder_advocate")
    graph.add_edge(START, "contextual_analyst_advocate")
    graph.add_edge(START, "deep_researcher_advocate")

    # Fan-in: all three → moderator
    graph.add_edge("direct_responder_advocate",   "moderator")
    graph.add_edge("contextual_analyst_advocate", "moderator")
    graph.add_edge("deep_researcher_advocate",    "moderator")

    graph.add_edge("moderator", END)

    return graph.compile()
