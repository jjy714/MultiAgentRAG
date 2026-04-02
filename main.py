import asyncio
import asyncio
from graph.main_graph import create_main_graph


## Async entry point that runs the MultiAgentRAG pipeline for a given question
async def run(question: str):
    """
    args   : {
        "question (str)": "the user's question to be processed by the graph"
    }
    return : {
        "dict": "final graph state containing 'final_answer' and all intermediate fields"
    }
    """
    graph = create_main_graph()
    initial_state = {
        "original_question": question,
        "complexity": "",
        "final_answer": None,
        "documents": [],
        "doc_ids": [],
        "notes": [],
        "plan": [],
        "step_question": [],
        "step_output": [],
        "step_notes": [],
        "is_report": False,
        "plan_summary": None,
        "web_needed": False,
        "web_result": None,
        "token_usage": {},
    }
    result = await graph.ainvoke(initial_state)
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result.get("final_answer", "No answer generated."))

    usage = result.get("token_usage", {})
    print("\n" + "=" * 60)
    print("TOKEN USAGE")
    print("=" * 60)
    print(f"Total tokens:      {usage.get('total_tokens', 0)}")
    print(f"Prompt tokens:     {usage.get('prompt_tokens', 0)}")
    print(f"Completion tokens: {usage.get('completion_tokens', 0)}")
    return result

# What is the difference between supervised and unsupervised learning?
# What is an agent?
if __name__ == "__main__":
    question = input("Enter your question: ")
    asyncio.run(run(question))