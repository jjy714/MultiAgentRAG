import asyncio
from graph.main_graph import create_main_graph


async def run(question: str):
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
    }
    result = await graph.ainvoke(initial_state)
    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result.get("final_answer", "No answer generated."))
    return result


if __name__ == "__main__":
    question = input("Enter your question: ")
    asyncio.run(run(question))