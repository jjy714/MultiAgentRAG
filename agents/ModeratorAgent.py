from collections import Counter
from graph.state import DiscussionState


def ModeratorAgent(state: DiscussionState) -> dict:
    votes = state.get("votes", [])
    vote_counts = Counter(v.get("vote", "medium") for v in votes)

    print(f"\n[ModeratorAgent] Vote tally: {dict(vote_counts)}")

    majority = vote_counts.most_common(1)[0]
    winning_tier, winning_count = majority

    if winning_count == 1:
        winning_tier = "medium"
        print("[ModeratorAgent] 3-way tie — defaulting to 'medium'")

    print(f"[ModeratorAgent] Final routing decision: '{winning_tier}'")
    return {"complexity": winning_tier}
