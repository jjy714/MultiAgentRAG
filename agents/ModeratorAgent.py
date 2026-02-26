from collections import Counter
from graph.state import DiscussionState


def ModeratorAgent(state: DiscussionState) -> dict:
    """
    Strict majority-vote moderator.
    Counts votes from all three advocates and picks the majority.
    Tie-break (1-1-1): defaults to 'medium'.
    No LLM call — pure deterministic vote counter.
    """
    votes = state.get("votes", [])
    vote_counts = Counter(v.get("vote", "medium") for v in votes)

    print(f"\n[ModeratorAgent] Vote tally: {dict(vote_counts)}")

    majority = vote_counts.most_common(1)[0]
    winning_tier, winning_count = majority

    # If it's a genuine 3-way tie, fall back to medium
    if winning_count == 1:
        winning_tier = "medium"
        print("[ModeratorAgent] 3-way tie — defaulting to 'medium'")

    print(f"[ModeratorAgent] Final routing decision: '{winning_tier}'")
    return {"complexity": winning_tier}
