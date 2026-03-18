from collections import Counter
from graph.state import DiscussionState


## LangGraph node function that tallies advocate votes and selects the majority routing tier
def ModeratorAgent(state: DiscussionState) -> dict:
    """
    args   : {
        "state (DiscussionState)": "discussion state containing 'votes' (List[AdvocateVote])"
    }
    return : {
        "dict": "updated state with 'complexity' (str) set to the winning tier"
    }
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
