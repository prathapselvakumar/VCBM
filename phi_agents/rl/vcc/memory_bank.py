from dataclasses import dataclass

import numpy as np

from phi_agents.rl.vcc.state_encoder import encode_state


@dataclass(frozen=True)
class CausalEvent:
    """A single causal (bookmarked) event stored in the episodic memory bank. See paper Section 3.2, Eq. 7."""

    turn_idx: int
    key: np.ndarray
    action_return: float


def build_memory_bank(
    turn_bookmarks: list[bool],
    turn_observations: list[str],
    turn_index_to_return_to_go: list[float],
    dim: int = 256,
) -> list[CausalEvent]:
    """Store one CausalEvent per bookmarked turn (Section 3.2, Eq. 7).

    Args:
        turn_bookmarks: per-turn bookmark flags.
        turn_observations: per-turn ipython execution-result text, used as the state for g_phi.
        turn_index_to_return_to_go: per-turn return-to-go G_t.
        dim: state encoder embedding dimension.

    Returns:
        List of CausalEvent, one per bookmarked turn, in turn order.
    """
    n = min(len(turn_bookmarks), len(turn_observations), len(turn_index_to_return_to_go))
    memory = []
    for turn_idx in range(n):
        if turn_bookmarks[turn_idx]:
            memory.append(
                CausalEvent(
                    turn_idx=turn_idx,
                    key=encode_state(turn_observations[turn_idx], dim=dim),
                    action_return=turn_index_to_return_to_go[turn_idx],
                )
            )
    return memory


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_and_weight(
    memory: list[CausalEvent],
    terminal_observation: str,
    top_k: int,
    temperature: float,
    dim: int = 256,
) -> dict[int, float]:
    """Retrieve the top-k causal events most similar to the terminal state and weight them.

    Section 3.3, Eq. 8-9: cosine similarity of each memory key to the terminal-state query,
    top-k retrieval, temperature-scaled softmax over the retrieved set.

    Args:
        memory: causal events from build_memory_bank.
        terminal_observation: final ipython observation text (or final assistant content),
            used as the terminal-state query z_H.
        top_k: number of events to retrieve.
        temperature: softmax temperature tau_temp.
        dim: state encoder embedding dimension, must match build_memory_bank's dim.

    Returns:
        Mapping from turn_idx to softmax retrieval weight, only for retrieved turns
        (weights sum to 1.0 over the retrieved set). Empty dict if memory is empty.
    """
    if not memory:
        return {}

    z_terminal = encode_state(terminal_observation, dim=dim)
    similarities = [(event, _cosine_similarity(z_terminal, event.key)) for event in memory]
    similarities.sort(key=lambda pair: pair[1], reverse=True)
    retrieved = similarities[: min(top_k, len(similarities))]

    scaled = np.array([sim / temperature for _, sim in retrieved])
    scaled = scaled - scaled.max()  # numerical stability
    exp_scaled = np.exp(scaled)
    softmax_weights = exp_scaled / exp_scaled.sum()

    return {
        event.turn_idx: float(weight)
        for (event, _), weight in zip(retrieved, softmax_weights, strict=True)
    }
