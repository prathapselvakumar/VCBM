import torch

def compute_credit_weights(
    turn_bookmarks: list[bool],
    turn_token_spans: list[tuple[int, int]],
    alpha: float = 0.1,
    device: torch.device | None = None
) -> torch.Tensor:
    """
    Map per-turn bookmark flags to per-token credit weights.
    
    Args:
        turn_bookmarks: list of bool, True if turn t is a visual bookmark
        turn_token_spans: list of (start, end) token indices for each turn
        alpha: background credit floor for non-bookmarked turns
        device: target device for the output tensor
    
    Returns:
        weights: tensor of shape [n_output_tokens]
    """
    if not turn_token_spans:
        return torch.ones(0, device=device)
        
    total_tokens = max(end for _, end in turn_token_spans)
    weights = torch.full((total_tokens,), alpha, device=device)
    
    for turn_idx, is_bookmark in enumerate(turn_bookmarks):
        if turn_idx < len(turn_token_spans):
            start, end = turn_token_spans[turn_idx]
            if is_bookmark:
                weights[start:end] = 1.0
                
    # Normalise so weights sum to total_tokens (preserve gradient scale)
    total_sum = weights.sum()
    if total_sum > 0:
        weights = weights * (total_tokens / total_sum)

    return weights


def compute_retrieval_credit_weights(
    turn_weights: dict[int, float],
    turn_token_spans: list[tuple[int, int]],
    alpha: float = 0.1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Map per-turn retrieval weights (Eq. 9) to per-token credit weights.

    Turns retrieved by retrieve_and_weight get their softmax weight (rescaled below);
    all other turns (non-bookmarked, or bookmarked but not in the top-k) get the
    background floor weight alpha, mirroring compute_credit_weights' normalization.

    Args:
        turn_weights: {turn_idx: softmax_weight} from retrieve_and_weight, weights sum to 1
            over the retrieved set.
        turn_token_spans: list of (start, end) token indices for each turn.
        alpha: background credit floor for non-retrieved turns.
        device: target device for the output tensor.

    Returns:
        weights: tensor of shape [n_output_tokens].
    """
    if not turn_token_spans:
        return torch.ones(0, device=device)

    total_tokens = max(end for _, end in turn_token_spans)
    weights = torch.full((total_tokens,), alpha, device=device)

    for turn_idx, retrieval_weight in turn_weights.items():
        if turn_idx < len(turn_token_spans):
            start, end = turn_token_spans[turn_idx]
            weights[start:end] = retrieval_weight

    # Normalise so weights sum to total_tokens (preserve gradient scale)
    total_sum = weights.sum()
    if total_sum > 0:
        weights = weights * (total_tokens / total_sum)

    return weights
