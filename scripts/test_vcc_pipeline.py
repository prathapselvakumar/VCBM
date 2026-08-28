import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure phi_agents is in path
sys.path.append(str(Path().resolve()))

from phi_agents.appworld.interface import AppWorldInterface
from phi_agents.rl.appworld_scenario_runner import compute_turn_token_spans
from phi_agents.rl.type_defs import PolicyMessage
from phi_agents.rl.vcc.credit_weights import compute_credit_weights, compute_retrieval_credit_weights
from phi_agents.rl.vcc.memory_bank import build_memory_bank, retrieve_and_weight
from phi_agents.rl.vcc.state_encoder import encode_state


def test_credit_weights():
    print("Running test_credit_weights...")
    turn_bookmarks = [False, True, False]
    turn_token_spans = [(0, 10), (10, 30), (30, 40)]

    weights = compute_credit_weights(turn_bookmarks, turn_token_spans, alpha=0.1)

    assert weights.shape[0] == 40
    assert torch.allclose(weights[10:30], torch.tensor(1.0 * 40 / 22))
    assert torch.allclose(weights[0:10], torch.tensor(0.1 * 40 / 22))
    assert torch.allclose(weights[30:40], torch.tensor(0.1 * 40 / 22))
    print("✓ compute_credit_weights test passed!")


def test_turn_token_spans():
    print("Running test_turn_token_spans...")
    messages = [
        PolicyMessage(
            "hello",
            prompt_tokens=[0],
            stopped_by_max_tokens_limit=False,
            generated_tokens=[1, 2, 3],
            generated_token_logprobs=[-0.1, -0.2, -0.3],
        ),
        PolicyMessage(
            "world",
            prompt_tokens=[0],
            stopped_by_max_tokens_limit=False,
            generated_tokens=[4, 5],
            generated_token_logprobs=[-0.4, -0.5],
        ),
    ]
    spans = compute_turn_token_spans(messages)
    assert spans == [(0, 3), (3, 5)]
    print("✓ compute_turn_token_spans test passed!")


def test_state_encoder():
    print("Running test_state_encoder...")
    v1 = encode_state("apis.venmo.send_money(amount=10)")
    v2 = encode_state("apis.venmo.send_money(amount=10)")
    v3 = encode_state("apis.spotify.get_playlists()")

    assert np.allclose(v1, v2), "Same text should produce the same embedding"
    assert not np.allclose(v1, v3), "Different text should produce different embeddings"
    assert np.isclose(np.linalg.norm(v1), 1.0), "Embedding should be L2-normalized"

    empty = encode_state("")
    assert np.allclose(empty, 0.0), "Empty text should produce a zero vector"
    print("✓ test_state_encoder test passed!")


def test_memory_bank_retrieval():
    print("Running test_memory_bank_retrieval...")
    turn_bookmarks = [False, True, False, True, True]
    turn_observations = [
        "print output, no state change",
        "apis.venmo.send_money(receiver='alice', amount=10) -> success",
        "apis.spotify.get_playlists() -> [...]",
        "apis.venmo.send_money(receiver='bob', amount=25) -> success",
        "apis.supervisor.complete_task() -> success",
    ]
    turn_returns = [1.0] * len(turn_bookmarks)

    memory = build_memory_bank(turn_bookmarks, turn_observations, turn_returns)
    assert [event.turn_idx for event in memory] == [1, 3, 4]

    terminal_observation = "apis.supervisor.complete_task() -> success"
    turn_weights = retrieve_and_weight(memory, terminal_observation, top_k=2, temperature=1.0)

    assert len(turn_weights) == 2
    assert 4 in turn_weights, "The turn identical to the terminal observation should be retrieved"
    assert abs(sum(turn_weights.values()) - 1.0) < 1e-6, "Retrieved weights should sum to 1"
    assert turn_weights[4] == max(turn_weights.values()), (
        "The most similar turn should get the highest weight"
    )

    assert retrieve_and_weight([], terminal_observation, top_k=2, temperature=1.0) == {}
    print("✓ test_memory_bank_retrieval test passed!")


def test_compute_retrieval_credit_weights():
    print("Running test_compute_retrieval_credit_weights...")
    turn_weights = {1: 0.75, 3: 0.25}
    turn_token_spans = [(0, 10), (10, 30), (30, 40), (40, 50)]

    weights = compute_retrieval_credit_weights(turn_weights, turn_token_spans, alpha=0.1)

    assert weights.shape[0] == 50
    total_raw = 0.1 * 10 + 0.75 * 20 + 0.1 * 10 + 0.25 * 10
    scale = 50 / total_raw
    assert torch.allclose(weights[10:30], torch.tensor(0.75 * scale))
    assert torch.allclose(weights[40:50], torch.tensor(0.25 * scale))
    assert torch.allclose(weights[0:10], torch.tensor(0.1 * scale))
    assert torch.allclose(weights[30:40], torch.tensor(0.1 * scale))
    print("✓ test_compute_retrieval_credit_weights test passed!")


def test_blend_alpha_boundary():
    print("Running test_blend_alpha_boundary...")
    monte_carlo_advantage = 0.6
    turn_weights = {1: 1.0}
    turn_token_spans = [(0, 10), (10, 20)]
    bookmark_weights = compute_retrieval_credit_weights(turn_weights, turn_token_spans, alpha=0.1)

    loop_adv = torch.full_like(bookmark_weights, monte_carlo_advantage)
    vcbm_adv = torch.tensor(monte_carlo_advantage) * bookmark_weights

    # alpha_blend = 0.0 => pure LOOP: uniform advantage across all tokens
    advantage_at_0 = (1 - 0.0) * loop_adv + 0.0 * vcbm_adv
    assert torch.allclose(advantage_at_0, loop_adv)

    # alpha_blend = 1.0 => pure VCBM: advantage follows the retrieval weights exactly
    advantage_at_1 = (1 - 1.0) * loop_adv + 1.0 * vcbm_adv
    assert torch.allclose(advantage_at_1, vcbm_adv)
    print("✓ test_blend_alpha_boundary test passed!")


def test_degenerate_few_bookmarks():
    print("Running test_degenerate_few_bookmarks...")
    # Fewer bookmarked turns than top_k: the VCBM path in train.py falls back to
    # compute_credit_weights (flat floor weighting) instead of retrieval.
    turn_bookmarks = [False, True, False]
    turn_token_spans = [(0, 10), (10, 30), (30, 40)]
    top_k = 4

    memory = build_memory_bank(
        turn_bookmarks,
        ["obs0", "obs1", "obs2"],
        [1.0, 1.0, 1.0],
    )
    assert len(memory) <= top_k, "This test only covers the degenerate few-bookmarks case"

    fallback_weights = compute_credit_weights(turn_bookmarks, turn_token_spans, alpha=0.1)
    direct_weights = compute_credit_weights(turn_bookmarks, turn_token_spans, alpha=0.1)
    assert torch.allclose(fallback_weights, direct_weights)
    print("✓ test_degenerate_few_bookmarks test passed!")


def test_execute_with_bookmark():
    print("Running test_execute_with_bookmark...")
    if "APPWORLD_ROOT" not in os.environ:
        raise RuntimeError(
            "Set the APPWORLD_ROOT environment variable to your AppWorld data directory "
            "before running this test (see the main README's Installation section)."
        )
    print("Initializing AppWorldInterface...")
    world = AppWorldInterface(stdout_to_devnull=True)
    try:
        task_id = "82e2fac_1"
        world.initialize(task_id, "test_vcc", raise_on_unsafe_syntax=False)

        print("Executing read API/comment...")
        code = "print('Hello world')"
        output, is_bookmark = world.execute_with_bookmark(code)
        print(f"Read API code output: {output}, is_bookmark={is_bookmark}")
        assert is_bookmark is False, f"Expected comment to not be a bookmark but got {is_bookmark}"

        print("Executing state-changing API...")
        # complete_task is a standard write API call in AppWorld supervisor app
        code = "apis.supervisor.complete_task()"
        output, is_bookmark = world.execute_with_bookmark(code)
        print(f"Write API code output: {output}, is_bookmark={is_bookmark}")
        assert is_bookmark is True, f"Expected complete_task to be a bookmark but got {is_bookmark}"

        print("✓ execute_with_bookmark test passed!")

    finally:
        world.close_server()


if __name__ == "__main__":
    test_credit_weights()
    test_turn_token_spans()
    test_state_encoder()
    test_memory_bank_retrieval()
    test_compute_retrieval_credit_weights()
    test_blend_alpha_boundary()
    test_degenerate_few_bookmarks()
    test_execute_with_bookmark()
