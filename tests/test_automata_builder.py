from src.automata_builder import create_states, create_transition_list


def test_create_states():
    patterns = ["abc", "bca", "abc"]

    state_map = create_states(patterns)

    assert len(state_map) == 2
    assert "abc" in state_map
    assert "bca" in state_map


def test_create_transition_list():
    patterns = ["abc", "bca", "cab"]
    state_map = create_states(patterns)

    transitions = create_transition_list(patterns, state_map)

    assert len(transitions) == 2
    assert transitions[0]["from_pattern"] == "abc"
    assert transitions[0]["to_pattern"] == "bca"