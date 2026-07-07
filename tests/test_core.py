import networkx as nx
import pandas as pd
import pytest

from propagation import (
    KemenyYoung,
    lpa,
    run_label_propagation,
    set_labels,
)


def test_set_labels():
    graph = nx.path_graph(4)

    labels = set_labels(
        graph,
        [1, 3],
    )

    assert labels == {
        0: 0,
        1: 1,
        2: 0,
        3: 1,
    }


def test_unknown_experimental_node_is_rejected():
    graph = nx.path_graph(3)

    with pytest.raises(
        ValueError,
        match="absent from the graph",
    ):
        set_labels(
            graph,
            [99],
        )


def test_lpa_preserves_input_dictionary():
    graph = nx.path_graph(2)
    labels = {
        0: 1,
        1: 0,
    }

    result = lpa(
        graph,
        labels,
        iterations=1,
        fixed_nodes=[0],
    )

    assert labels == {
        0: 1,
        1: 0,
    }

    assert result == {
        0: 1,
        1: 1,
    }


def test_run_label_propagation_freezes_observations():
    graph = nx.path_graph(2)

    result = run_label_propagation(
        graph,
        experimental_nodes=[0],
        iterations=10,
        freeze_experimental=True,
    )

    assert result == {
        0: 1,
        1: 1,
    }


def test_candidate_scoring():
    graph = nx.path_graph(3)
    aggregator = KemenyYoung(graph)

    scores = aggregator.score_candidate(
        {
            0: 1,
            1: 1,
            2: 0,
        }
    )

    assert scores == [2, 2, 0]

def test_candidate_selection():
    graph = nx.path_graph(3)
    aggregator = KemenyYoung(graph)

    selected = aggregator.first_selected_candidate(
        candidate_list=[
            {
                0: 1,
                1: 0,
                2: 0,
            },
            {
                0: 0,
                1: 1,
                2: 1,
            },
        ],
        candidate_names=[
            "A",
            "B",
        ],
    )

    expected = pd.Series(
        [
            ["A"],
            ["B"],
            ["B"],
        ],
        index=pd.Index(
            [0, 1, 2],
            name="node",
        ),
        name="selected_candidates",
    )

    pd.testing.assert_series_equal(
        selected,
        expected,
    )

def test_candidate_scoring_uses_active_neighbors():
    graph = nx.path_graph(4)
    aggregator = KemenyYoung(graph)

    candidate_a = {
        0: 1,
        1: 1,
        2: 1,
        3: 0,
    }

    candidate_b = {
        0: 0,
        1: 1,
        2: 0,
        3: 0,
    }

    selected = aggregator.first_selected_candidate(
        candidate_list=[
            candidate_a,
            candidate_b,
        ],
        candidate_names=[
            "A",
            "B",
        ],
    )

    # At node 1:
    # A has two active neighbors -> score 3.
    # B has zero active neighbors -> score 1.
    assert selected.loc[1] == ["A"]