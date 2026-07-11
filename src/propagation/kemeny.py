"""Label propagation and candidate aggregation for graph-based analysis."""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Hashable, Mapping, Sequence

import networkx as nx
import pandas as pd


def set_labels(
    graph: nx.Graph,
    experimental_nodes: Collection[Hashable],
) -> dict[Hashable, int]:
    """
    Assign binary labels to graph nodes.

    Nodes included in ``experimental_nodes`` receive label 1.
    All remaining graph nodes receive label 0.

    Parameters
    ----------
    graph
        Input NetworkX graph.
    experimental_nodes
        Nodes belonging to the experimental set.

    Returns
    -------
    dict
        Mapping from graph nodes to binary labels.

    Raises
    ------
    ValueError
        If an experimental node is absent from the graph.
    """
    experimental_nodes = set(experimental_nodes)
    graph_nodes = set(graph.nodes)

    unknown_nodes = experimental_nodes - graph_nodes

    if unknown_nodes:
        preview = ", ".join(
            map(str, sorted(unknown_nodes, key=str)[:5])
        )

        raise ValueError(
            "Experimental nodes are absent from the graph: "
            f"{preview}"
        )

    return {
        node: int(node in experimental_nodes)
        for node in graph.nodes
    }


def _validate_labels(
    graph: nx.Graph,
    labels: Mapping[Hashable, int],
) -> None:
    """Validate that labels form a complete binary graph assignment."""
    graph_nodes = set(graph.nodes)
    label_nodes = set(labels)

    missing_nodes = graph_nodes - label_nodes
    extra_nodes = label_nodes - graph_nodes

    if missing_nodes:
        preview = ", ".join(
            map(str, sorted(missing_nodes, key=str)[:5])
        )

        raise ValueError(
            "Labels are missing graph nodes: "
            f"{preview}"
        )

    if extra_nodes:
        preview = ", ".join(
            map(str, sorted(extra_nodes, key=str)[:5])
        )

        raise ValueError(
            "Labels contain nodes absent from the graph: "
            f"{preview}"
        )

    invalid_labels = {
        node: value
        for node, value in labels.items()
        if value not in (0, 1, False, True)
    }

    if invalid_labels:
        node, value = next(iter(invalid_labels.items()))

        raise ValueError(
            "Labels must be binary. "
            f"Node {node!r} has label {value!r}."
        )


def lpa(
    graph: nx.Graph,
    labels: Mapping[Hashable, int],
    iterations: int = 1000,
    *,
    fixed_nodes: Collection[Hashable] | None = None,
) -> dict[Hashable, int]:
    """
    Run synchronous deterministic binary label propagation.

    At each iteration, every node adopts the most frequent label among
    its neighbors. All updates are computed from the labels at the
    beginning of the iteration and are applied simultaneously.

    If labels 0 and 1 are equally represented among a node's neighbors,
    the node keeps its current label.

    Nodes without neighbors also keep their current label.

    Parameters
    ----------
    graph
        Input NetworkX graph.
    labels
        Complete initial binary label assignment.
    iterations
        Maximum number of synchronous propagation iterations.
    fixed_nodes
        Retained for compatibility with the previous interface.
        When provided, these nodes remain unchanged. In the intended
        coverage-classification workflow, this should be ``None``.

    Returns
    -------
    dict
        Final binary labels. After propagation, label 1 represents
        locally well-covered nodes and label 0 represents locally
        poorly-covered nodes.
    """
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise TypeError("iterations must be an integer.")

    if iterations < 0:
        raise ValueError("iterations must be non-negative.")

    _validate_labels(graph, labels)

    fixed_nodes = (
        set()
        if fixed_nodes is None
        else set(fixed_nodes)
    )

    unknown_fixed_nodes = fixed_nodes - set(graph.nodes)

    if unknown_fixed_nodes:
        preview = ", ".join(
            map(str, sorted(unknown_fixed_nodes, key=str)[:5])
        )

        raise ValueError(
            "Fixed nodes are absent from the graph: "
            f"{preview}"
        )

    current_labels = {
        node: int(labels[node])
        for node in graph.nodes
    }

    for _ in range(iterations):
        # Start from a copy so every update uses current_labels,
        # i.e. the labels at the beginning of this iteration.
        next_labels = current_labels.copy()

        for node in graph.nodes:
            if node in fixed_nodes:
                continue

            neighbors = list(graph.neighbors(node))

            # Isolated nodes retain their current state.
            if not neighbors:
                continue

            label_counts = Counter(
                current_labels[neighbor]
                for neighbor in neighbors
            )

            count_zero = label_counts.get(0, 0)
            count_one = label_counts.get(1, 0)

            if count_one > count_zero:
                next_labels[node] = 1

            elif count_zero > count_one:
                next_labels[node] = 0

            else:
                # Tie: retain the node's current label.
                next_labels[node] = current_labels[node]

        if next_labels == current_labels:
            break

        current_labels = next_labels

    return current_labels


def run_label_propagation(
    graph: nx.Graph,
    experimental_nodes: Collection[Hashable],
    iterations: int = 1000,
    *,
    freeze_experimental: bool = True,
) -> dict[Hashable, int]:
    """
    Initialize labels and classify nodes using synchronous propagation.

    Initially:

    - label 1 indicates an experimentally observed metabolite;
    - label 0 indicates an unobserved metabolite.

    During propagation, all nodes, including experimentally observed
    nodes, are allowed to change state. Therefore, after propagation:

    - label 1 indicates a locally well-covered node;
    - label 0 indicates a locally poorly-covered node.

    Parameters
    ----------
    graph
        Input NetworkX graph.
    experimental_nodes
        Experimentally observed nodes used to initialize the labels.
    iterations
        Maximum number of synchronous propagation iterations.
    freeze_experimental
        Retained only for backward compatibility. Experimental nodes are
        not frozen in this coverage-classification implementation.

    Returns
    -------
    dict
        Final binary coverage-state assignment.
    """
    experimental_nodes = set(experimental_nodes)

    initial_labels = set_labels(
        graph,
        experimental_nodes,
    )

    return lpa(
        graph,
        initial_labels,
        iterations=iterations,
        fixed_nodes=None,
    )

class KemenyYoung:
    """
    Aggregate propagated candidates using node-wise neighborhood scores.

    For a candidate active at node ``v``, its score is

        1 + degree(v)

    Otherwise, its score is zero.

    Notes
    -----
    This preserves the behavior of the original code. It is not the
    formal Kemeny-Young global rank-aggregation algorithm.
    """

    def __init__(self, graph: nx.Graph) -> None:
        if graph.number_of_nodes() == 0:
            raise ValueError(
                "The graph must contain at least one node."
            )

        self.graph = graph

    def score_candidate(
        self,
        status_matrix: Mapping[Hashable, int],
    ) -> list[int]:
        """
        Score active nodes using candidate-specific neighborhood support.

        An inactive node receives score zero. An active node receives one
        plus the number of its neighbors that are also active for the same
        candidate.
        """
        _validate_labels(
            self.graph,
            status_matrix,
        )

        scores = []

        for node in self.graph.nodes:
            if int(status_matrix[node]) == 0:
                scores.append(0)
                continue

            active_neighbors = sum(
                int(status_matrix[neighbor]) == 1
                for neighbor in self.graph.neighbors(node)
            )

            scores.append(
                1 + active_neighbors
            )

        return scores

    def vote_counting(
        self,
        all_candidate_scores: Sequence[Sequence[int]],
        candidate_names: Sequence[str],
    ) -> pd.DataFrame:
        """
        Combine candidate scores into a node-by-candidate table.
        """
        if len(all_candidate_scores) != len(candidate_names):
            raise ValueError(
                "all_candidate_scores and candidate_names "
                "must have the same length."
            )

        if len(candidate_names) == 0:
            raise ValueError(
                "At least one candidate is required."
            )

        if len(set(candidate_names)) != len(candidate_names):
            raise ValueError(
                "candidate_names must be unique."
            )

        number_of_nodes = self.graph.number_of_nodes()

        for candidate_name, scores in zip(
            candidate_names,
            all_candidate_scores,
            strict=True,
        ):
            if len(scores) != number_of_nodes:
                raise ValueError(
                    f"Candidate {candidate_name!r} has "
                    f"{len(scores)} scores, but the graph contains "
                    f"{number_of_nodes} nodes."
                )

        score_table = pd.DataFrame(
            all_candidate_scores,
            index=candidate_names,
        ).T

        score_table.index = pd.Index(
            list(self.graph.nodes),
            name="node",
        )

        return score_table

    def first_selected_candidate(
        self,
        candidate_list: Sequence[Mapping[Hashable, int]],
        candidate_names: Sequence[str],
    ) -> pd.Series:
        """
        Select the highest-scoring candidate at every graph node.

        Ties are retained. Nodes for which all candidates have score zero
        receive an empty list.
        """
        if len(candidate_list) != len(candidate_names):
            raise ValueError(
                "candidate_list and candidate_names "
                "must have the same length."
            )

        all_scores = [
            self.score_candidate(candidate)
            for candidate in candidate_list
        ]

        score_table = self.vote_counting(
            all_scores,
            candidate_names,
        )

        return score_table.apply(
            lambda row: (
                row.index[row == row.max()].tolist()
                if row.max() > 0
                else []
            ),
            axis=1,
        ).rename("selected_candidates")

    def propagate_and_select(
        self,
        experimental_sets: Sequence[Collection[Hashable]],
        candidate_names: Sequence[str],
        *,
        iterations: int = 1000,
        freeze_experimental: bool = True,
    ) -> tuple[list[dict[Hashable, int]], pd.Series]:
        """
        Run label propagation for multiple experiments and select winners.
        """
        if len(experimental_sets) != len(candidate_names):
            raise ValueError(
                "experimental_sets and candidate_names "
                "must have the same length."
            )

        propagated_candidates = [
            run_label_propagation(
                self.graph,
                experimental_nodes,
                iterations=iterations,
                freeze_experimental=freeze_experimental,
            )
            for experimental_nodes in experimental_sets
        ]

        selected_candidates = self.first_selected_candidate(
            propagated_candidates,
            candidate_names,
        )

        return propagated_candidates, selected_candidates

    # Backward-compatible names from the original script.
    score_candidat = score_candidate
    first_selected_candidat = first_selected_candidate


if __name__ == "__main__":
    graph = nx.path_graph(6)

    experimental_sets = [
        [0, 1],
        [4, 5],
    ]

    candidate_names = [
        "Candidate 1",
        "Candidate 2",
    ]

    aggregator = KemenyYoung(graph)

    propagated, selected = aggregator.propagate_and_select(
        experimental_sets=experimental_sets,
        candidate_names=candidate_names,
        iterations=20,
        freeze_experimental=True,
    )

    for name, candidate_status in zip(
        candidate_names,
        propagated,
        strict=True,
    ):
        print(f"{name}: {candidate_status}")

    print("\nSelected candidates:")
    print(selected)