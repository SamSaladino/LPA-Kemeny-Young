"""Label propagation algorithms for graph-based pathway analysis."""
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from joblib import Parallel, delayed
 
def set_labels(G, exp_status):
    """Set the initial labels for the nodes in the graph.
    The labels are based on the experimental status of the nodes.
    
    Parameters
    ----------
    G : NetworkX graph
    exp_status : list
        A list of nodes that are in the experimental set.
    
    Returns
    -------
    labels : dict"""
    labels = {nodes : (1 if nodes in exp_status else 0) for nodes in G.nodes()}
    return labels



def lpa(G, labels, iterations=1000):
    """Run the Label Propagation Algorithm. 
    For a given graph G with initial node labels,
    the algorithm proceeds by updating the label of each node
    to the one that appears most frequently among its neighbors.
    
    Parameters
    ----------
    G : NetworkX graph
    labels : dict
        A dictionary with nodes as keys and initial labels as values.
    iterations : int
        The number of times to run the algorithm.
    Returns
    -------
    labels : dict
        A dictionary with nodes as keys and label assignments as values."""
    
    for _ in range(iterations):
        for node in G.nodes():
            # Update the labels of the node's neighbors
            label_counts = {}
            for neighbor in G.neighbors(node):
                # Get the neighbor's label
                label = labels[neighbor]
                # Increment the count for that label
                label_counts[label] = label_counts.get(label, 0) + 1
            # Update the node's label
            max_count = 0
            # Initialize a list of labels that appear most frequently
            max_labels = []
            # Iterate over the labels and their counts
            for label, count in label_counts.items():
                # If the count of the current label is greater than the max_count
                if count > max_count:
                    # Update the max_count and reset the max_labels list
                    max_count = count
                    # Replace the max_labels list with the current label
                    max_labels = [label]
                # If the count is equal to the max_count
                elif count == max_count:
                    # Append the current label to the max_labels list
                    max_labels.append(label)
            # Select the initial label if conflicting labels
            if max_labels:  # Check if max_labels is not empty
                labels[node] = int(max_labels[0])

    return labels

def run_label_propagation(G,exp_status):
    """Wrapper to run LPA for a single experiment."""

    labels = set_labels(G, exp_status)
    
    return lpa(G, labels)

class KemenyYoung:
    def __init__(self, graph):
        self.graph = graph

    def score_candidat(self, status_matrix):
        """Score the method based on the status matrix and neighbor nodes.
        So you give it a status matrix and if status in node n is 1 return the name of the method
        and the score (given by neighbors count with status 1).

        Parameters
        ----------
        status_matrix : dict
            A dictionary with nodes as keys and label assignments as values.

        Returns
        -------
        list
            A list with the score of the node
        """
        scores = []
        for node in self.graph.nodes(): 
            if status_matrix[node] == 1:
                scores.append(1 + len(list(self.graph.neighbors(node))))
            else:
                scores.append(0)

        return scores

    def vote_counting(self, all_candidates_score, candidates_names):
        """Get the summary of the votes counting per node for each candidate.

        Parameters
        ----------
        all_candidates_score : list of list
            The list of votes for each candidate
        candidates_names : list of str
            The list of the candidates

        Returns
        -------
        pd.DataFrame
            A pandas dataframe of the votes summary
        """
        df = pd.DataFrame(all_candidates_score).T
        df.columns = candidates_names
        df.index = self.graph.nodes()
        return df

    def first_selected_candidat(self, candidat_list, candidates_names):
        """Get the first selected candidate or candidates based on the votes counting.

        Parameters
        ----------
        candidat_list : list of dict
            The list of the candidates
        candidates_names : list of str
            The list of the candidates

        Returns
        -------
        list
            A list of the first selected candidates
        """
        # Run in parallel the score_candidat function
        all_scores = Parallel(n_jobs=4)(
            delayed(self.score_candidat)(exp) for exp in candidat_list
        )
        # Get the votes counting on a dataframe and return the first selected candidate as a list
        first_selection = self.vote_counting(all_scores, candidates_names).apply(
            lambda row: row[row == row.max()].index.tolist() if row.max() > 0 else [], axis=1
        )

        return first_selection


if __name__ == "__main__":

# Create a sample graph
    G = nx.Graph()
    G.add_nodes_from(range(1, 60))

    # Arbitrary connections
    edges = [(random.randint(1, 60), random.randint(1, 60)) 
            for _ in range(100)]
    G.add_edges_from(edges)




    select_nodes1 = [14, 39, 50, 54, 56, 22, 44, 53, 16, 57, 47, 40, 55, 43, 10, 25, 9, 37, 19, 7, 21, 23, 31, 36, 52]
    select_nodes2 = [20, 18, 56, 40, 3, 30, 45, 34, 58, 31, 14, 32, 7, 28]

    Candidat1 = LPA.set_labels(G, select_nodes1)
    Candidat2 = LPA.set_labels(G, select_nodes2)

    votes_c1 = LPA.lpa(G, Candidat1, iterations=20)
    votes_c2 = LPA.lpa(G, Candidat2, iterations=20)

    ky = KemenyYoung(G)
    experiments = [votes_c1, votes_c2]
    all_scores = Parallel(n_jobs=4)(delayed(ky.score_candidat)(exp) for exp in experiments)
    print("All Scores:", all_scores)
    print(ky.first_selected_candidat(experiments,["Candidat1","Candidat2"]))

# # Plot the graph
# pos = nx.spring_layout(G)
# plt.figure(figsize=(10, 10))

# # Draw nodes
# nx.draw_networkx_nodes(G, pos, node_size=500, node_color='blue', alpha=0.8)
# nx.draw_networkx_nodes(G, pos, nodelist=select_nodes2, node_size=500, node_color='red', alpha=0.8)

# # Draw edges
# nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=1.0, alpha=0.5)

# # Draw labels
# nx.draw_networkx_labels(G, pos, font_size=12, font_color='white')

# plt.title('Graph with 30 Nodes and Arbitrary Connections')
# plt.show()



# print("Score Candidat1:", score_candidat(votes_c1,G))

# experiments = [Candidat1,Candidat2]
# all_scores = Parallel(n_jobs=4)(delayed(score_candidat)(exp, G) for exp in experiments)

# print("All Scores:", all_scores)

# print(vote_counting(all_scores,["Candidat1","Candidat2"],G))
# print(vote_counting(all_scores,["Candidat1","Candidat2"],G).apply(
#     lambda row: row[row == row.max()].index.tolist(), axis=1))

# print(first_selected_candidat(experiments,G,["Candidat1","Candidat2"]))