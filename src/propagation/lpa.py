import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

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

G = nx.Graph()

edges = [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4), (3, 4)]
G.add_edges_from(edges)

exp_status = [1, 4, 2]



status = set_labels(G, exp_status)
print(lpa(G, status, iterations=100))

# pos = nx.spring_layout(G, seed=42)
# plt.figure(figsize=(8, 8))
# colors = [karate_label_color_map[exp_status] for node in G.nodes()]
# nx.draw(G, pos, node_color=colors, with_labels=True, font_weight='bold', cmap=plt.cm.viridis)
# nx.draw_networkx_labels(G, pos, {node: initial_labels_karate[node] for node in G.nodes() if initial_labels_karate[node] == 1})
# plt.show()



G_karate = nx.karate_club_graph()
initial_labels_karate = {node: 1 if random.random() > 0.8 else 0 for node in G_karate.nodes()}
karate_label_color_map = {0: "gray", 1: "blue"}

pos = nx.spring_layout(G_karate, seed=42)
plt.figure(figsize=(8, 8))
colors = [karate_label_color_map[initial_labels_karate[node]] for node in G_karate.nodes()]
nx.draw(G_karate, pos, node_color=colors, with_labels=True, font_weight='bold', cmap=plt.cm.viridis)
nx.draw_networkx_labels(G_karate, pos, {node: initial_labels_karate[node] for node in G_karate.nodes() if initial_labels_karate[node] == 1})
plt.show()

new_labels = lpa(G_karate, initial_labels_karate, iterations=1000)
print(new_labels)

pos = nx.spring_layout(G_karate, seed=42)
plt.figure(figsize=(8, 8))
colors = [karate_label_color_map[new_labels[node]] for node in G_karate.nodes()]
nx.draw(G_karate, pos, node_color=colors, with_labels=True, font_weight='bold', cmap=plt.cm.viridis)
nx.draw_networkx_labels(G_karate, pos, {node: new_labels[node] for node in G_karate.nodes() if new_labels[node] == 1})
plt.show()


# Code to create the first example graph used initially (Zachary's Karate Club Graph)

# Create the graph (Zachary's Karate Club)
G_karate = nx.karate_club_graph()  # A standard test graph with a community structure

# Randomly assign initial labels with only two labels: 0 (unlabeled) and 1 (labeled)
# In this case, a small portion of nodes are labeled 1 initially, others remain 0
initial_labels_karate = {node: 1 if random.random() > 0.8 else 0 for node in G_karate.nodes()}

# Define color mapping for visualization (blue for labeled, gray for unlabeled)
karate_label_color_map = {0: "gray", 1: "blue"}

# Plotting function for the Karate Club graph with initial labels
def plot_karate_graph(graph, labels, title="Initial Labeled Karate Club Graph"):
    pos = nx.spring_layout(graph, seed=42)
    plt.figure(figsize=(8, 8))
    colors = [karate_label_color_map[labels[node]] for node in graph.nodes()]
    nx.draw(graph, pos, node_color=colors, with_labels=True, font_weight='bold', cmap=plt.cm.viridis)
    nx.draw_networkx_labels(graph, pos, {node: labels[node] for node in graph.nodes() if labels[node] == 1})
    plt.title(title)
    plt.show()

# Show the initial Karate Club graph with random labels
plot_karate_graph(G_karate, initial_labels_karate)

from sklearn.semi_supervised import LabelPropagation
import numpy as np

# Prepare data for scikit-learn's LabelPropagation
# Convert graph to adjacency matrix format for sklearn compatibility
adj_matrix = nx.to_numpy_array(G_karate)

# Initial label setup for scikit-learn's LabelPropagation (using -1 for unlabeled nodes)
# sklearn's LabelPropagation requires a complete label array with -1 for unknown labels
initial_labels_sklearn = np.array([1 if initial_labels_karate[node] == 1 else -1 for node in G_karate.nodes()])

# Apply Label Propagation using scikit-learn
label_prop_model = LabelPropagation(max_iter=500, kernel='rbf')  # rbf kernel suited for graphs
label_prop_model.fit(adj_matrix, initial_labels_sklearn)

# Get the propagated labels from the model
final_labels_sklearn = label_prop_model.transduction_

# Convert back to a dictionary format for easier visualization
final_labels_dict = {node: int(final_labels_sklearn[node]) for node in G_karate.nodes()}

# Plot the final labeled Karate Club graph using sklearn's LabelPropagation
plot_karate_graph(G_karate, final_labels_dict, "Final Labeled Karate Club Graph After sklearn LPA")
