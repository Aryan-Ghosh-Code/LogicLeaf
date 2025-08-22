import pandas as pd
import numpy as np
from collections import Counter
from graphviz import Digraph

data = pd.read_csv("weekend.csv")
features = list(data.columns[1:-1])
target = "Decision"

def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    return -sum((count/total) * np.log2(count/total) for count in counts.values())

def info_gain(df, feature, target):
    base_entropy = entropy(df[target])
    values = df[feature].unique()
    weighted_entropy = sum(
        (len(subset)/len(df)) * entropy(subset[target])
        for v in values
        if len((subset := df[df[feature] == v])) > 0
    )
    return base_entropy - weighted_entropy

def gini(labels):
    counts = Counter(labels)
    total = len(labels)
    return 1 - sum((count/total)**2 for count in counts.values())

def gini_index(df, feature, target):
    values = df[feature].unique()
    weighted_gini = sum(
        (len(subset)/len(df)) * gini(subset[target])
        for v in values
        if len((subset := df[df[feature] == v])) > 0
    )
    return weighted_gini

def build_tree(df, features, target, method):
    labels = df[target].tolist()

    if labels.count(labels[0]) == len(labels):
        return labels[0]  # Pure leaf
    if not features:
        return Counter(labels).most_common(1)[0][0] 

    if method == "entropy":
        gains = {f: info_gain(df, f, target) for f in features}
        best_feature = max(gains, key=gains.get)
    else:
        ginis = {f: gini_index(df, f, target) for f in features}
        best_feature = min(ginis, key=ginis.get)

    tree = {best_feature: {}}

    for v in df[best_feature].unique():
        subset = df[df[best_feature] == v].drop(columns=[best_feature])
        sub_features = [f for f in features if f != best_feature]
        tree[best_feature][v] = build_tree(subset, sub_features, target, method)

    return tree

node_counter = 0
def unique_id():
    global node_counter
    node_counter += 1
    return f"node{node_counter}"

def visualize_tree(tree, graph=None, parent=None, edge_label=""):
    if graph is None:
        graph = Digraph(format="png")
        graph.attr(rankdir="TB", splines="polyline")  # Better layout
        graph.attr("node", fontname="Helvetica")

    if isinstance(tree, dict):
        for feature, branches in tree.items():
            node_id = unique_id()
            graph.node(node_id, feature, shape="box", style="rounded,filled", color="lightblue")
            if parent:
                graph.edge(parent, node_id, label=edge_label)

            for value, subtree in branches.items():
                visualize_tree(subtree, graph, node_id, str(value))
    else:
        leaf_id = unique_id()
        graph.node(leaf_id, str(tree), shape="ellipse", style="filled", color="lightgreen")
        if parent:
            graph.edge(parent, leaf_id, label=edge_label)

    return graph

tree_entropy = build_tree(data, features, target, method="entropy")
graph_entropy = visualize_tree(tree_entropy)
graph_entropy.render("tree_entropy", view=True)

tree_gini = build_tree(data, features, target, method="gini")
graph_gini = visualize_tree(tree_gini)
graph_gini.render("tree_gini", view=True)