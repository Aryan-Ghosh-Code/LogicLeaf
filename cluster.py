# Hierarchical clustering (complete-link) on the provided 12 points.
# This runs the clustering, prints the full merge sequence (clusters merged and distances),
# shows the cluster "cover" after each merge, and draws a dendrogram using matplotlib.
#
# The output will be visible to you.
from math import sqrt
from itertools import combinations
import matplotlib.pyplot as plt

points = {
 'p1': (1,9),
 'p2': (2,10),
 'p3': (7,4),
 'p4': (10,3),
 'p5': (5,6),
 'p6': (6,11),
 'p7': (3,4),
 'p8': (4,9),
 'p9': (8,1),
 'p10': (3,12),
 'p11': (7,6),
 'p12': (11,2)
}

# compute pairwise distance
def euclid(a,b):
    return sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

names = list(points.keys())
n = len(names)

dist = {}
for i,j in combinations(names,2):
    d = euclid(points[i], points[j])
    dist[frozenset([i,j])] = d

# Initialize clusters: each point its own cluster (represented by frozenset of names)
clusters = [frozenset([name]) for name in names]

# helper: distance between clusters under complete-link = max distance between points in clusters
def cluster_distance(c1, c2):
    maxd = 0
    for a in c1:
        for b in c2:
            if a==b: continue
            d = dist[frozenset([a,b])]
            if d>maxd: maxd = d
    return maxd

merge_history = []  # list of tuples (step, clusterA, clusterB, newCluster, distance)
step = 0
clusters_current = clusters.copy()

# We'll keep order deterministic by sorting clusters by sorted tuple for tie-breaking
def cluster_key(c):
    return tuple(sorted(list(c)))

while len(clusters_current) > 1:
    # find pair with minimum complete-link distance
    best_pair = None
    best_dist = float('inf')
    for i in range(len(clusters_current)):
        for j in range(i+1, len(clusters_current)):
            c1 = clusters_current[i]
            c2 = clusters_current[j]
            d = cluster_distance(c1, c2)
            if d < best_dist - 1e-9 or (abs(d-best_dist) < 1e-9 and (cluster_key(c1), cluster_key(c2)) < (cluster_key(best_pair[0]), cluster_key(best_pair[1]))):
                best_dist = d
                best_pair = (c1, c2)
    cA, cB = best_pair
    newC = cA.union(cB)
    step += 1
    merge_history.append((step, cA, cB, newC, round(best_dist,4)))
    # update clusters list: remove cA and cB and add newC, maintain deterministic order
    clusters_next = [c for c in clusters_current if c!=cA and c!=cB]
    clusters_next.append(newC)
    clusters_next = sorted(clusters_next, key=lambda x: tuple(sorted(list(x))))
    clusters_current = clusters_next

# Print merge history and covers at each step
print("Hierarchical clustering (complete-link) merge sequence:\n")
for rec in merge_history:
    s, a, b, newc, d = rec
    print(f"Step {s:02d}: Merge {set(a)} and {set(b)} --> {set(newc)}   (distance = {d})")

# Print covers after each step (cover means current clusters after that merge)
clusters_current = [frozenset([name]) for name in names]
covers = [sorted([set(c) for c in clusters_current], key=lambda x: sorted(list(x)))]
for rec in merge_history:
    _, a, b, newc, _ = rec
    # update
    clusters_current = [c for c in clusters_current if c!=a and c!=b]
    clusters_current.append(newc)
    clusters_current = sorted(clusters_current, key=lambda x: tuple(sorted(list(x))))
    covers.append(sorted([set(c) for c in clusters_current], key=lambda x: sorted(list(x))))

print("\n\nCovers (clusters after each merge):\n")
for i,cover in enumerate(covers):
    print(f"After {i} merges: {cover}")

# Build data for dendrogram plotting
# Assign x positions to original leaves in alphabetical/name order for readability
leaf_order = sorted(names)  # p1..p12 sorted lexicographically
pos = {name: idx for idx,name in enumerate(leaf_order)}  # x positions
# For plotting, compute node positions iteratively using merge history in the order they occurred
node_positions = {}  # cluster frozenset -> (x_center, height)
# initial positions: singletons at height 0
for name in leaf_order:
    node_positions[frozenset([name])] = (pos[name], 0.0)

# Now compute for each merge: x_center = mean of child x's; height = distance from history
for rec in merge_history:
    _, a, b, newc, d = rec
    xa, ha = node_positions[a]
    xb, hb = node_positions[b]
    xcenter = (xa+xb)/2.0
    # height choose = d
    node_positions[newc] = (xcenter, d)

# Plot dendrogram
plt.figure(figsize=(10,5))
# draw leaves labels
for name in leaf_order:
    x,h = node_positions[frozenset([name])]
    plt.text(x, h-0.02, name, ha='center', va='top', fontsize=9)

# draw connections for each merge in sequence
for rec in merge_history:
    _, a, b, newc, d = rec
    xa, ha = node_positions[a]
    xb, hb = node_positions[b]
    xc, hc = node_positions[newc]
    # vertical lines from child to their height
    plt.plot([xa, xa], [ha, hc], linewidth=1)
    plt.plot([xb, xb], [hb, hc], linewidth=1)
    # horizontal line connecting children at height hc
    plt.plot([xa, xb], [hc, hc], linewidth=1)
    # mark cluster center
    plt.plot([xc], [hc], marker='o', markersize=3)

plt.xlabel("Points (sorted by name)")
plt.ylabel("Complete-link distance (height)")
plt.title("Dendrogram (Complete-link) for given points")
plt.ylim(-0.5, max(rec[4] for rec in merge_history)+1)
plt.xlim(-1, len(leaf_order))
plt.tight_layout()
plt.show()