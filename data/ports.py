import numpy as np
import torch
import random
import csv
import os
from itertools import combinations
from collections import Counter
import searoute as sr
import matplotlib.pyplot as plt
import seaborn as sns

# Load port coordinates from World Port Index CSV 
base_dir = os.path.dirname(__file__)
csv_file = os.path.join(base_dir, "UpdatedPub150.csv")

port_coordinates = {}
with open(csv_file, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            locode = row["UN/LOCODE"].strip()
            lat = float(row["Latitude"])
            lon = float(row["Longitude"])
            if locode and lat != 0 and lon != 0:
                port_coordinates[locode] = [lon, lat]
        except (ValueError, KeyError):
            continue

all_ports = list(port_coordinates.keys())
port_pair_counter = Counter()

print("Loaded ports from WPI CSV:", len(all_ports))
print("Sample ports:", all_ports[:5])

# Floyd-Warshall for triangle inequality 
def floyd_warshall(graph, num_ports):
    for k in range(num_ports):
        for i in range(num_ports):
            for j in range(num_ports):
                if graph[i, j] > graph[i, k] + graph[k, j]:
                    graph[i, j] = graph[i, k] + graph[k, j]
    return graph

#  Generate graphs 
def generate_realistic_graphs(num_graphs=1000, num_ports=4, min_distance=100):
    graphs = []
    attempts = 0
    max_attempts = num_graphs * 20
    valid_graphs_count = 0

    while len(graphs) < num_graphs and attempts < max_attempts:
        attempts += 1
        selected_ports = random.sample(all_ports, num_ports)
        selected_coords = [port_coordinates[port] for port in selected_ports]

        distances = np.zeros((num_ports, num_ports))
        valid = True

        for i, j in combinations(range(num_ports), 2):
            try:
                route = sr.searoute(selected_coords[i], selected_coords[j])
                dist = route.properties['length']
                if dist < min_distance:
                    valid = False
                    break
                distances[i, j] = distances[j, i] = dist

                port_pair = tuple(sorted((selected_ports[i], selected_ports[j])))
                port_pair_counter[port_pair] += 1
            except Exception:
                valid = False
                break

        if not valid:
            print(f"Skipping graph {attempts}: Invalid route or insufficient distance")
            continue

        distances = floyd_warshall(distances, num_ports)

        if np.any(distances < 0) or np.any(np.isnan(distances)):
            print(f"Skipping graph {attempts}: Invalid distances after Floyd-Warshall")
            continue

        graphs.append(distances)
        valid_graphs_count += 1

        if len(graphs) % 100 == 0:
            print(f"{len(graphs)} graphs collected...")

    print(f"Total valid graphs generated: {valid_graphs_count}")
    return graphs

#  Normalize edge weights (sum = 1) 
def preprocess_graphs(graphs):
    processed_graphs = []
    skipped_graphs = 0
    for graph in graphs:
        upper_sum = np.sum(graph[np.triu_indices(4, 1)])
        if upper_sum == 0:
            skipped_graphs += 1
            continue
        graph = graph / upper_sum
        processed_graphs.append(graph)
    
    print(f"Skipped {skipped_graphs} graphs due to zero upper sum")
    return processed_graphs

# Describe dataset 
def describe_graph_set(graphs_tensor):
    print("\n--- Graph Set Statistics ---")
    all_edges = graphs_tensor.flatten()
    print(f"Total edges: {all_edges.shape[0]}")
    print(f"Mean edge weight: {all_edges.mean():.4f}")
    print(f"STD edge weight: {all_edges.std():.4f}")
    print(f"Min edge weight: {all_edges.min():.4f}")
    print(f"Max edge weight: {all_edges.max():.4f}")

# Load 
if os.path.exists("real_graphs.npy"):
    real_graphs_tensor = torch.tensor(np.load("real_graphs.npy"), dtype=torch.float32)
    print(f"Loaded {real_graphs_tensor.shape[0]} real graphs from saved file.")
else:
    print("\nGenerating realistic sea-route graphs...")
    real_graphs = generate_realistic_graphs(num_graphs=1000, num_ports=4, min_distance=100)
    real_graphs = preprocess_graphs(real_graphs)
    real_graphs_tensor = np.array([graph[np.triu_indices(4, 1)] for graph in real_graphs])
    real_graphs_tensor = torch.tensor(real_graphs_tensor, dtype=torch.float32)
    np.save("real_graphs.npy", real_graphs_tensor.numpy())
    print("Saved real graphs to 'real_graphs.npy'.")

print("\nSummary:")
print(f"Total valid graphs: {len(real_graphs_tensor)}")
if len(real_graphs_tensor) > 0:
    print("Sample edge weights from first graph:", real_graphs_tensor[0])
    describe_graph_set(real_graphs_tensor)

if real_graphs_tensor.shape[0] == 0:
    print("Error: No valid graphs to train on.")
else:
    print(f"Final tensor shape: {real_graphs_tensor.shape}")

np.save("checkpoints/real_edge_weights.npy", real_graphs_tensor.numpy())

# Plot Real Data Edge-Weight Distribution 
all_real_weights = real_graphs_tensor.numpy().flatten()
plt.figure(figsize=(8, 5))
sns.kdeplot(all_real_weights, color="black", lw=2, label="Real Data")
plt.xlabel("Edge Weight", fontweight='bold')
plt.ylabel("Density", fontweight='bold')
plt.title("Real Data Edge-Weight Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("real_data_edge_weight_distribution.png", dpi=300)
plt.close()