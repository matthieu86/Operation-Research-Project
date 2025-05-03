import random
import time
import matplotlib.pyplot as plt
"""julien fonction"""
def create_matrix(n):
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0
            else:
                matrix[i][j] = random.randint(1, 100)
    return matrix


# --- Provided Min-Cost Flow Function (DO NOT EDIT) ---

def print_matrix_with_labels(matrix, title):
    # A simple helper to print matrices.
    print(title)
    for row in matrix:
        print(row)


def Min_Cost_Flow(n, capacities, costs, source=0, sink=None):
    if sink is None:
        sink = n - 1

    flow = [[0] * n for _ in range(n)]
    residual_cap = [row[:] for row in capacities]
    residual_cost = [[costs[i][j] if capacities[i][j] > 0 else 0 for j in range(n)] for i in range(n)]
    total_flow = 0
    total_cost = 0
    iteration = 1

    while True:
        # === 1. Bellman-Ford on the residual graph ===
        dist = [float('inf')] * n
        parent = [-1] * n
        dist[source] = 0

        print(f"\n⋆ Bellman-Ford table (Iteration n° {iteration}) :")
        for _ in range(n - 1):
            for u in range(n):
                for v in range(n):
                    if residual_cap[u][v] > 0 and dist[u] + residual_cost[u][v] < dist[v]:
                        dist[v] = dist[u] + residual_cost[u][v]
                        parent[v] = u

        if parent[sink] == -1:
            print("\nNo upgrading chain found.")
            break

        # === 2. Reconstruct the path and compute the minimum capacity ===
        path = []
        v = sink
        while v != source:
            path.append(v)
            v = parent[v]
        path.append(source)
        path.reverse()

        min_cap = float('inf')
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            min_cap = min(min_cap, residual_cap[u][v])

        labels = ['s'] + [chr(97 + i) for i in range(n - 2)] + ['t']
        print(" → ".join([labels[v] for v in path]) + f" with flow = {min_cap}")

        # === 3. Update the residual graph and costs ===
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            residual_cap[u][v] -= min_cap
            residual_cap[v][u] += min_cap
            flow[u][v] += min_cap
            residual_cost[v][u] = -costs[u][v]  # inverse cost

        total_flow += min_cap
        total_cost += min_cap * dist[sink]

        print("Upgraded residual graph :")
        print_matrix_with_labels(residual_cap, f"Residual Graph {iteration}")

        iteration += 1

    print("\n⋆ Final flow with costs:")

    labels = ['s'] + [chr(97 + i) for i in range(n - 2)] + ['t']
    cell_width = 7

    # Header
    print("     " + "".join(f"{labels[j]:>{cell_width}}" for j in range(n)))
    print("     " + "-" * (cell_width * n))

    # Matrix rows
    for i in range(n):
        row = [f"{labels[i]:<3}|"]
        row.append("    ")
        for j in range(n):
            if capacities[i][j] > 0:
                row.append(f"{flow[i][j]:>2}/{capacities[i][j]:<3}".rjust(cell_width))
            else:
                row.append(" " * cell_width)
        print("".join(row))
    print(f"\nTotal flow cost = {total_cost}")
    time.sleep(0.001)
    return total_flow, total_cost


# --- End Provided Function ---


# Dummy stubs for Ford-Fulkerson and Push–Relabel.
def ford_fulkerson(C, source, sink):
    # Replace this stub with your actual Ford-Fulkerson algorithm implementation.
    # Dummy implementation to simulate execution time:
    time.sleep(0.001)
    return None


def push_relabel(C, source, sink):
    # Replace this stub with your actual Push-Relabel algorithm implementation.
    time.sleep(0.001)
    return None


def generate_flow_problem(n):
    """
    Generate a random flow problem.
    Creates two n×n matrices (capacities and costs) where:
      - All diagonal entries are 0.
      - Exactly floor(n^2/2) randomly selected off-diagonal entries are assigned a random integer in [1, 100].
      - The remaining entries remain 0.
    """
    capacities = [[0 for _ in range(n)] for _ in range(n)]
    costs = [[0 for _ in range(n)] for _ in range(n)]

    # List of candidate off-diagonal pairs
    candidates = [(i, j) for i in range(n) for j in range(n) if i != j]
    num_edges = int((n * n) / 2)

    # Sample edges without replacement
    selected_edges = random.sample(candidates, num_edges)

    for (i, j) in selected_edges:
        capacities[i][j] = random.randint(1, 100)
        costs[i][j] = random.randint(1, 100)

    return capacities, costs


# --- Complexity Study ---

# Define the set of problem sizes to be tested.
n_values = [10, 20, 40, 100, 400, 1000, 4000]

# Data storage for measured execution times for each algorithm.
results_ff = []  # Ford-Fulkerson
results_pr = []  # Push-Relabel
results_min = []  # Min-Cost Flow (using the provided function)

# For each value of n, run the experiment several times.
# (Due to extensive printing in Min_Cost_Flow, you might reduce the number of iterations.)
num_iterations = 10

for n in n_values:
    times_ff = []
    times_pr = []
    times_min = []

    for _ in range(num_iterations):
        # Generate random flow problem
        capacities, costs = generate_flow_problem(n)
        source, sink = 0, n - 1

        # Measure Ford-Fulkerson execution time.
        start = time.process_time()
        ford_fulkerson(capacities, source, sink)
        elapsed_ff = time.process_time() - start
        times_ff.append(elapsed_ff)

        # Measure Push–Relabel execution time.
        start = time.process_time()
        push_relabel(capacities, source, sink)
        elapsed_pr = time.process_time() - start
        times_pr.append(elapsed_pr)

        # Measure Min-Cost Flow execution time using your provided function.
        start = time.process_time()
        Min_Cost_Flow(n, capacities, costs, source, sink)
        elapsed_min = time.process_time() - start
        times_min.append(elapsed_min)

    results_ff.append((n, times_ff))
    results_pr.append((n, times_pr))
    results_min.append((n, times_min))

# --- Plotting the Results ---

plt.figure(figsize=(10, 6))

# Plot point cloud for Ford–Fulkerson (using red crosses)
for n, t_list in results_ff:
    plt.scatter([n] * len(t_list), t_list, c='red', marker='x', label='Ford-Fulkerson' if n == n_values[0] else "")

# Plot point cloud for Push–Relabel (using blue circles)
for n, t_list in results_pr:
    plt.scatter([n] * len(t_list), t_list, c='blue', marker='o', label='Push-Relabel' if n == n_values[0] else "")

# Plot point cloud for Min-Cost Flow (using green triangles)
for n, t_list in results_min:
    plt.scatter([n] * len(t_list), t_list, c='green', marker='^', label='Min-Cost Flow' if n == n_values[0] else "")

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Problem size n (vertices)')
plt.ylabel('Execution time (CPU seconds)')
plt.title('Complexity Study: Execution Times vs. Problem Size')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.show()