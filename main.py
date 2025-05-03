from collections import deque
import time
def read_file(file):
    with open(file, "r") as f:
        lines = f.readlines()

    n = int(lines[0].strip())

    #reading matrices
    matrix_lines = [list(map(int, L.strip().split())) for L in lines[1:]]

    if len(matrix_lines) == n:
        # Cas d’un problème de flot max (1 matrice : capacités)
        capacity = matrix_lines
        cost = None
    elif len(matrix_lines) == 2 * n:
        # Cas d’un flot à coût min (2 matrices : capacités puis coûts)
        capacity = matrix_lines[:n]
        cost = matrix_lines[n:]
    else:
        raise ValueError("Not the expected format")

    return n, capacity, cost

def print_matrix(matrix, title="Matrix"):
    n = len(matrix)
    print(f"\n{title} ({n}x{n})")

    #Header row
    header = ["    "] + [f"v{j+1:>4}" for j in range(n)]
    print("".join(header))
    print("    " + "-----" * n)

    #Rows with labels
    for i in range(n):
        row = [f"v{i+1:<3}|"] + [f"{val:>5}" for val in matrix[i]]
        print("".join(row))

def display_flow_data(n, capacities, costs=None):
    print_matrix(capacities, "Capacity Matrix")
    if costs:
        print_matrix(costs, "Cost Matrix")


"""
A MODIFIER : 

- Prendre ce qui est stock en mememoire pas relire le file
- Attention la je met propo [1] mais comme c pr les propo >5 la cost matrix est en bas
- Le bail de mutlipilier float(inf) avec n 

"""
def Bellman_algo(n, costs_mat, s = 0):

    predecessor = [None]*n #
    distances = [float('inf')]*n # Initiate the distance of every vertex to infinty
    distances[s] = 0 # The distance from the source to itself is 0

    edges = []

    for i in range(n):
        for y in range(n):
            if costs_mat[i][y] != '0' and costs_mat[i][y] != 0: #and i != y
                edges.append((i,y,costs_mat[i][y]))
    
    # Bellman Algo
    for v in range(n-1):
        for i, y, cost in edges:
            if distances[i] + cost < distances[y]:
                distances[y] = distances[i] + cost
                if chr(97 + i) == 'a':
                    predecessor[y] = 's'
                else:
                    predecessor[y] = chr(96 + i)
    return distances, predecessor
# C LA QUI FAUT METTRE LA SAVE

# C AU DESSUS QUI FAUT METTRE LA SAVE

# NOUERVZYIDIYUZIYUZDGBIDZPDZIUHDZIUHDZIUHDUHDZIUHDZIUHDZIUHDIUZHDZUIHDIZUHDZIUHDZUDZUIDZHIUIUDZHHIUDZIHUDZUIHDZIUHDZIUHDZIUHDZUHHUIDZUHIODZUHIDZUIHUIHDZHIUDZHUIDZIUHDZUIHDZUIH

INF = float('inf')

def bellman_ford(n, residual_capacity, residual_cost, source, sink):
    dist = [INF] * n
    parent = [-1] * n
    dist[source] = 0

    for _ in range(n - 1):
        for u in range(n):
            for v in range(n):
                if residual_capacity[u][v] > 0 and dist[v] > dist[u] + residual_cost[u][v]:
                    dist[v] = dist[u] + residual_cost[u][v]
                    parent[v] = u

    if dist[sink] == INF:
        return None, None

    return parent, dist

def min_cost_max_flow(n, capacity, cost, source=0, sink=None):
    if sink is None:
        sink = n - 1

    flow = 0
    total_cost = 0

    # Initialize residual graphs
    residual_capacity = [row[:] for row in capacity]
    residual_cost = [row[:] for row in cost]

    # Add reverse edges for residual cost
    for u in range(n):
        for v in range(n):
            if capacity[u][v] > 0 and residual_capacity[v][u] == 0:
                residual_capacity[v][u] = 0
                residual_cost[v][u] = -cost[u][v]

    while True:
        parent, dist = bellman_ford(n, residual_capacity, residual_cost, source, sink)
        if parent is None:
            break  # No more augmenting path

        # Find bottleneck capacity (min residual capacity along the path)
        path_flow = INF
        v = sink
        while v != source:
            u = parent[v]
            path_flow = min(path_flow, residual_capacity[u][v])
            v = u

        # Update capacities and cost
        v = sink
        while v != source:
            u = parent[v]
            residual_capacity[u][v] -= path_flow
            residual_capacity[v][u] += path_flow
            total_cost += path_flow * residual_cost[u][v]
            v = u

        flow += path_flow

    return flow, total_cost





# Affiche une matrice avec des étiquettes s, a, b, ..., t
# Utilisé pour afficher les matrices de capacités, résiduelles et de flot final
def print_matrix_with_labels(matrix, title="Matrix"):
    n = len(matrix)
    labels = ['s'] + [chr(97 + i) for i in range(n - 2)] + ['t']  # ['s', 'a', 'b', ..., 't']
    print(f"\n{title} ({n}x{n})")
    print("    " + "  ".join(f"{labels[j]:>3}" for j in range(n)))
    print("    " + "-----" * n)
    for i in range(n):
        row = [f"{labels[i]:<2}|"] + [f"{matrix[i][j]:>4}" for j in range(n)]
        print(" ".join(row))

# Effectue une recherche en largeur dans le graphe résiduel pour trouver une chaîne améliorante
# Retourne un tableau des parents s'il existe un chemin de s à t, sinon None
def Breadth_Search(n, residual, source=0, sink=None):
    visited = [False] * n
    parent = [None] * n
    queue = deque([source])
    visited[source] = True

    while queue:
        u = queue.popleft()
        for v in range(n):
            if not visited[v] and residual[u][v] > 0:
                visited[v] = True
                parent[v] = u
                queue.append(v)
                if sink is not None and v == sink:
                    return parent
    return parent if visited[sink] else None

# Algorithme de Ford-Fulkerson pour calculer le flot maximum entre s (source) et t (puit)
def Ford_Fulkerson(n, capacities, source=0, sink=None):
    if sink is None:
        sink = n - 1  # Le dernier sommet est le puit t

    print("\n⋆ Capacity table printing :")
    print_matrix_with_labels(capacities, "Capacities")

    flow = [[0] * n for _ in range(n)]  # Matrice des flots initiaux (tous à 0)
    residual = [row[:] for row in capacities]  # Graphe résiduel initial
    max_flow = 0
    iteration = 1

    print("\nThe residual graph is the initial graph ")

    while True:
        print(f"\n⋆ Iteration {iteration} :")
        parent = Breadth_Search(n, residual, source, sink)

        if not parent:
            break  # Aucune chaîne améliorante trouvée → fin

        # Reconstruit le chemin s → t à partir du tableau des parents
        path = []
        v = sink
        while v != source:
            path.append(v)
            v = parent[v]
        path.append(source)
        path.reverse()

        # Détermine la capacité minimale (flot possible) dans ce chemin
        min_capacity = min(residual[parent[v]][v] for v in path[1:])

        labels = ['s'] + [chr(97 + i) for i in range(n - 2)] + ['t']
        print(" ".join([f"{labels[i]}" for i in range(len(parent)) if parent[i] is not None]))
        for v in path[1:]:
            print(f"Π({labels[v]}) = {labels[parent[v]]}")

        print("Upgrading chain detection : " + " → ".join(labels[v] for v in path) + f" of flow {min_capacity}.")

        # Met à jour le graphe résiduel et les flots
        for v in path[1:]:
            u = parent[v]
            residual[u][v] -= min_capacity
            residual[v][u] += min_capacity
            flow[u][v] += min_capacity

        print("Residual graph modification :")
        print_matrix_with_labels(residual, "Residual Graph")

        max_flow += min_capacity
        iteration += 1

    # Affiche la matrice des flots avec les valeurs / capacités
    print("\n⋆ Max flow printing :")
    print("Final flow :")
    labels = ['s'] + [chr(97 + i) for i in range(n - 2)] + ['t']
    print("     " + "   ".join(f"{l:>3}" for l in labels))
    print("     " + "----" * n)
    for i in range(n):
        row = [f"{labels[i]:<2}|"]
        for j in range(n):
            if capacities[i][j] > 0:
                row.append(f"{flow[i][j]}/{capacities[i][j]:>3}")
            else:
                row.append("     ")
        print(" ".join(row))

    print(f"\nMax flow value= {max_flow}")
    time.sleep(0.001)
    return max_flow

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def Push_Relabel(n, capacity, source=0, sink=None):
    if sink is None:
        sink = n - 1

    flow = [[0] * n for _ in range(n)]
    height = [0] * n
    excess = [0] * n

    height[source] = n

    for v in range(n):
        flow[source][v] = capacity[source][v]
        flow[v][source] = -flow[source][v]
        excess[v] = capacity[source][v]
        excess[source] -= capacity[source][v]

    def push(u, v):
        delta = min(excess[u], capacity[u][v] - flow[u][v])
        flow[u][v] += delta
        flow[v][u] -= delta
        excess[u] -= delta
        excess[v] += delta

    def relabel(u):
        min_height = float('inf')
        for v in range(n):
            if capacity[u][v] - flow[u][v] > 0:
                min_height = min(min_height, height[v])
        if min_height < float('inf'):
            height[u] = min_height + 1

    def discharge(u):
        while excess[u] > 0:
            for v in range(n):
                if capacity[u][v] - flow[u][v] > 0 and height[u] == height[v] + 1:
                    push(u, v)
                    if excess[u] == 0:
                        break
            else:
                relabel(u)

    active = [i for i in range(n) if i != source and i != sink and excess[i] > 0]
    p = 0
    while p < len(active):
        u = active[p]
        old_height = height[u]
        discharge(u)
        if height[u] > old_height:
            active.insert(0, active.pop(p))
            p = 0
        else:
            p += 1

    print("\n⋆ Push-Relabel Results:")
    labels = ['s'] + [chr(97 + i) for i in range(1, n - 1)] + ['t']
    print("     " + "   ".join(f"{l:>3}" for l in labels))
    print("     " + "----" * n)
    for i in range(n):
        row = [f"{labels[i]:<2}|"]
        row.append(" ")
        for j in range(n):
            if capacity[i][j] > 0:
                row.append(f"{flow[i][j]}/{capacity[i][j]:<3}")
            else:
                row.append("     ")
        print(" ".join(row))

    max_flow = sum(flow[source][i] for i in range(n))
    print(f"\nMaximum-flow value (Push-Relabel) = {max_flow}")
    time.sleep(0.001)

    return max_flow



""" 
MIN COST PROBLEM A PARTIR DE DESSOUS :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


jai ff hier soir. Mais vous pouvez demander à chat de le faire normalement il fait un truc pas mal.

Demander lui de faire un truc py pour resoudre problm de flot a cout min en utilisant l'algo de bellman ford.
et faut qu'il respect les conditions suiivantes:

Algorithme pour résoudre le flot à coût minimal.
⋆ La table détaillée de Bellman.
⋆ Valeur de flot d’une chaîne améliorante potentiellement trouvée.
⋆ Les modifications sur le graphe résiduel. 
"""

"""
#############################################
if __name__ == "__main__":
    fichier = "p1.txt"
    n, capacities, costs = read_file(fichier)
    display_flow_data(n, capacities, costs)

    pp= "proposal 6.txt"
    a, capa, cout = read_file(pp)
    display_flow_data(a, capa, cout)

    

    #visited_order = Breadth_Search(n, capacities, source=0)

    #print("Ordre de visite BFS :", [f"v{i + 1}" for i in visited_order])


    display_flow_data(n, capacities, costs)

   # source = 0  # v1
    #sink = n - 1  # v6

    #flow = ford_fulkerson(n, capacities, source, sink)
    #print(f"\nFlot maximum de v{source + 1} à v{sink + 1} : {flow}")

    print(Ford_Fulkerson(n, capacities))


    if cout:
        distance, pred = Bellman_algo(a, cout, s=0)
        print("\nBellman Results:\n   Costs  :   ", distance, "\nPredecessors : ",pred)
    else:
        print("Not a min-costs Problem")
    print("\n----------------------------------------------------------------------")
    print("\n----------------------------------------------------------------------")
    print("\n----------------------------------------------------------------------")

    print(Push_Relabel(n, capacities))
    print(costs)
    
    if cout:
        Min_Cost_Flow(a, capa, cout)
    else:
        print("No costs matrix available.")
"""
 


while True:
    chosen_proposition = input("Please enter the number or the proposal you want to use. (1 to 10)")
    chosen_proposition = "proposals Projet OR/proposal " + chosen_proposition + ".txt"
    n, capacity, costs = read_file(chosen_proposition)
    display_flow_data(n, capacity, costs)
    if costs:
        #BelCosts, BelPred = Bellman_algo(n, costs)
        #print("\nBellman Results:\n   Costs  :   ", BelCosts, "\nPredecessors : ",BelPred)
        #Min_Cost_Flow(n, capacity, costs)
        flow, cost = min_cost_max_flow(n, capacity, costs)
        print(f"\nMax flow = {flow}, Min cost = {cost}")

    else :
        print(Ford_Fulkerson(n, capacity))
        print(Push_Relabel(n, capacity))
    