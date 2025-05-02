from collections import deque
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
                predecessor[y] = i

    return distances, predecessor
# C LA QUI FAUT METTRE LA SAVE

# C AU DESSUS QUI FAUT METTRE LA SAVE

# NOUERVZYIDIYUZIYUZDGBIDZPDZIUHDZIUHDZIUHDUHDZIUHDZIUHDZIUHDIUZHDZUIHDIZUHDZIUHDZUDZUIDZHIUIUDZHHIUDZIHUDZUIHDZIUHDZIUHDZIUHDZUHHUIDZUHIODZUHIDZUIHUIHDZHIUDZHUIDZIUHDZUIHDZUIH


# Affiche une matrice avec des étiquettes s, a, b, ..., t
# Utilisé pour afficher les matrices de capacités, résiduelles et de flot final
def print_matrix_with_labels(matrix, title="Matrix"):
    n = len(matrix)
    labels = ['s'] + [chr(97 + i) for i in range(n - 2)] + ['t']  # ['s', 'a', 'b', ..., 't']
    print(f"\n{title} ({n}x{n})")
    print("    " + "  ".join(f"{labels[j]:>3}" for j in range(n)))
    print("    " + "----" * n)
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

    print("\n⋆ Affichage de la table des capacités :")
    print_matrix_with_labels(capacities, "Capacités")

    flow = [[0] * n for _ in range(n)]  # Matrice des flots initiaux (tous à 0)
    residual = [row[:] for row in capacities]  # Graphe résiduel initial
    max_flow = 0
    iteration = 1

    print("\nLe graphe résiduel initial est le graphe de départ.")

    while True:
        print(f"\n⋆ Itération {iteration} :")
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

        print("Détection d’une chaîne améliorante : " + " → ".join(labels[v] for v in path) + f" de flot {min_capacity}.")

        # Met à jour le graphe résiduel et les flots
        for v in path[1:]:
            u = parent[v]
            residual[u][v] -= min_capacity
            residual[v][u] += min_capacity
            flow[u][v] += min_capacity

        print("Modifications sur le graphe résiduel :")
        print_matrix_with_labels(residual, "Graphe Résiduel")

        max_flow += min_capacity
        iteration += 1

    # Affiche la matrice des flots avec les valeurs / capacités
    print("\n⋆ Affichage du flot max :")
    print("Flot final :")
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

    print(f"\nValeur du flot max = {max_flow}")
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

    print("\n⋆ Résultat Push-Relabel :")
    labels = ['s'] + [chr(97 + i) for i in range(1, n - 1)] + ['t']
    print("     " + "   ".join(f"{l:>3}" for l in labels))
    print("     " + "----" * n)
    for i in range(n):
        row = [f"{labels[i]:<2}|"]
        for j in range(n):
            if capacity[i][j] > 0:
                row.append(f"{flow[i][j]}/{capacity[i][j]:>3}")
            else:
                row.append("     ")
        print(" ".join(row))

    max_flow = sum(flow[source][i] for i in range(n))
    print(f"\nValeur du flot max (Push-Relabel) = {max_flow}")
    return max_flow


if __name__ == "__main__":
    fichier = "p1.txt"
    n, capacities, costs = read_file(fichier)
    display_flow_data(n, capacities, costs)

    pp= "proposal 6.txt"
    a, capa, cout = read_file(pp)
    display_flow_data(a, capa, cout)

    if cout:
        distance, pred = Bellman_algo(a, cout, s=0)
        print("\nBellman Results:\n   Vertex  :   ", distance, "\nPredecessors : ",pred)
    else:
        print("Not a min-costs Problem")


    #visited_order = Breadth_Search(n, capacities, source=0)

    #print("Ordre de visite BFS :", [f"v{i + 1}" for i in visited_order])


    display_flow_data(n, capacities, costs)

   # source = 0  # v1
    #sink = n - 1  # v6

    #flow = ford_fulkerson(n, capacities, source, sink)
    #print(f"\nFlot maximum de v{source + 1} à v{sink + 1} : {flow}")

    print(Ford_Fulkerson(n, capacities))

    print("\n----------------------------------------------------------------------")
    print("\n----------------------------------------------------------------------")
    print("\n----------------------------------------------------------------------")

    print(Push_Relabel(n, capacities))