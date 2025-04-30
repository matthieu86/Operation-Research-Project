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


if __name__ == "__main__":
    fichier = "p1.txt"
    n, capacites, costs = read_file(fichier)
    display_flow_data(n, capacites, costs)

    pp= "proposal 6.txt"
    a, capa, cout = read_file(pp)
    display_flow_data(a, capa, cout)

    if cout:
        distance, pred = Bellman_algo(a, cout, s=0)
        print("\nBellman Results:\n   Vertex  :   ", distance, "\nPredecessors : ",pred)
    else:
        print("Not a min-costs Problem")