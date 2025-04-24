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

if __name__ == "__main__":
    fichier = "p1.txt"
    n, capacites, couts = read_file(fichier)
    display_flow_data(n, capacites, couts)