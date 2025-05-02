
""" A AJOUTER ENTRE LES COMMENTAIRE DEDIÉ"""

" Peut etre faut ajouter un truc de shortest past ??"

"Bref la j'en suis à F-F partie 3 projet"

"""
VERSION 1 
def Breatdh_Search(n, capacities, source=0):
    path = []
    queue = [source]
    while len(queue) != 0:
        new = queue[-1]
        if new not in path:
            path.append(new)
            for y in range(n):
                if capacities[queue[-1]][y] not in path: #and capacities[queue[-1]][y] not in queue
                    #queue.append(capacities[queue[-1]][y])
                    queue.append(y)
                queue.pop(-1)    




        for i in range(n):
            for y in range(n):
                if capacities[i][y] not in path:
                    queue.append(capacities[i][y])


    return"""

"2e version du parcours en lagrgeur doit marcher normalement"


def Breadth_Search(n, capacities, source=0):
    visited = []
    queue = deque([source])

    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.append(current)
            for neighbor in range(n):
                if capacities[current][neighbor] != 0 and neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor)

    return visited


"""
def Ford_Fulkerson(n, capacities, source=0):

    for i in range(n):
        if capacities[i][-1] == 0: #check que T ait des predecesseurs car sinon c'est deja opti / Donc check si la last column est que 0
            return "The problem is already solved"

    chaine_ameliorante = Breadth_Search(n, capacities, source)

"""

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

"""FF avec une nouvelle version de parcours en largeur (encore)
car mon precedent parcours L (ui est ecnore en haut) ne faisais pas tous ce qui est necessaire pour FF / Edmond karp

A MODIF POUR AVOIR TOUTES LES ETAPES DEMANDE DANS LE SUJET
"""


def breadth_search_augmenting_path(residual, source, sink, parent):
    n = len(residual)
    visited = [False] * n
    queue = deque([source])
    visited[source] = True

    while queue:
        u = queue.popleft()
        for v in range(n):
            if not visited[v] and residual[u][v] > 0:
                parent[v] = u
                visited[v] = True
                queue.append(v)
                if v == sink:
                    return True
    return False


def bfs_augmenting_path(residual, source, sink, parent, node_names=None):
    n = len(residual)
    visited = [False] * n
    queue = deque([source])
    visited[source] = True
    levels = [[] for _ in range(n)]  # Pour tracer les étapes

    while queue:
        u = queue.popleft()
        for v in range(n):
            if not visited[v] and residual[u][v] > 0:
                parent[v] = u
                visited[v] = True
                queue.append(v)
                levels[v] = levels[u] + [v]
                if v == sink:
                    return True, levels[v]
    return False, []


def reconstruct_path(parent, source, sink):
    path = []
    v = sink
    while v != source:
        path.insert(0, v)
        v = parent[v]
    path.insert(0, source)
    return path


def print_matrix(matrix, title="Matrix", node_names=None):
    n = len(matrix)
    print(f"\n{title} ({n}x{n})")
    if node_names is None:
        node_names = [chr(115 + i) for i in range(n)]  # ['s','a','b','c'...]

    header = ["    "] + [f"{name:>4}" for name in node_names]
    print("".join(header))
    print("    " + "-----" * n)
    for i in range(n):
        row = [f"{node_names[i]:<3}|"] + [f"{val:>5}" for val in matrix[i]]
        print("".join(row))
