"""
AUTRE VERSION DE PARCOURS EN LARGEUR PAR CHAT

PLUS DETAILLE ET AVEC DES TRUCS EN PLUS

Voici une version améliorée de ta fonction BFS qui calcule à la fois :

La distance (en nombre d’arêtes) entre le sommet de départ et chaque sommet.
Le chemin le plus court en termes de nombre de sauts (pas de poids pris en compte).
Le tout, dans le style et la logique que tu avais commencé à utiliser.
"""

from collections import deque

def Breadth_Search(n, capacities, source=0):
    visited = [False] * n
    distance = [float('inf')] * n
    predecessor = [None] * n

    queue = deque()
    queue.append(source)
    visited[source] = True
    distance[source] = 0

    while queue:
        current = queue.popleft()
        for neighbor in range(n):
            if capacities[current][neighbor] != 0 and not visited[neighbor]:
                visited[neighbor] = True
                distance[neighbor] = distance[current] + 1
                predecessor[neighbor] = current
                queue.append(neighbor)

    return distance, predecessor


def print_bfs_paths(source, distance, predecessor):
    print(f"\nParcours en largeur depuis v{source+1} :")
    for dest in range(len(distance)):
        if distance[dest] == float('inf'):
            print(f"v{dest+1} : inaccessible")
        else:
            path = []
            current = dest
            while current is not None:
                path.insert(0, f"v{current+1}")
                current = predecessor[current]
            print(f"v{dest+1} : distance = {distance[dest]}, chemin = {' → '.join(path)}")


n, capacities, costs = read_file("graphe.txt")
distance, pred = Breadth_Search(n, capacities, source=0)
print_bfs_paths(0, distance, pred)
