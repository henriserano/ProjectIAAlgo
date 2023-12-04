import pandas as pd
import heapq
import random
from collections import deque
from itertools import compress


def heuristic(biscuits, remaining_length):
    max_value_per_length = max(biscuit.value / biscuit.length for biscuit in biscuits)
    return max_value_per_length * remaining_length

def load_defects(csv_filepath):
    return pd.read_csv(csv_filepath)

class Biscuit:
    def __init__(self, length, value, defect_thresholds):
        self.length = length
        self.value = value
        self.defect_thresholds = defect_thresholds
        self.children_indices = []  # Ajoutez cette ligne pour initialiser l'attribut

    def add_child(self, child_index):
        self.children_indices.append(child_index) # Ajout de la liste des enfants pour chaque biscuit

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value

    def add_child(self, child):
        self.children_indices.append(child)

class Defect:
    def __init__(self, position, defect_class):
        self.position = position
        self.defect_class = defect_class

class DoughRoll:
    def __init__(self, length, defects):
        self.length = length
        self.defects = defects

def overlaps_with_defects(position, biscuit, defects):
    defect_thresholds = biscuit.defect_thresholds.copy()
    for defect in defects:
        if position <= defect.position < position + biscuit.length:
            if defect_thresholds.get(defect.defect_class, 0) <= 0:
                return True
            defect_thresholds[defect.defect_class] -= 1
    return False

def search(dough_roll, biscuits):
    start_state = (0, [], 0)  # (position actuelle, indices des biscuits choisis, valeur totale actuelle)
    frontier = [(-0, start_state)]  # (priorité, état)
    explored = set()
    best_solution = ([], 0)  # (indices des biscuits choisis, valeur totale actuelle)

    while frontier:
        _, current_state = heapq.heappop(frontier)
        current_position, biscuits_indices, current_value = current_state

        # Mise à jour de la meilleure solution si la valeur actuelle est plus élevée
        if current_value > best_solution[1]:
            best_solution = (biscuits_indices, current_value)

        hashable_state = (current_position, tuple(biscuits_indices))
        if hashable_state in explored:
            continue
        explored.add(hashable_state)

        # Calcul de la longueur de pâte restante
        remaining_length = dough_roll.length - current_position

        for i, biscuit in enumerate(biscuits):
            if not biscuits_indices or i in biscuits[biscuits_indices[-1]].children_indices:
                new_position = current_position + biscuit.length
                if new_position <= dough_roll.length and not overlaps_with_defects(current_position, biscuit, dough_roll.defects):
                    new_biscuits_indices = biscuits_indices + [i]
                    new_value = current_value + biscuit.value
                    priority = -new_value - heuristic(biscuits, remaining_length)
                    heapq.heappush(frontier, (priority, (new_position, new_biscuits_indices, new_value)))

    # Retourne la meilleure solution trouvée, même si elle n'utilise pas toute la longueur de la pâte
    return [biscuits[i] for i in best_solution[0]]


def optimize_biscuit_placement(csv_filepath):
    defects_df = load_defects(csv_filepath)
    # Assurez-vous que la position est convertie en entier
    defects = [Defect(int(row['x']), row['class']) for _, row in defects_df.iterrows()]
    dough_roll = DoughRoll(500, defects)

    biscuits = [
        Biscuit(4, 6, {'a': 4, 'b': 2, 'c': 3}),
        Biscuit(8, 12, {'a': 5, 'b': 4, 'c': 4}),
        Biscuit(2, 1, {'a': 1, 'b': 2, 'c': 1}),
        Biscuit(5, 8, {'a': 2, 'b': 3, 'c': 2}),
    ]

    # Construire les liens entre les biscuits en tenant compte des défauts de la pâte
    for i, biscuit in enumerate(biscuits):
        for j, next_biscuit in enumerate(biscuits):
            if not overlaps_with_defects(biscuit.length, next_biscuit, dough_roll.defects):
                biscuit.add_child(j)  # Utilisez les indices au lieu des objets

    solution = search(dough_roll, biscuits)
    return solution, defects

def optimize_biscuit_placementt(csv_filepath):
    defects_df = load_defects(csv_filepath)
    # Assurez-vous que la position est convertie en entier
    defects = [Defect(int(row['x']), row['class']) for _, row in defects_df.iterrows()]
    dough_roll = DoughRoll(500, defects)

    biscuits = [
        Biscuit(4, 6, {'a': 4, 'b': 2, 'c': 3}),
        Biscuit(8, 12, {'a': 5, 'b': 4, 'c': 4}),
        Biscuit(2, 1, {'a': 1, 'b': 2, 'c': 1}),
        Biscuit(5, 8, {'a': 2, 'b': 3, 'c': 2}),
    ]

    # Construire les liens entre les biscuits en tenant compte des défauts de la pâte
    for i, biscuit in enumerate(biscuits):
        for j, next_biscuit in enumerate(biscuits):
            if not overlaps_with_defects(biscuit.length, next_biscuit, dough_roll.defects):
                biscuit.add_child(j)  # Utilisez les indices au lieu des objets

    #solution = search(dough_roll, biscuits)
    return dough_roll, biscuits

# ... autres parties du code inchangées ...

def print_solution(biscuits_sequence):
    if biscuits_sequence:
        print("Une combinaison de biscuits a été trouvée :")
        for i, biscuit in enumerate(biscuits_sequence):
            print(f"Biscuit {i}: Longueur {biscuit.length}, Valeur {biscuit.value}, Seuils de défauts {biscuit.defect_thresholds}")
        total_length = sum(biscuit.length for biscuit in biscuits_sequence)
        total_value = sum(biscuit.value for biscuit in biscuits_sequence)
        print(f"Longueur totale utilisée : {total_length}")
        print(f"Valeur totale : {total_value}")
    else:
        print("Aucune combinaison de biscuits n'a été trouvée.")

# Utiliser la fonction pour optimiser le placement des biscuits et imprimer la solutio
def print_dough_visualization(biscuits_sequence, defects):
    # Créez une ligne pour la pâte, initialement vide
    dough_line = [' ' for _ in range(500)]
    
    # Marquez les défauts dans la ligne de pâte
    for defect in defects:
        dough_line[defect.position] = 'D'  # D pour défaut

    # Marquez les biscuits dans la ligne de pâte
    current_position = 0
    for biscuit in biscuits_sequence:
        for i in range(biscuit.length):
            if dough_line[current_position + i] == ' ':
                dough_line[current_position + i] = 'B'  # B pour biscuit
        current_position += biscuit.length

    # Convertissez la ligne de pâte en une chaîne et imprimez-la
    print(''.join(dough_line))

# Utiliser la fonction pour optimiser le placement des

def a_star_search(dough_roll, biscuits):
    # Assume heuristic is defined elsewhere
    start_state = (0, [], 0)  # (current position, indices of biscuits used, total value)
    frontier = [(-heuristic(biscuits, dough_roll.length, 0), start_state)]
    explored = set()
    best_solution = ([], 0)  # (indices of biscuits used, total value)

    while frontier:
        _, (current_position, biscuits_indices, current_value) = heapq.heappop(frontier)

        # Check if we can improve the best solution
        if current_value > best_solution[1] and not is_end_of_roll(current_position, dough_roll):
            best_solution = (biscuits_indices, current_value)

        # Add states to the frontier
        for i, biscuit in enumerate(biscuits):
            new_position = current_position + biscuit.length
            new_biscuits_indices = biscuits_indices + [i]
            new_value = current_value + biscuit.value
            if new_position <= dough_roll.length and not overlaps_with_defects(current_position, biscuit, dough_roll.defects):
                # Check if the new value is within the acceptable range
                if 100 <= new_value <= 500:
                    heapq.heappush(frontier, (-new_value, (new_position, new_biscuits_indices, new_value)))

    # Construct the solution from the best indices
    solution = [biscuits[i] for i in best_solution[0]]
    return solution, best_solution[1]


def is_end_of_roll(current_position, dough_roll):
    """
    Checks if the current position is at the end of the dough roll.

    Parameters:
    current_position (int): The current position on the dough roll.
    dough_roll (DoughRoll): The dough roll object with a specified length.

    Returns:
    bool: True if the end of the roll is reached, False otherwise.
    """
    return current_position >= dough_roll.length

def get_neighbors(current_solution, biscuits, dough_roll):
    """
    Generates neighboring solutions by swapping each pair of biscuits.

    Parameters:
    current_solution (list of Biscuit): The current sequence of biscuits.
    biscuits (list of Biscuit): The complete list of available biscuits.
    dough_roll (DoughRoll): The dough roll object for checking defects.

    Returns:
    list of list of Biscuit: A list of neighbor solutions.
    """
    neighbors = []
    length = len(current_solution)

    for i in range(length):
        for j in range(i + 1, length):
            if can_swap(i, j, current_solution, dough_roll):
                neighbor = current_solution[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)

    return neighbors

def can_swap(i, j, solution, dough_roll):
    """
    Checks if two biscuits can be swapped without causing overlap with defects.

    Parameters:
    i (int): Index of the first biscuit in the solution.
    j (int): Index of the second biscuit in the solution.
    solution (list of Biscuit): The current sequence of biscuits.
    dough_roll (DoughRoll): The dough roll object for checking defects.

    Returns:
    bool: True if the biscuits can be swapped, False otherwise.
    """
    # Assume calculate_position is a function that determines the starting position of a biscuit in the dough
    position_i = calculate_position(i, solution)
    position_j = calculate_position(j, solution)

    # Check if swapping would cause either biscuit to overlap with a defect
    if overlaps_with_defects(position_i, solution[j], dough_roll.defects) or \
       overlaps_with_defects(position_j, solution[i], dough_roll.defects):
        return False
    return True

def calculate_position(index, solution):
    """
    Calculates the starting position of the biscuit at the given index in the solution.

    Parameters:
    index (int): The index of the biscuit in the solution.
    solution (list of Biscuit): The current sequence of biscuits.

    Returns:
    int: The starting position of the biscuit in the dough.
    """
    position = 0
    for biscuit in solution[:index]:
        position += biscuit.length
    return position



def dfs_search(dough_roll, biscuits):
    # Sort biscuits by the value-to-length ratio for efficient coverage
    biscuits.sort(key=lambda b: b.value / b.length, reverse=True)
    
    stack = [(0, [], 0)]  # (current position, indices of biscuits used, total value)
    best_solution = ([], 0)

    while stack:
        current_position, biscuits_indices, current_value = stack.pop()

        if current_value > best_solution[1] and not is_end_of_roll(current_position, dough_roll):
            best_solution = (biscuits_indices, current_value)

        for i, biscuit in enumerate(biscuits):
            new_position = current_position + biscuit.length
            new_biscuits_indices = biscuits_indices + [i]
            new_value = current_value + biscuit.value
            if new_position <= dough_roll.length and not overlaps_with_defects(current_position, biscuit, dough_roll.defects):
                # Prune paths that won't lead to a better solution
                if new_value + heuristic(biscuits, dough_roll.length - new_position) <= best_solution[1]:
                    continue
                stack.append((new_position, new_biscuits_indices, new_value))

    solution = [biscuits[i] for i in best_solution[0]]
    return solution, best_solution[1]





from collections import deque

def bfs_search(dough_roll, biscuits):
    print("Debut BFS")
    # Sort biscuits by the value-to-length ratio for efficient coverage
    biscuits.sort(key=lambda b: b.value / b.length, reverse=True)
    
    queue = deque([(0, [], 0)])  # (current position, indices of biscuits used, total value)
    best_solution = ([], 0)

    while queue:
        current_position, biscuits_indices, current_value = queue.popleft()

        if current_value > best_solution[1] and not is_end_of_roll(current_position, dough_roll):
            best_solution = (biscuits_indices, current_value)

        for i, biscuit in enumerate(biscuits):
            new_position = current_position + biscuit.length
            new_biscuits_indices = biscuits_indices + [i]
            new_value = current_value + biscuit.value
            if new_position <= dough_roll.length and not overlaps_with_defects(current_position, biscuit, dough_roll.defects):
                queue.append((new_position, new_biscuits_indices, new_value))

    solution = [biscuits[i] for i in best_solution[0]]
    return solution, best_solution[1]


# Adjust the heuristic function to use bitmask
def heuristiccc(biscuits, remaining_length):
    
    max_value_per_length = max(biscuit.value / biscuit.length for biscuit in biscuits)
    return max_value_per_length * remaining_length

# Adjust the overlaps_with_defects function to use bitmask



def hill_climbing_search(dough_roll, biscuits):
    # Sort biscuits by the value-to-length ratio for efficient coverage
    biscuits.sort(key=lambda b: b.value / b.length, reverse=True)

    # Start with a random solution
    current_solution = random.sample(biscuits, k=len(biscuits))
    current_value = sum(b.value for b in current_solution)

    while True:
        neighbors = get_neighbors(current_solution, biscuits, dough_roll)
        neighbor_values = [sum(b.value for b in neighbor) for neighbor in neighbors]

        # Find the best neighbor
        best_neighbor = max(zip(neighbors, neighbor_values), key=lambda nv: nv[1])

        # If the best neighbor is better than the current solution, move to the neighbor
        if best_neighbor[1] > current_value:
            current_solution, current_value = best_neighbor
        else:
            # No improvement found
            break

    return current_solution, current_value

        
def evaluate_algorithms(dough_roll, biscuits):
    algorithms = {
        #'A*': lambda: a_star_search(dough_roll, biscuits),
        'Breadth-First Search (BFS)': lambda: bfs_search(dough_roll, biscuits),
        'Hill Climbing Search': lambda: hill_climbing_search(dough_roll, biscuits),
        'Depth-First Search (DFS)': lambda: dfs_search(dough_roll, biscuits)
    }

    best_overall_value = 0
    best_overall_solution = None
    best_algorithm_name = ""
    print("Boucle de chaque items")
    for name, algorithm_func in algorithms.items():
        try:
            solution, value = algorithm_func()  # Call the search algorithm here
            print(f"Value of algorithm {name}: {value}")

            if value > best_overall_value:
                best_overall_value = value
                best_overall_solution = solution
                best_algorithm_name = name
        except ValueError as e:
            print(f"Error in algorithm {name}: {e}")

    if best_overall_solution:
        print(f"The best overall solution was found by {best_algorithm_name} with a value of {best_overall_value}:")
        print_solution(best_overall_solution)
        print_dough_visualization(best_overall_solution, dough_roll.defects)
    else:
        print("No valid solution was found by any algorithm.")


if __name__ == "__main__":
    print("Debut code")
    dough_roll, biscuits = optimize_biscuit_placementt('defects.csv')
    print("Debut d'evaluate")
    evaluate_algorithms(dough_roll, biscuits)
    solution, defects = optimize_biscuit_placement('defects.csv')
    print_solution(solution)
    print_dough_visualization(solution, defects)
