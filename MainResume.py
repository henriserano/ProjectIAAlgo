
import pandas as pd
import heapq
import random
from collections import deque
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
# Define the heuristic function for use in search algorithms
def heuristic(biscuits, remaining_length):
    max_value_per_length = max(biscuit.value / biscuit.length for biscuit in biscuits)
    return max_value_per_length * remaining_length

# Load defects from a CSV file into a DataFrame
def load_defects(csv_filepath):
    return pd.read_csv(csv_filepath)

# Define the Biscuit class
class Biscuit:
    def __init__(self, length, value, defect_thresholds):
        self.length = length
        self.value = value
        self.defect_thresholds = defect_thresholds
        self.children_indices = []

    def add_child(self, child_index):
        self.children_indices.append(child_index)

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value
    def __str__(self):
        return 'La valeur de mon biscuit est : ',self.length," ma value est ",self.value," mes défauts sont ",self.defect_thresholds

# Define the Defect class
class Defect:
    def __init__(self, position, defect_class):
        self.position = position
        self.defect_class = defect_class
    def __str__(self):
        return "Type : "+str(self.defect_class)+" position : "+str(self.position)

# Define the DoughRoll class
class DoughRoll:
    def __init__(self, length, defects):
        self.length = length
        self.defects = defects

# Check if a biscuit's position overlaps with any defects
def overlaps_with_defects(position, biscuit, defects):
    defect_thresholds = biscuit.defect_thresholds.copy()
    for defect in defects:
        if position <= defect.position < position + biscuit.length:
            if defect_thresholds.get(defect.defect_class, 0) <= 0:
                return True
            defect_thresholds[defect.defect_class] -= 1
    return False

# Define the search function for finding the best arrangement of biscuits
def search(dough_roll, biscuits):
    # Start state has no biscuits placed and no value
    start_state = (0, [], 0)  # (current position, biscuit indices, current value)
    frontier = [(-0, start_state)]  # Priority queue for states with negative priority for max-heap
    explored = set()
    best_solution = ([], 0)  # (biscuit indices, total value)

    while frontier:
        _, current_state = heapq.heappop(frontier)
        current_position, biscuits_indices, current_value = current_state

        # Update the best solution if the current value is higher
        if current_value > best_solution[1]:
            best_solution = (biscuits_indices, current_value)

        # Skip already explored states
        hashable_state = (current_position, tuple(biscuits_indices))
        if hashable_state in explored:
            continue
        explored.add(hashable_state)

        # Remaining dough length
        remaining_length = dough_roll.length - current_position

        # Try placing each biscuit at the current position
        for i, biscuit in enumerate(biscuits):
            new_position = current_position + biscuit.length
            if new_position <= dough_roll.length and not overlaps_with_defects(current_position, biscuit, dough_roll.defects):
                new_biscuits_indices = biscuits_indices + [i]
                new_value = current_value + biscuit.value
                # Estimate priority based on the new value and heuristic
                priority = -new_value - heuristic(biscuits, remaining_length)
                heapq.heappush(frontier, (priority, (new_position, new_biscuits_indices, new_value)))

    # Construct the final solution using the indices
    return [biscuits[i] for i in best_solution[0]]


# Optimize the placement of biscuits based on defects in the dough roll
def optimize_biscuit_placement(csv_filepath):
    defects_df = load_defects(csv_filepath)
    defects = [Defect(float(row['x']), row['class']) for _, row in defects_df.iterrows()]
    dough_roll = DoughRoll(500, defects)

    # Define the actual list of Biscuit objects with their lengths, values, and defect thresholds.
    biscuits = [
        Biscuit(4, 6, {'a': 4, 'b': 2, 'c': 3}),
        Biscuit(8, 12, {'a': 5, 'b': 4, 'c': 4}),
        Biscuit(2, 1, {'a': 1, 'b': 2, 'c': 1}),
        Biscuit(5, 8, {'a': 2, 'b': 3, 'c': 2}),
    ]

    for i, biscuit in enumerate(biscuits):
        for j, next_biscuit in enumerate(biscuits):
            if not overlaps_with_defects(biscuit.length, next_biscuit, dough_roll.defects):
                biscuit.add_child(j)

    solution_hill_climbing = hill_climbing_search(dough_roll, biscuits)
    constrained_solution = constraint_based_search( dough_roll, defects, biscuits)
    print_solution(constrained_solution)
    return solution_hill_climbing, defects

        
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()  # Flush the buffer
    
def constraint_based_search(dough_roll, defects, biscuits):
    # Fonction pour générer une solution aléatoire initiale
    def generate_initial_solution():
        solution = []
        total_length = 0
        while total_length < dough_roll.length:
            biscuit = random.choice(biscuits)
            if total_length + biscuit.length <= dough_roll.length:
                solution.append(biscuit)
                total_length += biscuit.length
        return solution

    # Fonction pour calculer la valeur totale d'une solution
    def calculate_value(solution):
        return sum(biscuit.value for biscuit in solution)

    # Fonction pour vérifier si la solution respecte les contraintes
    def respects_constraints(solution):
        total_length = sum(biscuit.length for biscuit in solution)
        if total_length > dough_roll.length:
            return False

        for i, biscuit in enumerate(solution):
            position = sum(b.length for b in solution[:i])
            if overlaps_with_defects(position, biscuit, dough_roll.defects):
                return False

        return True

    # Fonction pour générer un voisin qui respecte les contraintes
    def get_constrained_random_neighbor(solution):
        max_attempts = 100  # Limite le nombre d'essais pour trouver un voisin valide
        for _ in range(max_attempts):
            neighbor = solution[:]
            idx_to_change = random.randrange(len(solution))
            neighbor[idx_to_change] = random.choice(biscuits)
            
            if respects_constraints(neighbor):
                return neighbor

        return solution  # Retourne la solution actuelle si aucun voisin valide n'est trouvé

    # Initialisation d'une solution aléatoire
    current_solution = generate_initial_solution()
    current_value = calculate_value(current_solution)

    max_steps_without_improvement = 100

    # Boucle de recherche
    for step in range(max_steps_without_improvement):
        neighbor = get_constrained_random_neighbor(current_solution)
        neighbor_value = calculate_value(neighbor)

        if neighbor_value > current_value:
            current_solution, current_value = neighbor, neighbor_value

    return current_solution



def get_color(biscuit_type):
    color_dict = {
        'Type1': 'blue',
        'Type2': 'green',
        'Type3': 'yellow',
    }
    
    return color_dict.get(biscuit_type, 'grey')

def print_dough_visualization(biscuits_sequence, defects):
    # Trouvez les valeurs min et max pour la normalisation des couleurs
    global min_value, max_value
    min_value = min(biscuit.value for biscuit in biscuits_sequence)
    max_value = max(biscuit.value for biscuit in biscuits_sequence)

    # Initialisation de la visualisation
    fig, ax = plt.subplots()
    current_position = 0
    bar_height = 1

    # Affichage des biscuits
    for biscuit in biscuits_sequence:
        # Utilisez la valeur du biscuit pour obtenir la couleur
        biscuit_color = get_color(biscuit.value)
        ax.add_patch(patches.Rectangle((current_position, 0), biscuit.length, bar_height, facecolor=biscuit_color))
        current_position += biscuit.length

    # Affichage des défauts
    for defect in defects:
        ax.add_patch(patches.Rectangle((defect.position, -bar_height), 1, bar_height, facecolor='red'))

    # Ajustement des axes
    ax.set_xlim(0, 500)  # Remplacez 500 par la longueur totale de la pâte si nécessaire
    ax.set_ylim(-bar_height, bar_height)
    ax.axis('off')
    plt.show()

# Main entry point for running the optimization

def hill_climbing_search(dough_roll, biscuits):
    # Create an initial random solution within the length of the dough roll
    current_solution = []
    total_length = 0
    sorted_biscuits = sorted(biscuits, key=lambda x: x.length)
    
    while total_length < dough_roll.length:
        # Filtrer les biscuits qui peuvent s'adapter à l'espace restant
        fitting_biscuits = [biscuit for biscuit in sorted_biscuits if total_length + biscuit.length <= dough_roll.length]
        
        # Si aucun biscuit ne s'adapte à l'espace restant, sortez de la boucle
        if not fitting_biscuits:
            break
        
        # Choisir un biscuit au hasard parmi ceux qui s'adaptent
        biscuit = random.choice(fitting_biscuits)
        current_solution.append(biscuit)
        total_length += biscuit.length

    current_value = sum(biscuit.value for biscuit in current_solution)

    # Function to calculate the total length of biscuits in the solution
    def calculate_length(solution):
        return sum(biscuit.length for biscuit in solution)

    # Function to calculate the value of a solution
    def calculate_value(solution):
        return sum(biscuit.value for biscuit in solution)

    # Function to create a neighbor solution
    def get_random_neighbor(solution):
        neighbor = solution[:]
        idx_to_change = random.randrange(len(solution))
        neighbor[idx_to_change] = random.choice(biscuits)
        
        # Make sure the total length does not exceed the length of the dough roll
        while calculate_length(neighbor) > dough_roll.length:
            neighbor.pop(random.randrange(len(neighbor)))
        
        return neighbor

    # Function to print the progress bar
    def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
        sys.stdout.flush()

    # Perform hill climbing
    max_steps_without_improvement = 100
    steps_without_improvement = 0
    for step in range(max_steps_without_improvement):
        # Print the progress bar
        print_progress_bar(step, max_steps_without_improvement, prefix='Progress:', suffix='Complete', length=50)

        neighbor = get_random_neighbor(current_solution)
        neighbor_value = calculate_value(neighbor)

        # If the neighbor solution is better, move to it
        if neighbor_value > current_value:
            current_solution, current_value = neighbor, neighbor_value
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

    # Print the completed progress bar
    print_progress_bar(max_steps_without_improvement, max_steps_without_improvement, prefix='Progress:', suffix='Complete', length=50)
    print()  # New line at the end

    # Return the best solution found
    return current_solution, current_value

# Correcting the print_solution function to handle a list of Biscuit objects
def print_solution(solution):
    if solution:
        print("A combination of biscuits has been found:")
        total_length = 0
        total_value = 0
        for i, biscuit in enumerate(solution):  # solution should be a list of Biscuit objects
            print(f"Biscuit {i}: Length {biscuit.length}, Value {biscuit.value}, Defect Thresholds {biscuit.defect_thresholds}")
            total_length += biscuit.length
            total_value += biscuit.value
        print(f"Total length used: {total_length}")
        print(f"Total value: {total_value}")
    else:
        print("No combination of biscuits was found.")



# Main entry point for running the optimization
if __name__ == "__main__":
    solution, defects = optimize_biscuit_placement('defects.csv')
    
    # Run hill climbing search to improve the solution
    #for i in defects:
    #    print(i)
    #print_solution(solution[0])
    #print_dough_visualization(solution[0], defects)
=======
import pandas as pd
import heapq
import random
from collections import deque
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
# Define the heuristic function for use in search algorithms
def heuristic(biscuits, remaining_length):
    max_value_per_length = max(biscuit.value / biscuit.length for biscuit in biscuits)
    return max_value_per_length * remaining_length

# Load defects from a CSV file into a DataFrame
def load_defects(csv_filepath):
    return pd.read_csv(csv_filepath)

# Define the Biscuit class
class Biscuit:
    def __init__(self, length, value, defect_thresholds):
        self.length = length
        self.value = value
        self.defect_thresholds = defect_thresholds
        self.children_indices = []

    def add_child(self, child_index):
        self.children_indices.append(child_index)

    def __lt__(self, other):
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value
    def __str__(self):
        return 'La valeur de mon biscuit est : ',self.length," ma value est ",self.value," mes défauts sont ",self.defect_thresholds

# Define the Defect class
class Defect:
    def __init__(self, position, defect_class):
        self.position = position
        self.defect_class = defect_class
    def __str__(self):
        return "Type : "+str(self.defect_class)+" position : "+str(self.position)

# Define the DoughRoll class
class DoughRoll:
    def __init__(self, length, defects):
        self.length = length
        self.defects = defects

# Check if a biscuit's position overlaps with any defects
def overlaps_with_defects(position, biscuit, defects):
    defect_thresholds = biscuit.defect_thresholds.copy()
    for defect in defects:
        if position <= defect.position < position + biscuit.length:
            if defect_thresholds.get(defect.defect_class, 0) <= 0:
                return True
            defect_thresholds[defect.defect_class] -= 1
    return False

# Define the search function for finding the best arrangement of biscuits
def search(dough_roll, biscuits):
    # Start state has no biscuits placed and no value
    start_state = (0, [], 0)  # (current position, biscuit indices, current value)
    frontier = [(-0, start_state)]  # Priority queue for states with negative priority for max-heap
    explored = set()
    best_solution = ([], 0)  # (biscuit indices, total value)

    while frontier:
        _, current_state = heapq.heappop(frontier)
        current_position, biscuits_indices, current_value = current_state

        # Update the best solution if the current value is higher
        if current_value > best_solution[1]:
            best_solution = (biscuits_indices, current_value)

        # Skip already explored states
        hashable_state = (current_position, tuple(biscuits_indices))
        if hashable_state in explored:
            continue
        explored.add(hashable_state)

        # Remaining dough length
        remaining_length = dough_roll.length - current_position

        # Try placing each biscuit at the current position
        for i, biscuit in enumerate(biscuits):
            new_position = current_position + biscuit.length
            if new_position <= dough_roll.length and not overlaps_with_defects(current_position, biscuit, dough_roll.defects):
                new_biscuits_indices = biscuits_indices + [i]
                new_value = current_value + biscuit.value
                # Estimate priority based on the new value and heuristic
                priority = -new_value - heuristic(biscuits, remaining_length)
                heapq.heappush(frontier, (priority, (new_position, new_biscuits_indices, new_value)))

    # Construct the final solution using the indices
    return [biscuits[i] for i in best_solution[0]]


# Optimize the placement of biscuits based on defects in the dough roll
def optimize_biscuit_placement(csv_filepath):
    defects_df = load_defects(csv_filepath)
    defects = [Defect(float(row['x']), row['class']) for _, row in defects_df.iterrows()]
    dough_roll = DoughRoll(500, defects)

    # Define the actual list of Biscuit objects with their lengths, values, and defect thresholds.
    biscuits = [
        Biscuit(4, 6, {'a': 4, 'b': 2, 'c': 3}),
        Biscuit(8, 12, {'a': 5, 'b': 4, 'c': 4}),
        Biscuit(2, 1, {'a': 1, 'b': 2, 'c': 1}),
        Biscuit(5, 8, {'a': 2, 'b': 3, 'c': 2}),
        # Add more Biscuit instances as needed
    ]

    for i, biscuit in enumerate(biscuits):
        for j, next_biscuit in enumerate(biscuits):
            if not overlaps_with_defects(biscuit.length, next_biscuit, dough_roll.defects):
                biscuit.add_child(j)

    solution = hill_climbing_search(dough_roll, biscuits)
    return solution, defects

        
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()  # Flush the buffer
    

def get_color(biscuit_type):
    color_dict = {
        'Type1': 'blue',
        'Type2': 'green',
        'Type3': 'yellow',
    }
    
    return color_dict.get(biscuit_type, 'grey')

def print_dough_visualization(biscuits_sequence, defects):
    # Trouvez les valeurs min et max pour la normalisation des couleurs
    global min_value, max_value
    min_value = min(biscuit.value for biscuit in biscuits_sequence)
    max_value = max(biscuit.value for biscuit in biscuits_sequence)

    # Initialisation de la visualisation
    fig, ax = plt.subplots()
    current_position = 0
    bar_height = 1

    # Affichage des biscuits
    for biscuit in biscuits_sequence:
        # Utilisez la valeur du biscuit pour obtenir la couleur
        biscuit_color = get_color(biscuit.value)
        ax.add_patch(patches.Rectangle((current_position, 0), biscuit.length, bar_height, facecolor=biscuit_color))
        current_position += biscuit.length

    # Affichage des défauts
    for defect in defects:
        ax.add_patch(patches.Rectangle((defect.position, -bar_height), 1, bar_height, facecolor='red'))

    # Ajustement des axes
    ax.set_xlim(0, 500)  # Remplacez 500 par la longueur totale de la pâte si nécessaire
    ax.set_ylim(-bar_height, bar_height)
    ax.axis('off')
    plt.show()

# Main entry point for running the optimization

def hill_climbing_search(dough_roll, biscuits):
    # Create an initial random solution within the length of the dough roll
    current_solution = []
    total_length = 0
    sorted_biscuits = sorted(biscuits, key=lambda x: x.length)
    
    while total_length < dough_roll.length:
        # Filtrer les biscuits qui peuvent s'adapter à l'espace restant
        fitting_biscuits = [biscuit for biscuit in sorted_biscuits if total_length + biscuit.length <= dough_roll.length]
        
        # Si aucun biscuit ne s'adapte à l'espace restant, sortez de la boucle
        if not fitting_biscuits:
            break
        
        # Choisir un biscuit au hasard parmi ceux qui s'adaptent
        biscuit = random.choice(fitting_biscuits)
        current_solution.append(biscuit)
        total_length += biscuit.length

    current_value = sum(biscuit.value for biscuit in current_solution)

    # Function to calculate the total length of biscuits in the solution
    def calculate_length(solution):
        return sum(biscuit.length for biscuit in solution)

    # Function to calculate the value of a solution
    def calculate_value(solution):
        return sum(biscuit.value for biscuit in solution)

    # Function to create a neighbor solution
    def get_random_neighbor(solution):
        neighbor = solution[:]
        idx_to_change = random.randrange(len(solution))
        neighbor[idx_to_change] = random.choice(biscuits)
        
        # Make sure the total length does not exceed the length of the dough roll
        while calculate_length(neighbor) > dough_roll.length:
            neighbor.pop(random.randrange(len(neighbor)))
        
        return neighbor

    # Function to print the progress bar
    def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
        sys.stdout.flush()

    # Perform hill climbing
    max_steps_without_improvement = 100
    steps_without_improvement = 0
    for step in range(max_steps_without_improvement):
        # Print the progress bar
        print_progress_bar(step, max_steps_without_improvement, prefix='Progress:', suffix='Complete', length=50)

        neighbor = get_random_neighbor(current_solution)
        neighbor_value = calculate_value(neighbor)

        # If the neighbor solution is better, move to it
        if neighbor_value > current_value:
            current_solution, current_value = neighbor, neighbor_value
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1

    # Print the completed progress bar
    print_progress_bar(max_steps_without_improvement, max_steps_without_improvement, prefix='Progress:', suffix='Complete', length=50)
    print()  # New line at the end

    # Return the best solution found
    return current_solution, current_value

# Correcting the print_solution function to handle a list of Biscuit objects
def print_solution(solution):
    if solution:
        print("A combination of biscuits has been found:")
        total_length = 0
        total_value = 0
        for i, biscuit in enumerate(solution):  # solution should be a list of Biscuit objects
            print(f"Biscuit {i}: Length {biscuit.length}, Value {biscuit.value}, Defect Thresholds {biscuit.defect_thresholds}")
            total_length += biscuit.length
            total_value += biscuit.value
        print(f"Total length used: {total_length}")
        print(f"Total value: {total_value}")
    else:
        print("No combination of biscuits was found.")



# Main entry point for running the optimization
if __name__ == "__main__":
    solution, defects = optimize_biscuit_placement('defects.csv')
    # Run hill climbing search to improve the solution
    for i in defects:
        print(i)
    print_solution(solution[0])
    print_dough_visualization(solution[0], defects)
>>>>>>> 2382f8b71541c72ac238fcedd0c3267d6c2d12c4
