import pandas as pd
import heapq
import random
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define the heuristic function for use in search algorithms
def heuristic(biscuits, remaining_length):
    return max(biscuit.value / biscuit.length for biscuit in biscuits) * remaining_length


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
        self.position = 0

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
            defect_type = defect.defect_class
            if defect_type in defect_thresholds and defect_thresholds[defect_type] > 0:
                defect_thresholds[defect_type] -= 1
            else:
                return True  # Retourne True si le seuil de défaut est dépassé
    return False


# Define the search function for finding the best arrangement of biscuits
def search(dough_roll, biscuits):
    # Start state has no biscuits placed and no value
    start_state = (0, [], 0)  # (current position, biscuit indices, current value)
    frontier = [(-0, start_state)]  # Priority queue for states with negative priority for max-heap
    explored = set()
    best_solution = ([], 0)  # (biscuit indices, total value)

    while frontier:
        if current_value + heuristic(biscuits, remaining_length) <= best_solution[1]:
            continue

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

def update_biscuit_positions(solution):
        position = 0
        for biscuit in solution:
            biscuit.position = position
            position += biscuit.length
# Optimize the placement of biscuits based on defects in the dough roll
def optimize_biscuit_placement(csv_filepath):
    defects_df = load_defects(csv_filepath)
    global defects
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
    

    
    if validate_solution(solution_hill_climbing[0], dough_roll, defects):
        print("La solution est valide.")
        update_biscuit_positions(solution_hill_climbing[0])
        print_solution(solution_hill_climbing[0], "Hill Climbing")
        #print_dough_visualization(solution_hill_climbing[0], defects)
    else:
        print("La solution n'est pas valide.")
        print_solution(solution_hill_climbing[0], "Hill Climbing")
        
    if validate_solution(constrained_solution, dough_roll, defects):
        print("La solution est valide.")
        update_biscuit_positions(constrained_solution)
        print_solution(constrained_solution, "Counstrainte solution")
        
        #print_dough_visualization(constrained_solution, defects)
    else:
        print("La solution n'est pas valide.")
        print_solution(constrained_solution, "Counstrainte solution")
    
    return solution_hill_climbing, defects


def validate_solution(solution, dough_roll, defects):
    current_position = 0
    is_valid = True

    for biscuit in solution:
        defect_thresholds = biscuit.defect_thresholds.copy()

        # Parcourir chaque défaut et vérifier s'il chevauche le biscuit actuel
        for defect in defects:
            if current_position <= defect.position < current_position + biscuit.length:
                if defect.defect_class in defect_thresholds and defect_thresholds[defect.defect_class] > 0:
                    defect_thresholds[defect.defect_class] -= 1
                else:
                    # Défaut non autorisé ou limite dépassée
                    print(f"Biscuit à la position {current_position} dépasse le seuil de défauts pour '{defect.defect_class}'.")
                    is_valid = False

        current_position += biscuit.length

    return is_valid

        
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()  # Flush the buffer
    
def constraint_based_search(dough_roll, defects, biscuits):
    # Fonction pour générer une solution aléatoire initiale
    sorted_biscuits = sorted(biscuits, key=lambda b: b.value / b.length, reverse=True)

    def generate_initial_solution():
        solution = []
        total_length = 0

        while total_length < dough_roll.length:
            eligible_biscuits = [
                b for b in sorted_biscuits
                if b.length <= (dough_roll.length - total_length) and not overlaps_with_defects(total_length, b, dough_roll.defects)
            ]

            if not eligible_biscuits:
                # Si aucun biscuit n'est éligible, cela peut être dû à un défaut bloquant la position actuelle.
                # Augmentez total_length pour passer le défaut si c'est le cas.
                total_length += 1
                continue
            
            biscuit = eligible_biscuits[0]
            biscuit.position = total_length  # Mise à jour de la position du biscuit
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
        max_attempts = 100
        for _ in range(max_attempts):
            neighbor = solution[:]
            idx_to_replace = random.randrange(len(neighbor))
            # Sélectionner un biscuit à remplacer

            # Filtrer les biscuits éligibles pour le remplacement
            remaining_length = dough_roll.length - sum(b.length for i, b in enumerate(neighbor) if i != idx_to_replace)
            eligible_biscuits = [b for b in sorted_biscuits if b.length <= remaining_length]
            if eligible_biscuits:
            # Remplacer par un biscuit ayant un meilleur rapport valeur/longueur
                chosen_biscuit = max(eligible_biscuits, key=lambda b: b.value / b.length) 
                # S'assurer que le biscuit choisi ne viole pas les seuils de défauts.
                temp_position = sum(b.length for i, b in enumerate(neighbor) if i < idx_to_replace)
                if not overlaps_with_defects(temp_position, chosen_biscuit, dough_roll.defects):
                    chosen_biscuit.position = temp_position  # Mise à jour de la position
                    neighbor[idx_to_replace] = chosen_biscuit
                    if respects_constraints(neighbor):
                        return neighbor
            

        return solution
 # Retourne la solution actuelle si aucun voisin valide n'est trouvé

    # Initialisation d'une solution aléatoire
    print("Debut de generation")
    current_solution = generate_initial_solution()
    print("Debut de calculate value")
    current_value = calculate_value(current_solution)
    print("Debut boucle")
    max_steps_without_improvement = 100

    # Boucle de recherche avec barre de progression
    for step in range(max_steps_without_improvement):
        # Mise à jour de la barre de progression
        print_progress_bar(iteration=step, total=max_steps_without_improvement, prefix='Progress:', suffix='Complete', length=50)

        neighbor = get_constrained_random_neighbor(current_solution)
        neighbor_value = calculate_value(neighbor)

        if neighbor_value > current_value:
            current_solution, current_value = neighbor, neighbor_value

    # Affichage final de la barre de progression
    print_progress_bar(iteration=max_steps_without_improvement, total=max_steps_without_improvement, prefix='Progress:', suffix='Complete', length=50)
    
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
def calculate_value(solution):
        return sum(biscuit.value for biscuit in solution)
def respects_constraints(solution, dough_roll):
    total_length = sum(biscuit.length for biscuit in solution)
    if total_length > dough_roll.length:
        return False

    for i, biscuit in enumerate(solution):
        position = sum(b.length for b in solution[:i])
        if overlaps_with_defects(position, biscuit, defects):
            return False

    return True
# Main entry point for running the optimization
def hill_climbing_search(dough_roll, biscuits):
    # Créer une solution initiale qui respecte les contraintes de défauts
    sorted_biscuits = sorted(biscuits, key=lambda b: b.value / b.length, reverse=True)

    # Créer une solution initiale en respectant les contraintes de défauts
    current_solution = []
    total_length = 0

    while total_length < dough_roll.length:
        eligible_biscuits = [
        b for b in sorted_biscuits 
        if total_length + b.length <= dough_roll.length and not overlaps_with_defects(total_length, b, dough_roll.defects)
        ]

        if not eligible_biscuits:
            total_length += 1  # Incrémentez légèrement la longueur totale pour contourner les défauts
            continue

        # Choisissez le biscuit qui maximise l'utilisation de l'espace restant
        biscuit = min(eligible_biscuits, key=lambda b: dough_roll.length - (total_length + b.length))
        biscuit.position = round(total_length, 14)
        current_solution.append(biscuit)
        total_length += biscuit.length

    current_value = calculate_value(current_solution)



    # Function to calculate the value of a solution
    
    
        # Function to create a neighbor solution
    # Fonction pour créer un voisin qui respecte les contraintes
    def get_random_neighbor(solution):
        for _ in range(100):
            neighbor = solution[:]
            idx_to_change = random.randrange(len(solution))
            remaining_length = dough_roll.length - sum(b.length for i, b in enumerate(neighbor) if i != idx_to_change)

            eligible_biscuits = [
                b for b in sorted_biscuits 
                if b.length <= remaining_length and not overlaps_with_defects(sum(b.length for i, b in enumerate(neighbor) if i < idx_to_change), b, dough_roll.defects)
            ]

            if eligible_biscuits:
                # Choisissez un biscuit qui maximise l'utilisation de l'espace restant
                best_biscuit = min(eligible_biscuits, key=lambda b: remaining_length - b.length)
                best_biscuit.position = sum(b.length for i, b in enumerate(neighbor) if i < idx_to_change)
                neighbor[idx_to_change] = best_biscuit

                if respects_constraints(neighbor, dough_roll):
                    return neighbor

        return solution



    # Function to print the progress bar
    def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
        percent = ("{0:.1f}").format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + '-' * (length - filled_length)
        sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
        sys.stdout.flush()

    # Perform hill climbing
    max_steps_without_improvement = 200  # Augmentez si nécessaire

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
    return current_solution, calculate_value(current_solution)

# Correcting the print_solution function to handle a list of Biscuit objects
def print_solution(solution,name):
    print("L'algorithme de recherche est le ",name)
    if solution:
        print("A combination of biscuits has been found:")
        total_length = 0
        total_value = 0
        for i, biscuit in enumerate(solution):  # solution should be a list of Biscuit objects
            #print(f"Biscuit {i}: Length {biscuit.length}, Value {biscuit.value}, Defect Thresholds {biscuit.defect_thresholds}")
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