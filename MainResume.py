import pandas as pd
import random
import sys
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time


global precision
#If we want to have a perfect solution we will need a precision = 1e-14 but that will take approximatily 24 hours to run
precision = 1
global myiteration
myiteration = 50
# Define the heuristic function for use in search algorithms
def heuristic(biscuits, remaining_length):
    """This function is used to calculate the heuristic value of a state in the search space.

    Args:
        biscuits (_type_): the list of biscuits
        remaining_length (_type_): the remaining length of the dough roll

    Returns:
        _type_: the heuristic value of the state
    """
    return max(biscuit.value / biscuit.length for biscuit in biscuits) * remaining_length


# Load defects from a CSV file into a DataFrame
def load_defects(csv_filepath):
    """This function is used to load the defects from a CSV file into a DataFrame

    Args:
        csv_filepath (_type_): the path of the CSV file

    Returns:
        _type_: the DataFrame containing the defects
    """
    return pd.read_csv(csv_filepath)

# Define the Biscuit class
class Biscuit:
    """This class is used to define a biscuit
    """
    def __init__(self, length, value, defect_thresholds):
        """This function is used to initialize a biscuit

        Args:
            length (_type_): the length of the biscuit
            value (_type_): the value of the biscuit
            defect_thresholds (_type_): the defect thresholds of the biscuit
        """
        self.length = length
        self.value = value
        self.defect_thresholds = defect_thresholds
        self.children_indices = []
        self.position = 0

    def add_child(self, child_index):
        self.children_indices.append(child_index)

    def __lt__(self, other):
        """This function is used to compare two biscuits

        Args:
            other (_type_): the other biscuit

        Returns:
            _type_: True if the first biscuit is less than the other biscuit, False otherwise
        """
        return self.value < other.value

    def __eq__(self, other):
        return self.value == other.value
    def __str__(self):
        return f"La length de mon biscuit est : {self.length}, ma value est {self.value}, mes défauts sont {self.defect_thresholds}"

# Define the Defect class
class Defect:
    """This class is used to define a defect
    """
    def __init__(self, position, defect_class):
        """This function is used to initialize a defect

        Args:
            position (_type_): the position of the defect
            defect_class (_type_): the class of the defect
        """
        self.position = position
        self.defect_class = defect_class
        
    def __str__(self):
        """This function is used to print a defect

        Returns:
            _type_: Type : {self.defect_class} position : {self.position}
        """
        return f"Type : {self.defect_class} position : {self.position}"

# Define the DoughRoll class
class DoughRoll:
    def __init__(self, length, defects):
        """This function is used to initialize a dough roll

        Args:
            length (_type_): the length of the dough roll
            defects (_type_): the defects of the dough roll
        """
        self.length = length
        self.defects = defects

# Check if a biscuit's position overlaps with any defects
def overlaps_with_defects(position, biscuit, defects):
    """This function is used to check if a biscuit's position overlaps with any defects

    Args:
        position (float): the position of the biscuit
        biscuit (Biscuit): the biscuit
        defects (Defect): the defects

    Returns:
        _type_: True if the biscuit's position overlaps with any defects, False otherwise
    """
    defect_thresholds = biscuit.defect_thresholds.copy()
    for defect in defects:
        if position <= defect.position < position + biscuit.length:
            defect_type = defect.defect_class
            if defect_thresholds.get(defect_type, 0) > 0:
                defect_thresholds[defect_type] -= 1
            else:
                return True
    return False

def plot_defects_on_1d_space(length=500, initial_view=20, defects_file='defects.csv', biscuits=None):
    def update(val):
        ax.set_xlim(slider.val, slider.val + length_slider.val)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    x = np.linspace(0, length, length)
    y = np.zeros_like(x)
    ax.plot(x, y, color='gray', label='Roll of Dough')

    unique_ratios = set()  # Collect unique biscuit ratios

    if biscuits is not None:
        for biscuit in biscuits:
            x_position = biscuit.position
            biscuit_width = biscuit.length
            biscuit_ratio = (biscuit.value / biscuit.length) / 2
            unique_ratios.add(biscuit_ratio)  # Collect unique biscuit ratios
            biscuit_color = plt.cm.Oranges(biscuit_ratio)  # Use Oranges colormap for shades of orange
            biscuit_ellipse = patches.Ellipse((x_position + biscuit_width / 2, 0), biscuit_width, 1, edgecolor=biscuit_color, facecolor=biscuit_color, label=f'Biscuit {biscuit.value}')
            ax.add_patch(biscuit_ellipse)
            
            # Add vertical dotted line between biscuits
            ax.axvline(x_position, color='#CCCCCC', linestyle=':')
            ax.axvline(x_position + biscuit_width, color='#CCCCCC', linestyle=':')
        unique_ratios = sorted(unique_ratios)

    # Read defects from the CSV file
    defects_data = pd.read_csv(defects_file)
    
    # Plot defects with different colors based on their types
    colors = {'a': '#009900', 'b': '#00AAAA', 'c': '#0066AA'}
    for _, row in defects_data.iterrows():
        x_position = row['x']
        defect_type = row['class']
        defect_y = {'a': 0.15, 'b': 0.1, 'c': 0.05}[defect_type]
        ax.scatter(x_position, defect_y, color=colors[defect_type], marker='x')    

    # Add legend with relevant information
    legend_labels = {'a': 'Defect Type A', 'b': 'Defect Type B', 'c': 'Defect Type C'}
    legend_handles = [plt.Line2D([0], [0], marker='x', color=colors[type_], markerfacecolor=colors[type_], markersize=8, label=legend_labels[type_]) for type_ in colors]

    # Add legend entries for unique biscuit ratios
    for biscuit_ratio in unique_ratios:
        biscuit_color = plt.cm.Oranges(biscuit_ratio)
        biscuit_type = '4' if biscuit_ratio == 0.8 else '1 & 2' if biscuit_ratio == 0.75 else '3'
        legend_handles.append(patches.Ellipse((0, 0), 1, 1, color=biscuit_color, label=f'Biscuit Type: {biscuit_type} with Ratio: {biscuit_ratio*2}'))

    legend_handles.append(plt.Line2D([0], [0], color='gray', label='Roll of Dough'))
    ax.legend(handles=legend_handles)

    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_length_slider = plt.axes([0.2, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    # Set logarithmic scale for the zoom bar
    length_slider_ticks = [1, 10, 100, 500]
    length_slider = Slider(ax_length_slider, 'Zoom', 1, length, valinit=initial_view)
    length_slider.valtext.set_text('1')  # Set initial text value

    def update_length_slider(val):
        val = int(val)
        length_slider.valtext.set_text(str(val))
        update(val)

    length_slider.on_changed(update_length_slider)

    # Set logarithmic scale for the length slider
    ax_length_slider.set_xscale("log")
    ax_length_slider.set_xticks(length_slider_ticks)
    ax_length_slider.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Set linear scale for the scroll bar
    slider = Slider(ax_slider, 'Scroll', 0, 500, valinit=0)
    slider.on_changed(update)

    plt.show()


  
# Optimize the placement of biscuits based on defects in the dough roll
def optimize_biscuit_placement(csv_filepath):
    """This function is used to optimize the placement of biscuits based on defects in the dough roll

    Args:
        csv_filepath (string): the path of the CSV file

    Returns:
        _type_: the solution and the defects
    """
    defects_df = load_defects(csv_filepath)
    global defects
    defects = [Defect(float(row['x']), row['class']) for _, row in defects_df.iterrows()]
    dough_roll = DoughRoll(500, defects)

    # Define the actual list of Biscuit objects with their lengths, values, and defect thresholds.
    biscuits = [
        Biscuit(4, 6, {'a': 4, 'b': 2, 'c': 3}), #
        Biscuit(8, 12, {'a': 5, 'b': 4, 'c': 4}),#
        Biscuit(2, 1, {'a': 1, 'b': 2, 'c': 1}), #
        Biscuit(5, 8, {'a': 2, 'b': 3, 'c': 2}), #
    ]

    # Add children to each biscuit based on their lengths
    for i, biscuit in enumerate(biscuits):
        for j, next_biscuit in enumerate(biscuits):
            if not overlaps_with_defects(biscuit.length, next_biscuit, dough_roll.defects):
                biscuit.add_child(j)

    # Run the search algorithm
    solution_hill_climbing = hill_climbing_search(dough_roll, biscuits)
    constrained_solution = constraint_based_search( dough_roll, defects, biscuits)
    
    # Check if the solution is validate and print it
    if validate_solution(solution_hill_climbing[0], dough_roll, defects):
        print("The solution is valid")
        print_solution(solution_hill_climbing[0], "Hill Climbing")
        for i in solution_hill_climbing[0]:
            print(f"Position: {i.position} Length: {i.length} Defect Thresholds: {i.defect_thresholds}")
        plot_defects_on_1d_space(biscuits=solution_hill_climbing[0])
        #print_dough_visualization(solution_hill_climbing[0], defects)
    else:
        print("The solution is not valid.")
        print_solution(solution_hill_climbing[0], "Hill Climbing")
        
        plot_defects_on_1d_space(biscuits=solution_hill_climbing[0])
 
        
    if validate_solution(constrained_solution, dough_roll, defects):
        print("The solution is valid")
        print_solution(constrained_solution, "Counstrainte solution")
        for i in constrained_solution:
            print(f"Position: {i.position} Length: {i.length} Defect Thresholds: {i.defect_thresholds}")
        plot_defects_on_1d_space(biscuits=constrained_solution)
        
    else:
        print("The solution is not valid")
        print_solution(constrained_solution, "Counstrainte solution")
        plot_defects_on_1d_space(biscuits=constrained_solution)
    
    return solution_hill_climbing, defects


def validate_solution(solution, dough_roll, defects):
    """This function is used to validate a solution

    Args:
        solution (tab): the solution
        dough_roll (DougRoll): the dough roll
        defects (Defect): the defects

    Returns:
        _type_: True if the solution is valid, False otherwise
    """
    is_valid = True

    for biscuit in solution:
        defect_thresholds = biscuit.defect_thresholds.copy()

        for defect in defects:
            if biscuit.position <= defect.position < biscuit.position + biscuit.length:
                if defect.defect_class in defect_thresholds and defect_thresholds[defect.defect_class] > 0:
                    defect_thresholds[defect.defect_class] -= 1
                else:
                    print(f"Biscuit à la position {biscuit.position} dépasse le seuil de défauts pour '{defect.defect_class}'.")
                    is_valid = False
                    break  # Arrêter la vérification dès qu'un problème est détecté

        if not is_valid:
            break  # Arrêter la vérification dès qu'un problème est détecté

    return is_valid


        


def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', start_time=None):
    """Call in a loop to create terminal progress bar

    Args:
        iteration (int): _description_
        total (_type_): _description_
        prefix (str, optional): _description_. Defaults to ''.
        suffix (str, optional): _description_. Defaults to ''.
        length (int, optional): _description_. Defaults to 50.
        fill (str, optional): _description_. Defaults to '█'.
        start_time (_type_, optional): _description_. Defaults to None.
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    if start_time:
        elapsed_time = time.time() - start_time
        estimated_total = elapsed_time / iteration * total
        remaining_time = estimated_total - elapsed_time
        suffix = f"{suffix} | {format_time(remaining_time)} remaining"

    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()

def format_time(seconds):
    """ Formats seconds into a human-readable string of HH:MM:SS """
    return time.strftime('%H:%M:%S', time.gmtime(seconds))
    
def constraint_based_search(dough_roll, defects, biscuits):
    print("Starting of the constraint based search")
    # Fonction pour générer une solution aléatoire initiale
    sorted_biscuits = sorted(biscuits, key=lambda b: b.value / b.length, reverse=True)
    
    def generate_initial_solution():
        solution = []
        total_length = 0
        max_length = dough_roll.length
        iteration = 0
        total_iterations = 6 / precision
        start_time = time.time()
        while total_length < max_length:
            # Mise à jour de la barre de progression pour la génération initiale
            iteration += 1
            print_progress_bar(iteration, total_iterations, prefix=':', suffix='Complete', length=50,start_time=start_time)

            eligible_biscuits = [
                b for b in sorted_biscuits
                if b.length <= (max_length - total_length) and not overlaps_with_defects(total_length, b, dough_roll.defects)
            ]

            if not eligible_biscuits:
                total_length += precision  # Essayer de contourner les défauts
                continue

            selected_biscuit = eligible_biscuits[0]
            biscuit_copy = Biscuit(selected_biscuit.length, selected_biscuit.value, selected_biscuit.defect_thresholds)
            biscuit_copy.position = total_length
            solution.append(biscuit_copy)
            total_length += biscuit_copy.length

        # Final update to the progress bar to indicate completion
        print_progress_bar(iteration=max_length, total=max_length, prefix='Generating Solution:', suffix='Complete', length=50)
        print()  # New line at the end

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
        max_attempts = 400
        for _ in range(max_attempts):
            neighbor = solution[:]
            idx_to_replace = random.randrange(len(neighbor))
            # Filtrer les biscuits éligibles pour le remplacement
            remaining_length = dough_roll.length - sum(b.length for i, b in enumerate(neighbor) if i != idx_to_replace)
            eligible_biscuits = [b for b in sorted_biscuits if b.length <= remaining_length]
            
            if eligible_biscuits:
            # Remplacer par un biscuit ayant un meilleur rapport valeur/longueur
                chosen_biscuit = max(eligible_biscuits, key=lambda b: b.value / b.length) 
                # S'assurer que le biscuit choisi ne viole pas les seuils de défauts.
                temp_position = sum(b.length for i, b in enumerate(neighbor) if i < idx_to_replace)
                if not overlaps_with_defects(temp_position, chosen_biscuit, dough_roll.defects):
                    chosen_biscuit_copy = Biscuit(chosen_biscuit.length, chosen_biscuit.value, chosen_biscuit.defect_thresholds)
                    chosen_biscuit_copy.position = temp_position
                    neighbor[idx_to_replace] = chosen_biscuit_copy
                    if respects_constraints(neighbor):
                        # Recalculer les positions des biscuits après le remplacement
                        total_length = 0
                        for biscuit in neighbor:
                            biscuit.position = total_length
                            total_length += biscuit.length
                        return neighbor
        
        return solution
    # Initialisation d'une solution aléatoire
    print("Debut de generation")
    current_solution = generate_initial_solution()
    print("Debut de calculate value")
    current_value = calculate_value(current_solution)
    print("Debut boucle")
    max_steps_without_improvement = myiteration  # Augmentez si nécessaire

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
    print()  # Nouvelle ligne à la fin
    
    return current_solution

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
    
    print("Starting of the hill climbing search")
    # Créer une solution initiale qui respecte les contraintes de défauts
    sorted_biscuits = sorted(biscuits, key=lambda b: b.value / b.length, reverse=True)
    current_solution = []
    total_length = 0
    max_length = dough_roll.length
    step = 0
    iteration = 0
    total_iterations = 6 / precision
    start_time = time.time()
    while total_length < max_length:
        # Mise à jour de la barre de progression pour la génération initiale
        iteration += 1
        print_progress_bar(iteration, total_iterations, prefix='Hill Climbing:', suffix='Complete', length=50,start_time=start_time)

        eligible_biscuits = [b for b in sorted_biscuits if total_length + b.length <= max_length and not overlaps_with_defects(total_length, b, dough_roll.defects)]
        if not eligible_biscuits:
            total_length += precision  # Essayer de contourner les défauts
            continue
        selected_biscuit = max(eligible_biscuits, key=lambda b: b.value / b.length)  # Sélectionner le biscuit avec le meilleur ratio valeur/longueur parmi les éligibles

        # Créer une copie du biscuit avec la position mise à jour
        biscuit_copy = Biscuit(selected_biscuit.length, selected_biscuit.value, selected_biscuit.defect_thresholds)
        biscuit_copy.position = total_length
        current_solution.append(biscuit_copy)  
        total_length += biscuit_copy.length

    # Final update to the progress bar to indicate completion
    print_progress_bar(iteration=max_length, total=max_length, prefix='Hill Climbing:', suffix='Complete', length=50)
    print()  # New line at the end

    current_value = calculate_value(current_solution)

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
    
            best_biscuit = None
            best_biscuit_value = -1
            for biscuit in eligible_biscuits:
                if biscuit.value / biscuit.length > best_biscuit_value and not overlaps_with_defects(sum(b.length for i, b in enumerate(neighbor) if i < idx_to_change), biscuit, dough_roll.defects):
                    best_biscuit = biscuit
                    best_biscuit_value = biscuit.value / biscuit.length
            if best_biscuit:
                best_biscuit.position = sum(b.length for i, b in enumerate(neighbor) if i < idx_to_change)
                neighbor[idx_to_change] = best_biscuit
                if respects_constraints(neighbor, dough_roll):
                    return neighbor
        return solution

    # Perform hill climbing
    max_steps_without_improvement = myiteration  # Augmentez si nécessaire
    
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
