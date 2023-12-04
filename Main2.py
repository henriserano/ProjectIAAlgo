import pandas as pd
import heapq

# Étape 1: Charger les données du CSV
def heuristic(biscuits, remaining_length):
    # On peut utiliser la valeur maximale par unité de longueur comme heuristique
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

# Utiliser la fonction pour optimiser le placement des biscuits
solution, defects = optimize_biscuit_placement('defects.csv')
print_solution(solution)
print_dough_visualization(solution, defects)
