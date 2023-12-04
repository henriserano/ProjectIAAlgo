import pandas as pd
import heapq

# Étape 1: Charger les données du CSV
def load_defects(csv_filepath):
    return pd.read_csv(csv_filepath)

class Biscuit:
    def __init__(self, length, value, defect_thresholds):
        self.length = length
        self.value = value
        self.defect_thresholds = defect_thresholds

    def __lt__(self, other):
        # Définir une logique de comparaison, par exemple comparer les valeurs des biscuits
        return self.value < other.value

    # Vous pouvez aussi définir __eq__ si vous voulez gérer l'égalité
    def __eq__(self, other):
        return self.value == other.value

class Defect:
    def __init__(self, position, defect_class):
        self.position = position
        self.defect_class = defect_class

class DoughRoll:
    def __init__(self, length):
        self.length = length
        self.defects = []

def heuristic(dough_roll, state, biscuits, max_value_biscuit, min_biscuit_length):
    current_position, _ = state
    remaining_length = dough_roll.length - current_position
    max_biscuits_placeable = remaining_length // min_biscuit_length
    return max_value_biscuit * max_biscuits_placeable


def overlaps_with_defects(position, biscuit, defects):
    # Copier les seuils de défauts pour ne pas modifier l'objet biscuit
    defect_thresholds = biscuit.defect_thresholds.copy()

    # Vérifier si le biscuit chevauche un défaut
    for defect in defects:
        if position <= defect.position < position + biscuit.length:
            if defect_thresholds.get(defect.defect_class, 0) <= 0:
                return True
            defect_thresholds[defect.defect_class] -= 1
    return False


def a_star(dough_roll, biscuits):
    start_state = (0, ())  # Position de départ et tuple vide pour les positions des biscuits
    frontier = [(0, start_state)]  # Priorité et état initial
    explored = set()  # Ensemble des états explorés
    max_value_biscuit = max(biscuits, key=lambda b: b.value).value
    min_biscuit_length = min(biscuits, key=lambda b: b.length).length
    while frontier:
        _, current_state = heapq.heappop(frontier)
        current_position, biscuits_positions = current_state

        if current_position == dough_roll.length:
            # Retourner les biscuits placés en reconstruisant les objets Biscuit à partir de biscuits_positions
            return [biscuits[pos] for pos in biscuits_positions]

        # Convertir l'état en une forme hashable avant de le vérifier ou de l'ajouter à l'ensemble explored
        hashable_state = (current_position, tuple(biscuits_positions))

        if hashable_state in explored:
            continue

        explored.add(hashable_state)

        for i, biscuit in enumerate(biscuits):
            new_position = current_position + biscuit.length
            if new_position <= dough_roll.length and not overlaps_with_defects(current_position, biscuit, dough_roll.defects):
                # Ajouter l'indice du biscuit au tuple des positions
                new_state = (new_position, biscuits_positions + (i,))
                heapq.heappush(frontier, (heuristic(dough_roll,new_state, biscuits,max_value_biscuit,min_biscuit_length), new_state))

    return None  # Aucune solution trouvée




# Étape 4: Optimisation de l'espace
def optimize_biscuit_placement(csv_filepath):
    defects_df = load_defects(csv_filepath)
    dough_roll = DoughRoll(500)  # Longueur du rouleau de pâte
    for _, row in defects_df.iterrows():
        dough_roll.defects.append(Defect(row['x'], row['class']))
    
    # Définissez vos biscuits ici avec leurs tailles, valeurs, et seuils de défauts
    biscuits = [
        Biscuit(4, 6, {'a': 4, 'b': 2, 'c': 3}),
        Biscuit(8, 12, {'a': 5, 'b': 4, 'c': 4}),
        Biscuit(2, 1, {'a': 1, 'b': 2, 'c': 1}),
        Biscuit(5, 8, {'a': 2, 'b': 3, 'c': 2}),
    ]

    return a_star(dough_roll, biscuits)

# Utiliser la fonction pour optimiser le placement des biscuits
solution = optimize_biscuit_placement('defects.csv')
print(solution)
