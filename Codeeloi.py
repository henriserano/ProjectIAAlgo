import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Initialize counters
a_positions = []
b_positions = []
c_positions = []

lenDough = 500
a = 0
b = 0
c = 0
class Biscuit():
    def __init__(self,a,b,c,l,v) -> None:
        self.a = a
        self.b = b
        
biscuits = [
    {'a': 4, 'b': 2, 'c': 3, 'l': 4, 'v': 6},
    {'a': 5, 'b': 4, 'c': 4, 'l': 8, 'v': 12},
    {'a': 1, 'b': 2, 'c': 1, 'l': 2, 'v': 1},
    {'a': 2, 'b': 3, 'c': 2, 'l': 5, 'v': 8}
]

# Open the CSV file in read mode
data = pd.read_csv("defects.csv")
with open('defects.csv', 'r') as file:
    # Create a CSV reader object
    reader = csv.DictReader(file)

    # Iterate through each row of the CSV
    for row in reader:
        # Increment the appropriate counter based on the class of defects
        if row['class'] == 'a':
            a_positions.append(float(row['x']))
            a += 1
        elif row['class'] == 'b':
            b_positions.append(float(row['x']))
            b += 1
        elif row['class'] == 'c':
            c_positions.append(float(row['x']))
            c += 1

# Print statements for the first part of the code
print("LE BUT EST D'OBTENIR LA PLUS GRANDE VALEUR CUMULÉE")
print("\nvalue sur la lenth de chaque biscuit :")

for i in range(len(biscuits)) : 
    print("\tbiscuit", i+1, ": ", biscuits[i]['v'] / biscuits[i]['l'])
print("Sans imperfection la max value serait de :", lenDough * biscuits[3]['v'] / biscuits[3]['l']**2)

# Afficher les résultats
print("\nNombre d'imperfections 'a':", a)
print("Nombre d'imperfections 'b':", b)
print("Nombre d'imperfections 'c':", c)

# Plotting the positions on a 1-dimensional space
plt.scatter(a_positions, [1] * len(a_positions), color='red', label='Class A')
plt.scatter(b_positions, [2] * len(b_positions), color='blue', label='Class B')
plt.scatter(c_positions, [3] * len(c_positions), color='green', label='Class C')

plt.title('Defect Positions on 1-Dimensional Space')
plt.xlabel('Position on the 1-Dimensional Space')
plt.yticks([1, 2, 3], ['Class A', 'Class B', 'Class C'])
plt.legend()
plt.show()

# Plotting the positions on a 1-dimensional space
fig, axs = plt.subplots(figsize=(10, 6))

# Display the capacity of biscuits to absorb defects in general on the value using grouped bar graphs
general_capacity = [(biscuits[i]['a']*a + biscuits[i]['b']*b + biscuits[i]['c']*c) / biscuits[i]['v'] for i in range(len(biscuits))]
bar_width = 0.2
bar_positions = np.arange(1, len(biscuits)+1)

axs.bar(bar_positions - bar_width, general_capacity, width=bar_width, color='purple', alpha=0.7, label='General')
axs.set_title("Capacity of Biscuits to Absorb Defects on the Value")
axs.set_xticks(bar_positions)
axs.set_xticklabels([f'Biscuit {i}' for i in range(1, len(biscuits)+1)])
axs.legend()

# Display the capacity of biscuits to absorb 'a', 'b', and 'c' type defects on the value using grouped bar graphs
for i, defect_type in enumerate(['a', 'b', 'c']):
    defect_count = locals()[defect_type]
    capacity = [biscuits[j][defect_type] * defect_count / biscuits[j]['v'] for j in range(len(biscuits))]
    axs.bar(bar_positions + i * bar_width, capacity, width=bar_width, label=f'{defect_type.upper()} Type', alpha=0.7)

axs.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.show()

# Plotting the positions on a 1-dimensional space for the length
fig, axs_length = plt.subplots(figsize=(10, 6))

# Display the capacity of biscuits to absorb defects in general on the length using grouped bar graphs
general_capacity_length = [(biscuits[i]['a']*a + biscuits[i]['b']*b + biscuits[i]['c']*c) / biscuits[i]['l'] for i in range(len(biscuits))]

axs_length.bar(bar_positions - bar_width, general_capacity_length, width=bar_width, color='purple', alpha=0.7, label='General')
axs_length.set_title("Capacity of Biscuits to Absorb Defects on the Length")
axs_length.set_xticks(bar_positions)
axs_length.set_xticklabels([f'Biscuit {i}' for i in range(1, len(biscuits)+1)])
axs_length.legend()

# Display the capacity of biscuits to absorb 'a', 'b', and 'c' type defects on the length using grouped bar graphs
for i, defect_type in enumerate(['a', 'b', 'c']):
    defect_count = locals()[defect_type]
    capacity_length = [biscuits[j][defect_type] * defect_count / biscuits[j]['l'] for j in range(len(biscuits))]
    axs_length.bar(bar_positions + i * bar_width, capacity_length, width=bar_width, label=f'{defect_type.upper()} Type', alpha=0.7)

axs_length.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.show()
