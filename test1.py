import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

def plot_defects_on_1d_space_interactive(length=500, initial_view=20, defects_file='defects.csv', biscuits=None):
    # Function to update the plot based on slider values
    def update_plot(start, zoom):
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(10, 2))
        x = np.linspace(0, length, length)
        y = np.zeros_like(x)
        ax.plot(x, y, color='gray', label='Roll of Dough')

        unique_ratios = set()
        if biscuits:
            for biscuit in biscuits:
                x_position = biscuit['position']
                biscuit_width = biscuit['length']
                biscuit_ratio = (biscuit['value'] / biscuit['length']) / 2
                unique_ratios.add(biscuit_ratio)
                biscuit_color = plt.cm.Oranges(biscuit_ratio)
                biscuit_ellipse = patches.Ellipse((x_position + biscuit_width / 2, 0), biscuit_width, 1, edgecolor=biscuit_color, facecolor=biscuit_color)
                ax.add_patch(biscuit_ellipse)
                ax.axvline(x_position + biscuit_width, color='#CCCCCC', linestyle=':')

        # Read defects from the CSV file
        defects_data = pd.read_csv(defects_file)
        colors = {'a': '#009900', 'b': '#00AAAA', 'c': '#0066AA'}
        for _, row in defects_data.iterrows():
            x_position = row['x']
            defect_type = row['class']
            defect_y = {'a': 0.15, 'b': 0.1, 'c': 0.05}[defect_type]
            ax.scatter(x_position, defect_y, color=colors[defect_type], marker='x')

        # Set limits
        ax.set_xlim(start, start + zoom)

        # Add legend
        legend_labels = {'a': 'Defect Type A', 'b': 'Defect Type B', 'c': 'Defect Type C'}
        legend_handles = [plt.Line2D([0], [0], marker='x', color=colors[type_], markersize=8, label=legend_labels[type_]) for type_ in colors]
        for biscuit_ratio in unique_ratios:
            biscuit_color = plt.cm.Oranges(biscuit_ratio)
            legend_handles.append(patches.Ellipse((0, 0), 1, 1, color=biscuit_color, label=f'Biscuit Ratio: {biscuit_ratio:.2f}'))
        ax.legend(handles=legend_handles)

        plt.show()

    # Create widgets for sliders
    start_slider = widgets.IntSlider(value=0, min=0, max=length, step=1, description='Start:')
    zoom_slider = widgets.IntSlider(value=initial_view, min=1, max=length, step=1, description='Zoom:')

    # Display widgets
    display(start_slider, zoom_slider)

    # Update plot whenever sliders are changed
    widgets.interactive(update_plot, start=start_slider, zoom=zoom_slider)

# Example usage
biscuits_example = [
    {"position": 20, "length": 40, "value": 80},
    {"position": 100, "length": 50, "value": 90},
    # ... add more biscuits as needed
]

plot_defects_on_1d_space_interactive(biscuits=biscuits_example)
