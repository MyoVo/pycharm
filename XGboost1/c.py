import matplotlib.pyplot as plt

# Data for each label
data = {
    'Foetus': [3.591547, 2.691547, 4.241088, 2.028951],
    'Prone': [3.339021, 3.039021, 3.749232, 2.170239],
    'Side': [3.645966, 2.892625, 3.994245, 2.104223],
    'Supine': [3.382548, 2.910360, 3.858231, 2.239909]
}

# Features
features = ['Back', 'Waist', 'Buttocks', 'Legs']

# Titles for each plot
titles = ['Foetus', 'Prone', 'Side', 'Supine']

# Colors for each plot
colors = ['orange', 'blue', 'green', 'red']

# Create separate plots for each label and save as SVG
for label, values, title, color in zip(data.keys(), data.values(), titles, colors):
    plt.figure(figsize=(10, 6))
    plt.plot(features, values, marker='o', linestyle='-', color=color)
    plt.title(f'Pressure Distribution - {title}')
    plt.xlabel('')
    plt.ylabel('Pressure (kPa)')
    plt.grid(True)
    # Ensure lines are included in SVG
    plt.savefig(f'{title}_pressure_distribution.svg', format='svg')
    plt.show()
