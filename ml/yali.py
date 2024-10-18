import matplotlib.pyplot as plt
import numpy as np

# Data from the image
labels = ['P1', 'P2', 'P3', 'P4']
supine = [3.32, 2.21, 3.61, 1.98]
side = [3.55, 2.83, 3.73, 2.14]
foetus = [3.59, 2.98, 3.93, 1.64]
prone = [2.89, 3.22, 3.11, 2.04]

# Number of groups
x = np.arange(len(labels))
y = np.array([0, 1, 2, 3])  # Supine, Side, Foetus, Prone
x, y = np.meshgrid(x, y)

# Flatten the arrays for 3D plotting
x = x.flatten()
y = y.flatten()
z = np.zeros_like(x)

# Heights (pressure values) for each point
height = np.array([supine, side, foetus, prone]).flatten()

# Colors for each sleeping position
colors = ['r', 'g', 'b', 'y']  # Supine, Side, Foetus, Prone

# Create a 3D line plot with larger figure size
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

# Plotting the lines for each sleeping position with different colors
for i, color in enumerate(colors):
    ax.plot(x[y == i], y[y == i], height[y == i], color=color, label=['Supine', 'Side', 'Foetus', 'Prone'][i], marker='o')

# Setting labels and ticks
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels([r'$P_1$', r'$P_2$', r'$P_3$', r'$P_4$'])

ax.set_yticks(np.arange(4))
ax.set_yticklabels(['Supine', 'Side', 'Foetus', 'Prone'])

ax.legend()

# Adjust layout to minimize blank spaces
plt.subplots_adjust(left=0.01, right=0.95, top=0.95, bottom=0.05)
plt.savefig('yali.png', format='png', bbox_inches='tight', dpi=100)
# Show plot
plt.show()
