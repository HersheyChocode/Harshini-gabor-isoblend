import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.plot(x, y)
plt.title("Sine Wave")

# CRITICAL: This opens the Matplotlib viewer window
plt.show()