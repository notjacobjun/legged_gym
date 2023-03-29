import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# create velocity and distance arrays
velocities = np.arange(0.25, 3.25, 0.25)
distances = np.arange(0, 11)

# create random data for line plot
data = np.random.rand(len(velocities), len(distances))

# create line plot using seaborn
sns.set()
for i in range(len(velocities)):
    sns.lineplot(x=distances, y=data[i], label=f"Velocity = {velocities[i]} m/s")

plt.xlabel('Distance (m)')
plt.ylabel('Distance Traveled')
plt.legend()
plt.show()
