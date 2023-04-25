import statistics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

n_trials = 20
increment_val, starting_velocity, ending_velocity = 0.25, 0.25, 3.00
num_increments = ending_velocity // starting_velocity
velocities, x_distances_traveled, avg_base_heights = list(np.arange(starting_velocity, ending_velocity + increment_val, increment_val)), {}, {}

# read the distance and height values for each respective file
distance_df = pd.read_pickle('distance_data.pkl')
height_df = pd.read_pickle('height_data.pkl')

print(distance_df)
print(height_df)

# transform the list of values in the distance and height df to be averages instead
# for row_idx, row in height_df.iterrows():
#     for col_idx, value in row.items():
#         # print(row_idx, col_idx)
#         height_df.loc[[row_idx],[col_idx]] = statistics.mean(value)
# print(height_df)

# for row_idx, row in distance_df.iterrows():
#     for col_idx, value in row.items():
#         # print(row_idx, col_idx)
#         distance_df.loc[[row_idx],[col_idx]] = statistics.mean(value)
# print(distance_df)

# save the dataframes into heatmaps
# height_df.style.background_gradient(cmap="Blues")
# cm = sns.light_palette("green", as_cmap=True)
# distance_df.style.background_gradient(cmap=cm)

# height_df.to_html("height_data.html")
# distance_df.to_html("distance_data.html")

# # hardcoded the num_measurements value to 2 since we are measuring against the velocity and the height 
# num_velocities, num_measurements = len(velocities), 2

# # reshape the dataframes to plot easily in the heatmap
# # height_df = height_df.pivot(index=)

# Create the heatmaps
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# convert the dytype to float
distance_df = distance_df.astype(float)
height_df = height_df.astype(float)

sns.heatmap(distance_df, ax=axes[0])
sns.heatmap(height_df, ax=axes[1])

axes[0].set_xlabel("Gap size (meters)")
axes[0].set_ylabel("Velocity")
axes[0].set_title("Distance traveled based on changes of velocity and gap size")
axes[0].invert_yaxis()

axes[1].set_xlabel("Gap size (meters)")
axes[1].set_ylabel("Velocity")
axes[1].set_title("Average height based on velocity and gap size")
axes[1].invert_yaxis()

plt.savefig("/home/learning/Pictures/RLHF_toy_heatmap.png")
plt.show()

