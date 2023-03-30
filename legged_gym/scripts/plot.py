import matplotlib.pyplot as plt

# create dictionary of distances traveled with corresponding velocities
distances = {'0.25': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
             '0.50': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
             '0.75': [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0],
             '1.00': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
             '1.25': [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0],
             '1.50': [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0],
             '1.75': [0.0, 3.5, 7.0, 10.5, 14.0, 17.5, 21.0, 24.5, 28.0],
             '2.00': [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0],
             '2.25': [0.0, 4.5, 9.0, 13.5, 18.0, 22.5, 27.0, 31.5, 36.0],
             '2.50': [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0],
             '2.75': [0.0, 5.5, 11.0, 16.5, 22.0, 27.5, 33.0, 38.5, 44.0],
             '3.00': [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]}

# create data for scatter plot
x_data = []
y_data = []
for velocity in distances:
    for i, distance in enumerate(distances[velocity]):
        x_data.append(float(velocity))
        y_data.append(distance)

# create scatter plot using matplotlib
plt.scatter(x_data, y_data, s=50)
plt.xlabel('Velocity')
plt.ylabel('Distance Traveled')
plt.title('Distances Traveled at Different Velocities')
plt.show()
