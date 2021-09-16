import os
import matplotlib.pyplot as plt
import numpy as np
import math

data_file = open(os.getcwd() + "\Lidar\data.txt")
data = data_file.readline()
data_file.close()

# Remove brackets
for character in "{ }":
    data = data.replace(character, '')
# Slit the string into a list of points
data_list = data.split(',', data.count(','))

# Create arrays for x and y
data_x = []
data_y = []
for x in data_list:
    compoents = x.split(':', 1)
    angle = -math.radians(float(compoents[0]))
    lenght = float(compoents[1])
    if (lenght > 10):
        data_x.append(lenght * np.cos(angle))   
        data_y.append(lenght * np.sin(angle))

#plt.plot(data_x, data_y)
plt.plot(data_x, data_y, '.')
plt.show()