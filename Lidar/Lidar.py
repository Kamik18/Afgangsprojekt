import os

print("Directory:")

data_file = open(os.getcwd() + "\Lidar\data.txt")
data = data_file.readline()
print(data)
data_file.close()
