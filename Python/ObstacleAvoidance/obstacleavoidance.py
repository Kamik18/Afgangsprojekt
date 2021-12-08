#%%
import cv2
import matplotlib.pyplot as plt

img_dir = '../data/Background'

img = cv2.imread(f'{img_dir}/Alfa_Laval_Sensor_062.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
edge_img = cv2.Canny(img, 100,200)
plt.imshow(img)