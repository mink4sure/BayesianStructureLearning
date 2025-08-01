import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from ultralytics import YOLO

# Path to an image
source = "data/images/DSCF7038.JPG"

# Path to a directory to save results. Directory needs to exist
destination  = "data/measurements/7038"

# Some parameters for data generation
patch_size_x = 2000
patch_size_y = 3000
N_patches = 100
noise = 50    # Standard deviation of the noise

# Loading the image
img = cv.imread(source)
img_size_x, img_size_y, _ = img.shape

# Generating random starting positions
x_start_max = img_size_x - patch_size_x
y_start_max = img_size_y - patch_size_y
x_starts = np.random.randint(0, x_start_max, N_patches)
y_starts = np.random.randint(0, y_start_max, N_patches)

# Configuring YOLO model
model = YOLO("yolo11x.pt")

# Tracking results
global_xy = np.array([[0, 1, 2, 3, 4, 5]])

# Creating the patches
for i in range(N_patches):
  start_x = x_starts[i]
  start_y = y_starts[i]
  stop_x = start_x + patch_size_x
  stop_y = start_y + patch_size_y

  noise_x, noise_y = np.random.normal(0, noise, 2)

  patch = img[start_x:stop_x, start_y:stop_y]
  name = destination+"/patch"+str(i)+".png"
  print(name)
  # cv.imwrite(name, patch)

  # Running image recongnition on all the
  # patches and saving the results
  results = model.predict(
    source    = patch,
    save      = True,
    project   = destination,
    name      = "patch"+str(i),
    save_txt  = True,
    save_conf = True,
    verbose   = False
  )

  boxes = results[0].boxes
  classes = boxes.cls
  confidences = boxes.conf
  xywh = boxes.xywh
  print("YOLO RESULTS:")
  print("Detected classes: ", classes)
  print("Detected confidences: ", confidences)
  print("Detected xywh: ", xywh)

  # For now olny interested in xy positions
  # transformed to "global" image coordinates
  xywh[:, 0] += start_y + noise_y
  xywh[:, 1] += start_x + noise_x
  print("xywh: ", xywh)

  xywhlc = np.ones((xywh.shape[0], 6))
  xywhlc[:, :4] = xywh.cpu().numpy()
  xywhlc[:, 4] = classes.cpu().numpy()
  xywhlc[:, 5] = confidences.cpu().numpy()

  global_xy = np.vstack((global_xy, xywhlc))

plt.plot(global_xy)
plt.show()

print("global XY: ", global_xy)
np.savetxt(destination+"/global_xy.csv", global_xy, delimiter=",")

# Plotting the data
global_xy = global_xy[global_xy[:, 4] != 60]
img = plt.imread(source)
plt.imshow(img)
plt.scatter(global_xy[:, 0], global_xy[:, 1], alpha=.6, color="red")
plt.savefig(destination+"/data_visualizaiton.png")
plt.show()
