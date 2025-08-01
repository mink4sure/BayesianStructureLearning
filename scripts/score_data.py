import matplotlib.pyplot as plt
from math import dist
import numpy as np
import seaborn as sns

# Getting the image
image_source = "data/images/DSCF7038.JPG"
img = plt.imread(image_source)

# Getting inferred cluster means
inferred_cluster_means_source = "results/7038/cluster_centers.txt"
result_destination = "results/7038"
yolo_cluser_source = "data/images/yolo/7038/labels/DSCF7038.txt"

# Read the inference result file
x_inferred = []
y_inferred = []
with open(inferred_cluster_means_source, 'r') as file:
    for line in file:
        x, y = map(float, line.strip().split())
        x_inferred.append(x)
        y_inferred.append(y)

# Read second file (YOLO-style points with scaled coordinates)
x2_vals, y2_vals = [], []
with open(yolo_cluser_source, 'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) >= 3:
            x = float(parts[1]) * 6000  # scale x
            y = float(parts[2]) * 4000  # scale y
            x2_vals.append(x)
            y2_vals.append(y)


# Plotting
plt.figure(figsize=(8, 6))
plt.imshow(img)

# First set
plt.scatter(x_inferred, y_inferred, color='blue', label='Inferred Clusters')
for i, (x, y) in enumerate(zip(x_inferred, y_inferred)):
    plt.text(x, y, str(i), fontsize=12, ha='right', va='bottom', color="blue")

# Second set (YOLO scaled)
plt.scatter(x2_vals, y2_vals, color='red', label='YOLO Points')
for i, (x, y) in enumerate(zip(x2_vals, y2_vals)):
    plt.text(x, y, f"Y{i}", fontsize=12, ha='left', va='top', color='red')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('XY Coordinates from source.txt')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# This variable needs to be manually set.
# In a tuple (a, b), a corresponds to the index of the inffered cluster
# and b corresponds to the index of the yolo detected cluster.
pairs = [(5, 0), (0, 1), (1, 2), (4, 3)]

# Calculating distances based upon these pairs
distances = []
for s_idx, y_idx in pairs:
    p1 = (x_inferred[s_idx]/6000, y_inferred[s_idx]/4000)
    p2 = (x2_vals[y_idx]/6000, y2_vals[y_idx]/4000)
    d = dist(p1, p2)
    distances.append(d)
# np.savetxt(result_destination+"distances.txt", distances)

# Plotting these distances
plt.figure()
sns.boxplot(distances)
plt.show()


# Plotting
plt.figure()
plt.imshow(img)
plt.scatter(x_inferred, y_inferred, color='blue', label='Inferred cluster centers')
plt.scatter(x2_vals, y2_vals, color='red', label='YOLO detected objects')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('XY Coordinates from source.txt')
# plt.legend()
plt.tight_layout()
plt.savefig(result_destination+"cluster_distances.png")
plt.show()
