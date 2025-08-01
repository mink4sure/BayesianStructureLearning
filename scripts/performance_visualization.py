import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example input
directories = [
  "test",
]


scene_names = [
  "Test Result",
]

# === Read Data ===
good_merged_split_values = []
distances_values = []

for dir_path in directories:
    with open(os.path.join(dir_path, "good_merged_split.txt")) as f:
        values = list(map(int, f.read().strip().split()))
        good_merged_split_values.append(values)

    with open(os.path.join(dir_path, "distances.txt")) as f:
        distances = list(map(float, f.read().strip().splitlines()))
        distances_values.append(distances)

# Convert to numpy array for bar plotting
gms_array = np.array(good_merged_split_values, dtype=np.float16)  # Shape: (n_scenes, 3)
n_scenes = len(scene_names)
for i in range(15):
  # print(gms_array[i])
  # print(np.sum(gms_array[i]))
  # print(gms_array[i]/np.sum(gms_array[i]))
  gms_array[i] = gms_array[i]/np.sum(gms_array[i])

print(gms_array)

x = np.arange(n_scenes)  # Scene centers

# === Plotting ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

# --- Top: Grouped Bar Plot (Good, Merged, Split) ---
labels = ['Correct', 'Merged', 'Split']
colors = ['#4CAF50', '#FFC107', '#F44336']

width = 0.5  # Width of individual bars in the bar plot
bottom = np.zeros(len(scene_names))
for i in range(3):
    ax1.bar(x, gms_array[:, i], width, bottom=bottom, label=labels[i], color=colors[i])
    bottom += gms_array[:, i]

ax1.set_ylabel("Relative Occurrences")
ax1.legend()
# ax1.set_xticks(x)
# ax1.set_xticklabels(scene_names, rotation=45, ha='right')
ax1.set_xticks(x)
ax1.set_xticklabels(scene_names, rotation=45, ha='right')

# --- Bottom: Boxplot of Distances ---
sns.boxplot(data=distances_values, ax=ax2)
# plt.setp((ax1, ax2), xticks=x, xticklabels=scene_names)
# plt.xticks(ticks=range(len(scene_names)), labels=scene_names, rotation=45)
ax2.set_xticks(x)
ax2.set_xticklabels(scene_names, rotation=45, ha='right')
# ax2.boxplot(distances_values, patch_artist=True,
#             boxprops=dict(facecolor='lightblue', color='blue'),
#             medianprops=dict(color='red'),
#             whiskerprops=dict(color='blue'),
#             capprops=dict(color='blue'),
#             flierprops=dict(markerfacecolor='blue', marker='o', markersize=5))

ax2.set_ylabel("Distances (rel. to image size)")

fig.suptitle("Quality of Cluster Assignment over Different Scenes")
plt.tight_layout()
plt.show()
