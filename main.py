# importing the required module
import matplotlib.pyplot as plt
import numpy as np

obj_size = 256
dist_to_hologram = obj_size + 0.5
wave_len = 1
value = 100

prj = [0] * obj_size
prj[value] = 1

point_hologram = [0] * obj_size
hologram = [0] * obj_size
for i in range(0, obj_size):
    j = i - value
    hypot = np.hypot(dist_to_hologram, j)
    hologram[i] = 1 * np.cos((hypot / wave_len - np.fix(hypot / wave_len)) / wave_len * 2 * np.pi)
    point_hologram[i] = 1 if hologram[i] > 0 else 0

fig, axs = plt.subplots(2)
fig.suptitle('Point hologram')
axs[0].plot(range(0, obj_size), point_hologram)
axs[1].plot(range(0, obj_size), hologram)
plt.show()