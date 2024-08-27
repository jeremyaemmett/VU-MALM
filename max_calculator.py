import numpy as np
import matplotlib.pyplot as plt

max = 150

bar_weight = 20.0

plate_set = [25, 25, 20, 20, 15, 15, 10, 10, 5, 5, 2.5, 2.5, 1.5, 1.5]
plate_sel = np.zeros(len(plate_set))

pcts = [0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05]
reps = [8, 5, 3, 1, 1, 1, 1]
rest = [2, 2, 3, 3, 5, 5, 5]

plates = []
for i in range(0, len(pcts)):

    total_weight = pcts[i] * max
    plate_weight_per_side = (total_weight - bar_weight) / 2.0

    selected_plate_index = np.where(np.array(plate_set) <= plate_weight_per_side)[0][0]
    plates.append(plate_set[selected_plate_index])

    print(plate_weight_per_side, plates)


