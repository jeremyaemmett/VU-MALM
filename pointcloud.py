import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def read_csv_header(filename, column_idx, var_type, header_lines):
    with open(filename) as f:
        reader = csv.reader(f, delimiter = ' ')
        if header_lines != 0:
            for h in range(0,header_lines):
                header = next(reader)
        vals = []
        for row in reader:
            if var_type == 'string':
                val = row[column_idx]
            if var_type == 'integer':
                val = int(row[column_idx])
            if var_type == 'float':
                if row[column_idx] == '':
                    val = -9999.0
                else:
                    val = float(row[column_idx])
            vals.append(val)
    return vals

file = 'C:/Users/Jeremy/Desktop/xyz1.txt'
x = read_csv_header(file, 0, 'float', 0)
y = read_csv_header(file, 1, 'float', 0)
z = read_csv_header(file, 2, 'float', 0)
z = [1.5 * z[i] for i in range(len(z))]

x, y, z = x[0::10], y[0::10], z[0::10]
x, y, z = np.array(x), np.array(y), np.array(z)

x2 = []
for i in range(0, len(x)):
    x2.append([x[i], y[i], z[i]])

# cluster data
kmeans = KMeans(init="k-means++", n_clusters=12, random_state=42, algorithm='elkan')
kmeans.fit(x2)
print(kmeans.labels_)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.scatter(x, y, z, alpha = 0.2, c=kmeans.labels_, cmap='viridis')
ax.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
           kmeans.cluster_centers_[:,2], s = 300, c = 'r',
           marker='.', label = 'Centroid')
plt.show()