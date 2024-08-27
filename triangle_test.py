import numpy as np
from matplotlib import pyplot as plt
import random

def midpoint(x1,y1,x2,y2):
    x_new = (x1+x2)/2.0
    y_new = (y1+y2)/2.0
    return(x_new,y_new)

fig, ax = plt.subplots()
#plt.axis('off')
ax.set_aspect('equal')
ax.set_facecolor('black')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
fig.patch.set_facecolor('black')

# Corners
xcoords = [0,1,np.cos(np.radians(60.0))]
ycoords = [0,0,np.sin(np.radians(60.0))]
ax.plot(xcoords,ycoords,marker='.',linestyle='none',color='white')

# Random point
x = np.random.uniform(0,1)*np.cos(np.radians(np.random.uniform(0,60)))
y = np.random.uniform(0,1)*np.sin(np.radians(np.random.uniform(0,60)))

ax.plot(x, y, marker='.', color='blue')

end = 1000
for i in range(0,end):

    # Random corner
    corner_idx = random.randint(0,2)
    xcorner = xcoords[corner_idx]
    ycorner = ycoords[corner_idx]

    xold = x
    yold = y

    # Midpoint
    x,y = midpoint(x,y,xcorner,ycorner)
    ax.plot(x,y,marker='.',color='white')
    if i == end - 1 and (i <= 10 or i%100 == 0):
        ax.plot([xold,x,xcorner],[yold,y,ycorner],marker='.',color='blue',markersize=10,linewidth=5)
        ax.plot(xold,yold,marker='o',color='gold',markersize=10)
        ax.plot(xcorner,ycorner, marker='o', color='gold',markersize=10)
        ax.plot(x,y,marker='o',color='red',markersize=10)

    if i <= 10 or i%100 == 0:
        plt.savefig(r'C:\Users\Jeremy\Desktop\triangles\{0}.png'.format(i))

plt.show()