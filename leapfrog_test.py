import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.pyplot as plt

dt = 0.25

dq = np.array([0.1,0.2,0.1,0.3,0.2,0.2,0.1,0.4,0.2,0.3,0.3,0.1,0.1,0.3,0.2,0.3,0.1,0.2])

dqdt=dq/dt

q = np.array([5.0])
p = np.array([5.0])

for i in range(1,len(dq)):
    q = np.append(q,q[i-1]+dqdt[i-1]*dt)
    p_1 = p[i-1]+dqdt[i-1]*2.0*dt
    p_i = (1.0-0.05)*((p[i-1]+p_1)/2.0)
    p_i = p_i + 0.05*p[i-1]
    #print(q)
    if (i-1)%2 == 0:
        p = np.append(p,p_i)
        p = np.append(p,p_1)

fig, ax = plt.subplots()
ax.plot(q)
ax.plot(p)
plt.show()


