import numpy as np
from scipy.linalg import solve
from matplotlib import plt

nx = 4
alpha = 0.07

# A matrix
a_a = np.full(nx, -alpha)
a_a[-1] = -2*alpha #
a_b = np.full(nx+1, 2*(1+alpha))
a_b[0] = 2*(1+alpha) #
a_b[-1] = 2*(1+alpha) #
a_c = np.full(nx, -alpha)
a_c[0] = -2*alpha #

#print('A')
#print(a_c)
#print(a_b)
#print(a_a)

# Populate the tridiagonal A array with its a,b, and c bands
a = np.zeros((nx, nx))
for l in range(1, len(a_a)):
#    print('row: ', l, ' col: ', l-1)
    a[l, l-1] = a_a[l-1]
    a[l-1, l] = a_c[l-1]
for l in range(0,len(a_b)-1):
    a[l, l] = a_b[l]

print(a)
stop

# B matrix
b_a = np.full(nx, alpha)
b_a[-1] = 2*alpha #
b_b = np.full(nx+1, 2*(1-alpha))
b_b[0] = 2*(1-alpha) #
b_b[-1] = 2*(1-alpha) #
b_c = np.full(nx, alpha)
b_c[0] = 2*alpha #

#print('B')
#print(b_c)
#print(b_b)
#print(b_a)

# Initial U
u=np.zeros(nx+2)
u[0]=4
u[-1]=4

# U
#print('U')
#print(u)

def calculate_bu(u, b_a, b_b, b_c):

    print(u)
    bu = []
    for l in range(1, nx):
        bu.append(u[l] * np.sum(b_a[l] + b_b[l] + b_c[l]))
        print('sum: ', np.sum(b_a[l] + b_b[l] + b_c[l]))

    b0 = (u[0] * b_b[0])
    b1 = (u[-1] * b_b[-1])

    bu.insert(0, b0)
    bu.append(b1)

    return(np.array(bu))

for i in range(0,2):
    bu = calculate_bu(u, b_a, b_b, b_c)
    print(bu)
    print(a)
    u = solve(a, bu)
    print(u)