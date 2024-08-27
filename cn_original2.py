import numpy as np
from scipy.linalg import solve
from matplotlib import pyplot as plt

def mltply_B(u,alpha):
    u_i = 2*(1-alpha)*u
    u_i=np.insert(u_i,0,0)
    u_i=np.append(u_i,0)

    u_iplus1=alpha*u
    u_iplus1=np.append(u_iplus1,[0,0])

    u_iminus1=alpha*u
    u_iminus1=np.insert(u_iminus1,0,0)
    u_iminus1=np.insert(u_iminus1,0,0)

    B_u=u_i+u_iplus1+u_iminus1
    B_u=B_u[1:-1]
    B_u[0]=(1-alpha)*u[0] # B top
    B_u[-1]=(1-alpha)*u[-1] # B bot

    return B_u


def TDMAsolver(a, b, c, d):
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays

    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


nx = 4
alpha = 0.07

u=np.zeros(nx+1)
u[0]=4
u[-1]=4

a=np.full(nx, -alpha)
b=np.full(nx+1, 2*(1+alpha))
b[0]=1+alpha # A top
b[-1]=1+alpha # A bot
c=np.full(nx, -alpha)

d=mltply_B(u,alpha)

u1=TDMAsolver(a,b,c,d)

print(np.sum(u))

print(np.sum(u1))