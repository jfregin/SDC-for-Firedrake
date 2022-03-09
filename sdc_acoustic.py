from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

a = 0                           # time of start
b = 3                           # time of end
n = 512                         # number of spatial nodes
n_steps = 154.                  # number of time steps
dt = (b-a)/(n_steps)            # delta t

mesh = PeriodicIntervalMesh(n, 1)
Vu = FunctionSpace(mesh, 'Lagrange', 1)
Vp = FunctionSpace(mesh, 'CG', 1)
W = MixedFunctionSpace((Vu, Vp))

U0 = Function(W)
u0, p0 = U0.split()

x1 = 0.25
x0 = 0.75
sigma = 0.1
k = 7.2*np.pi


def p_0(x, sigma=sigma):
    return exp(-x**2/sigma**2)
                 
def p_1(x, p0=p_0, sigma=sigma, k=k):
    return p0(x)*cos(k*x/sigma)

def p_init(x, p0=p_0, p1=p_1, x0=x0, x1=x1, coeff=1.):
    return p_0(x-x0) + coeff*p_1(x-x1)

x = SpatialCoordinate(mesh)[0]
p0.interpolate(p_init(x))

# problem specific constants
c_s = Constant(1)                   # speed of sound
u_mean = Constant(0.05)                  # mean flow

w, phi = TestFunctions(W)
u, p = TrialFunctions(W)

Uf = Function(W)
u1, p1 = split(Uf)

F = (w * (u1 - u0) + phi * (p1 - p0)) * dx + dt * c_s * (w * p1.dx(0) + phi * u1.dx(0)) * dx
fast_problem = NonlinearVariationalProblem(F, Uf)
fast_solver = NonlinearVariationalSolver(fast_problem)

Us = Function(W)
aslow = (w * u + phi * p) * dx
Lslow = (w * u0 + phi * p0) * dx - dt * u_mean * (w * u0.dx(0) + phi * p0.dx(0)) * dx
slow_problem = LinearVariationalProblem(aslow, Lslow, Us)
slow_solver = LinearVariationalSolver(slow_problem)

outfile = File("acoustic.pvd")

t = a
while t < b:
    print(t)
    slow_solver.solve()
    U0.assign(Us)
    fast_solver.solve()
    U0.assign(Uf)
    u0, p0 = U0.split()
    t += dt
    outfile.write(u0, p0)
