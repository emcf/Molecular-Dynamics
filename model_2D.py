import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde

m = 1 # Mass of each particle
A, B = 1, 1

# Calculates the force exerted on particle i
def lennard_jones_force(i):
    global A, B, xs
    F_net = np.zeros(2)
    for j in range(len(xs)):
        if (i != j):
            r = np.sqrt(np.sum((xs[j] - xs[i])**2)) + 0.5 # Distance
            R = -(xs[j] - xs[i]) / r # Normalized direction vector
            F = 12*A/pow(r, 13) - 6*B/pow(r, 7) # Force between i and j
            F_net += F*R
    return F_net

# performs the first step of leapfrog integration for particle i
def leapfrog_integrate_1(i):
    global xs, vs, dt, m
    a = (1/m) * lennard_jones_force(i)
    vs[i] += (a / 2) * dt
    xs[i] += vs[i] * dt

# performs the second step of leapfrog integration for particle i
def leapfrog_integrate_2(i):
    global xs, vs, dt, m
    a = (1/m) * lennard_jones_force(i)
    vs[i] += (a / 2) * dt

# Lets particles bounce off walls
def enforce_boundaries(i):
    global xs, vs
    if (xs[i, 0] > box_size):
        vs[i, 0] = -np.abs(vs[i, 0])
    if (xs[i, 0] < -box_size):
        vs[i, 0] = np.abs(vs[i, 0])
    if (xs[i, 1] > box_size):
        vs[i, 1] = -np.abs(vs[i, 1])
    if (xs[i, 1] < -box_size):
        vs[i, 1] = np.abs(vs[i, 1])

# Initializes a model with n particles
def initialize(n, temp):
    # Initialize pyplot stuff
    global fig, ax, box_size
    box_size = 5
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.show()

    # Set particle positions on a grid
    global xs, vs
    x = np.arange(-(box_size/2), (box_size/2), box_size/np.sqrt(n))
    y = np.arange(-(box_size/2), (box_size/2), box_size/np.sqrt(n))
    X,Y = np.meshgrid(x,y)
    xs = np.array([X.flatten(),Y.flatten()]).T + np.random.rand(n, 2)*0.1
    vs = (np.random.rand(n, 2) - 0.5) * temp

def observe():
    global xs, fig, ax, t
    ax.clear()
    plt.xlim(-box_size, box_size)
    plt.ylim(-box_size, box_size)
    ax.scatter(xs[:, 0], xs[:, 1])
    fig.canvas.draw()
    plt.savefig('figures/' + str(t) + '.png')

def update():
    for i in range(len(xs)):
        leapfrog_integrate_1(i)
        leapfrog_integrate_2(i)
        enforce_boundaries(i)

def compute_total_energy():
    global m, xs, vs, A, B, g
    total_potential = 0
    total_kinetic = 0
    for i in range(len(xs)):
        # Compute particle velocity
        v = np.sqrt(np.sum(vs[i]**2))
        # Compute kinetic energy
        KE = m * v * v / 2
        total_kinetic += KE
        for j in range(len(xs)):
            if (i != j):
                # Compute distance between particles
                r = np.sqrt(np.sum((xs[j] - xs[i])**2))
                # Compute lennard-jones potential
                V = (-A/pow(r, 12)) + (B/pow(r, 6))
                total_potential += V
    return (total_potential/2 + total_kinetic)

def compute_total_momentum():
    global m, xs, vs, A, B, g
    p_total = 0
    for i in range(len(xs)):
        v = np.sqrt(np.sum(vs[i]**2))
        p_total += m * v
    return p_total

# Initialize 64 particles at an initial temperature of 0.1 (arbitrary units)
# Note: n must have an integer square root, or else a grid will not form
initialize(n = 64, temp = 0.1)
dt = 0.05
for t in range(10000):
    observe()
    update()
    print('time =', t)
