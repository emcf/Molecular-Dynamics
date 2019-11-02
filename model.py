import numpy as np
import matplotlib.pyplot as plt

# Calculates the force exerted on particle i
def lennard_jones_force(i):
    global xs
    sigma = 0.1
    epsilon = 1
    F_net = np.zeros(2)
    for j in range(len(xs)):
        if (i != j):
            r = np.sqrt(np.sum((xs[j] - xs[i])**2)) # Calculate distance
            R = (xs[j] - xs[i]) / r # Normalized direction vector
            F = -24*pow(sigma, 6)*epsilon*(pow(r, 6) - 2*pow(sigma, 6)) / (pow(r, 13) + 1e-10)
            F_net += F*R
    return F_net

# verlet integrates particle i
# TODO: Make this leapfrog instead
def verlet_integrate(i):
    global xs
    global xs_prev
    dt = 0.1
    xs_new = xs
    a = lennard_jones_force(i)
    xs_new[i] = 2*xs[i] - xs_prev[i] + a*dt**2
    xs_prev = xs
    xs = xs_new

# Initializes a model with n particles
def initialize(n):
    # Initialize pyplot stuff
    global fig, ax
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.show()
    fig.canvas.draw()
    # Initialize particle positions
    global xs, xs_prev
    xs = np.random.rand(n, 2)
    xs_prev = xs

def observe():
    global xs, fig, ax
    ax.clear()
    ax.scatter(xs[:, 0], xs[:, 1])
    fig.canvas.draw()

def update():
    for i in range(len(xs)):
        verlet_integrate(i)

initialize(n = 50)
for t in range(1000):
    observe()
    update()
