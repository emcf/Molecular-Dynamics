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
            r = np.sqrt(np.sum((xs[j] - xs[i])**2))# Distance
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

# Initializes a model with n particles
def initialize(n):
    # Initialize pyplot stuff
    global fig, ax, box_size
    box_size = 100
    #plt.xlim(-2, 200)
    #plt.ylim(-2, 20)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.show()

    # Set particle positions on a grid
    global xs, vs
    #x = np.arange(-(box_size/2), (box_size/2), box_size/np.sqrt(n))
    #y = np.arange(-(box_size/2), (box_size/2), box_size/np.sqrt(n))
    #X,Y = np.meshgrid(x,y)
    #xs = np.array([X.flatten(),Y.flatten()]).T + np.random.rand(n, 2)*3
    #vs = np.random.rand(n, 2) * 0
    xs = (np.random.rand(n, 2) - 0.5) * box_size
    vs = (np.random.rand(n, 2) - 0.5)

def observe():
    global xs, fig, ax, t
    ax.clear()
    plt.xlim(-box_size, box_size)
    plt.ylim(-box_size, box_size)
    #ax.set_facecolor((0.1, 0.1, 0.1))
    ax.scatter(xs[:, 0], xs[:, 1]) #, color = (0.7, 0.7, 0.7))
    fig.canvas.draw()
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    # plt.savefig('figures/' + str(t) + '.png', bbox_inches='tight', pad_inches = -0.06)

def update():
    for i in range(len(xs)):
        leapfrog_integrate_1(i)
        leapfrog_integrate_2(i)

def compute_total_energy():
    global xs, vs, A, B, g
    total_potential = 0
    total_kinetic = 0
    for i in range(len(xs)):
        v = np.sqrt(np.sum(vs[i]**2))
        KE = m * v * v / 2
        total_kinetic += KE
        for j in range(len(xs)):
            if (i != j):
                r = np.sqrt(np.sum((xs[j] - xs[i])**2))
                V = (-A/pow(r, 12)) + (B/pow(r, 6))
                total_potential += V
    return (total_potential/2 + total_kinetic)

dt = 0.1
initialize(n = 4)
for t in range(10000):
    observe()
    update()
    print(compute_total_energy())
