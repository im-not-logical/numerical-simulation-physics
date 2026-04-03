# import matplotlib
# matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import matplotlib.animation as animation

#physics constant
N = 500
x = np.linspace(-5, 5, N)
dx = x[1] - x[0]
kinetic_coeff = -1.0 / (2.0 * dx**2)
off_diag = kinetic_coeff * np.ones(N - 1)

# Particle in a Box Matrix
V_box = np.where(np.abs(x) < 2, 0, 1000)
H_box = np.diag(-2.0 * kinetic_coeff * np.ones(N) + V_box) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
energies_box, psi_box = eigh(H_box)
psi0 = psi_box[:, 0] / np.sqrt(dx)
psi1 = psi_box[:, 1] / np.sqrt(dx)

# Tunneling
V_barrier = np.where((x > 0) & (x < 1), 5.0, 0.0)
V_barrier = np.where(np.abs(x) > 4.5, 1000, V_barrier)
H_bar = np.diag(-2.0 * kinetic_coeff * np.ones(N) + V_barrier) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
energies_bar, psi_bar = eigh(H_bar)
psi_tunnel = psi_bar[:, 2] / np.sqrt(dx) #state where E < 5


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Quantum State Time Evolution', fontsize=16, fontweight='bold')

# Panel 1
ax1.set_xlim(-3, 3)
ax1.set_ylim(0, 1.5)
ax1.plot(x, V_box, 'k--', lw=1)
ax1.set_title('Superposition Sloshing (Particle in Box)')
ax1.set_xlabel('Position')
ax1.set_ylabel(r'Probability Density $|\Psi|^2$')
ax1.grid(True)

# Panel 2
ax2.set_xlim(-4, 4)
ax2.set_ylim(-1, 8)
ax2.plot(x, V_barrier, 'k--', lw=1, label='Potential Barrier V(x)')
ax2.set_title('Phase Penetration (Tunneling)')
ax2.set_xlabel('Position')
ax2.set_ylabel('Wavefunction Amplitude (Real Part)')
ax2.grid(True)
ax2.legend(loc='upper left')


line_prob, = ax1.plot([], [], 'b-', lw=2)
line_real, = ax2.plot([], [], 'r-', lw=2)

def init():
    line_prob.set_data([], [])
    line_real.set_data([], [])
    return line_prob, line_real


def update(frame):
    t = frame * 0.05

    Psi_box = (psi0 * np.exp(-1j * energies_box[0] * t) + psi1 * np.exp(-1j * energies_box[1] * t)) / np.sqrt(2)
    line_prob.set_data(x, np.abs(Psi_box)**2)


    Psi_bar = psi_tunnel * np.exp(-1j * energies_bar[2] * t)
    line_real.set_data(x, energies_bar[2] + np.real(Psi_bar))

    return line_prob, line_real


ani = animation.FuncAnimation(fig, update, frames=150, init_func=init, blit=True, interval=20)
plt.tight_layout()
plt.show()
# ani.save('qmt.gif', writer='pillow', fps=30)
# print("Saved 'qmt.gif'")
