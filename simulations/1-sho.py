import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# parameters and initial conditions
m  = 1.0    # kg
k  = 10.0   # N/m
x0 = 1.0    # m,
v0 = 0.0    # m/s

omega = np.sqrt(k / m)          # natural frequency
T     = 2 * np.pi / omega       # period
t_end = 5 * T

# numerically solving
def shm(t, y):
    x, v = y
    return [v, -omega**2 * x]

sol = solve_ivp(
    shm,
    t_span=(0, t_end),
    y0=[x0, v0],
    method='RK45',
    dense_output=True,
    rtol=1e-9,
    atol=1e-9
)

t = np.linspace(0, t_end, 2000)
x_num = sol.sol(t)[0]
v_num = sol.sol(t)[1]

# analytical solution
x_ana = x0 * np.cos(omega * t)
v_ana = -x0 * omega * np.sin(omega * t)

# energy
E_num = 0.5 * m * v_num**2 + 0.5 * k * x_num**2
E_ana = 0.5 * k * x0**2

# matplotlib
fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=False)

# mass on spring simulation
axes[0].set_xlim(-x0 * 1.5, x0 * 1.5)
axes[0].set_ylim(-1, 1)
axes[0].set_yticks([])
axes[0].set_title('1D SHO Simulation (Mass on Spring)')
axes[0].axhline(0, color='black', lw=1)
mass_point, = axes[0].plot([], [], 'bo', markersize=15)

# position vs time
axes[1].plot(t, x_ana, '--', label='Analytical', lw=1.5, color='orange')
trace_line, = axes[1].plot([], [], label='Simulated', lw=2, color='blue')
vline1 = axes[1].axvline(0, color='gray', linestyle=':')
axes[1].set_xlim(0, t_end)
axes[1].set_ylim(-x0 * 1.2, x0 * 1.2)
axes[1].set_ylabel('x (m)')
axes[1].set_title('Simple Harmonic Motion — Simulated vs Analytical')
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# energy drift
axes[2].plot(t, E_num - E_ana, color='steelblue')
axes[2].set_ylabel(r'$\Delta E (J)$')
axes[2].set_title('Energy drift')
axes[2].set_xlabel('Time (s)')
axes[2].grid(True, alpha=0.3)
vline3 = axes[2].axvline(0, color='gray', linestyle=':')

plt.tight_layout()

# animation
def init():
    mass_point.set_data([], [])
    trace_line.set_data([], [])
    vline1.set_xdata([0])
    vline3.set_xdata([0])
    return mass_point, trace_line, vline1, vline3

def update(frame):
    mass_point.set_data([x_num[frame]], [0])
    trace_line.set_data(t[:frame], x_num[:frame])
    vline1.set_xdata([t[frame]])
    vline3.set_xdata([t[frame]])

    return mass_point, trace_line, vline1, vline3

# create animation
ani = anim.FuncAnimation(
    fig,
    update,
    frames=range(0, len(t), 10),
    init_func=init,
    blit=True,
    interval=20 # 20 milliseconds between frames
)
plt.show()
ani.save('animations/sho.gif', writer='pillow', fps=30)
