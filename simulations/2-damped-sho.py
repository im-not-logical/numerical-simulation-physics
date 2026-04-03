import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Parameters ---
m  = 1.0    # kg
k  = 10.0   # N/m
x0 = 1.0    # m
v0 = 0.0    # m/s

omega0 = np.sqrt(k / m)         # natural  frequency
T0     = 2 * np.pi / omega0     # undamped time period
t_end  = 8 * T0

# three cases of gamma
b_under    = 1.0                 # underdamped:    gamma < omega0
b_critical = 2 * m * omega0    # critically damped: gamma = omega0 exactly
b_over     = 10.0               # overdamped:     gamma > omega0

cases = {
    'Underdamped'      : b_under,
    'Critically damped': b_critical,
    'Overdamped'       : b_over,
}

# ODE RHS
def damped_shm(t, y, gamma):
    x, v = y
    return [v, -omega0**2 * x - 2 * gamma * v]

# analytical solutions
def x_analytical(t, gamma):
    if gamma < omega0:
        omega_d = np.sqrt(omega0**2 - gamma**2)
        A = x0
        B = (v0 + gamma * x0) / omega_d
        return np.exp(-gamma * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    elif gamma == omega0:
        A = x0
        B = v0 + gamma * x0
        return np.exp(-gamma * t) * (A + B * t)
    else:
        delta = np.sqrt(gamma**2 - omega0**2)
        r1 = -gamma + delta
        r2 = -gamma - delta
        C2 = (v0 - r1 * x0) / (r2 - r1)
        C1 = x0 - C2
        return C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)

# running all cases
t = np.linspace(0, t_end, 3000)

results = {}
for label, b in cases.items():
    gamma = b / (2 * m)
    sol = solve_ivp(
        damped_shm,
        t_span=(0, t_end),
        y0=[x0, v0],
        method='RK45',
        dense_output=True,
        args=(gamma,),
        rtol=1e-10,
        atol=1e-10
    )
    x_num = sol.sol(t)[0]
    v_num = sol.sol(t)[1]
    x_ana = x_analytical(t, gamma)
    E = 0.5 * m * v_num**2 + 0.5 * k * x_num**2

    results[label] = {
        'gamma' : gamma,
        'x_num' : x_num,
        'v_num' : v_num,
        'x_ana' : x_ana,
        'E'     : E,
        'error' : x_num - x_ana,
    }

# matplotlib
fig, axes = plt.subplots(3, 1, figsize=(11, 11))
colors = {'Underdamped': 'steelblue', 'Critically damped': 'darkorange', 'Overdamped': 'seagreen'}

lines_x = {}
lines_E = {}
lines_phase = {}
vlines = []

# position vs time
for label, res in results.items():
    axes[0].plot(t, res['x_ana'], '--', color=colors[label], lw=1, alpha=0.6)
    lines_x[label], = axes[0].plot([], [], label=f"{label} (γ={res['gamma']:.2f})", color=colors[label], lw=2)

axes[0].axhline(0, color='k', lw=0.5, ls=':')
vlines.append(axes[0].axvline(0, color='gray', linestyle=':', alpha=0.7))
axes[0].set_xlim(0, t_end)
axes[0].set_ylim(-x0 * 1.1, x0 * 1.1)
axes[0].set_ylabel('x (m)')
axes[0].set_title('Damped SHO — Simulation')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# energy decay
for label, res in results.items():
    lines_E[label], = axes[1].plot([], [], label=label, color=colors[label], lw=2)

vlines.append(axes[1].axvline(0, color='gray', linestyle=':', alpha=0.7))
axes[1].set_xlim(0, t_end)
axes[1].set_ylim(0, 0.5 * k * x0**2 * 1.1)
axes[1].set_ylabel('E (J)')
axes[1].set_title('Mechanical energy decay')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# phase space plot
for label, res in results.items():
    lines_phase[label], = axes[2].plot([], [], label=label, color=colors[label], lw=1.5)

axes[2].set_xlim(-x0 * 1.1, x0 * 1.1)
max_v = x0 * omega0 * 1.1
axes[2].set_ylim(-max_v, max_v)
axes[2].set_ylabel('Velocity v (m/s)')
axes[2].set_xlabel('Position x (m)')
axes[2].set_title('Phase Space (x vs v)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# animation
def init():
    artists = []
    for label in cases.keys():
        lines_x[label].set_data([], [])
        lines_E[label].set_data([], [])
        lines_phase[label].set_data([], [])
        artists.extend([lines_x[label], lines_E[label], lines_phase[label]])
    for vline in vlines:
        vline.set_xdata([0])
        artists.append(vline)
    return artists

def update(frame):
    artists = []
    for label, res in results.items():
        lines_x[label].set_data(t[:frame], res['x_num'][:frame])
        lines_E[label].set_data(t[:frame], res['E'][:frame])
        lines_phase[label].set_data(res['x_num'][:frame], res['v_num'][:frame])
        artists.extend([lines_x[label], lines_E[label], lines_phase[label]])

    for vline in vlines:
        vline.set_xdata([t[frame]])
        artists.append(vline)

    return artists

ani = animation.FuncAnimation(
    fig,
    update,
    frames=range(0, len(t), 15),
    init_func=init,
    blit=True,
    interval=30
)
plt.show()
ani.save('animations/damped-sho.gif', writer='pillow', fps=30)
