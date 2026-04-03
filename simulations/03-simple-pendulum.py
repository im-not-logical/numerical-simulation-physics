import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

# parameters
g      = 9.81   # m/s^2
L      = 1.0    # m
omega0 = np.sqrt(g / L)   # natural frequency
T_lin  = 2 * np.pi / omega0  # period under small angle approximation

# Initial conditions
theta0_deg = 75.0
theta0     = np.radians(theta0_deg)
dtheta0    = 0.0   # rad/s, starting from rest

t_start = 0.0
t_end   = 5 * T_lin
t_eval  = np.linspace(t_start, t_end, 2000)

# equations of motion
def nonlinear_rhs(t, y):
    theta, dtheta = y
    return [dtheta, -(g / L) * np.sin(theta)]

def linear_rhs(t, y):
    theta, dtheta = y
    return [dtheta, -(g / L) * theta]

y0 = [theta0, dtheta0]

sol_nl = solve_ivp(nonlinear_rhs, [t_start, t_end], y0,
                   method='RK45', t_eval=t_eval, rtol=1e-9, atol=1e-9)
sol_lin = solve_ivp(linear_rhs,   [t_start, t_end], y0,
                    method='RK45', t_eval=t_eval, rtol=1e-9, atol=1e-9)

t         = sol_nl.t
theta_nl  = sol_nl.y[0]
omega_nl  = sol_nl.y[1]    # dtheta/dt

theta_lin = sol_lin.y[0]
omega_lin = sol_lin.y[1]

# analytic solution
theta_analytic = theta0 * np.cos(omega0 * t)

# energy
E_nl  = 0.5 * (L * omega_nl)**2  / (g * L) + (1 - np.cos(theta_nl))
E_lin = 0.5 * (L * omega_lin)**2 / (g * L) + (1 - np.cos(theta_lin))
E_sho = 0.5 * (omega_lin**2 + omega0**2 * theta_lin**2) / omega0**2

# matplotlib
fig = plt.figure(figsize=(14, 11))
fig.suptitle(f'Simple Pendulum — $\\theta_0 = {theta0_deg:.0f}°$', fontsize=16)

gs = fig.add_gridspec(3, 2, width_ratios=[1.5, 1])

axes = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[2, 0])
]
ax_anim = fig.add_subplot(gs[:, 1]) # Spans all 3 rows in the second column

# theta vs time comparison
ax = axes[0]
ax.plot(t / T_lin, np.degrees(theta_nl),  label='Nonlinear (RK45)',   color='steelblue', lw=2)
ax.plot(t / T_lin, np.degrees(theta_lin), label='Linear approx (RK45)', color='tomato', lw=2, ls='--')
ax.plot(t / T_lin, np.degrees(theta_analytic), label='Linear (analytic)',
        color='black', lw=1, ls=':', alpha=0.7)
ax.set_xlabel('Time  /  $T_{\\mathrm{lin}}$')
ax.set_ylabel('$\\theta$  (degrees)')
ax.set_title('Angular displacement')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# phase space
ax = axes[1]
ax.plot(np.degrees(theta_nl),  omega_nl,  color='steelblue', lw=1.5, label='Nonlinear')
ax.plot(np.degrees(theta_lin), omega_lin, color='tomato',    lw=1.5, ls='--', label='Linear')
ax.set_xlabel('$\\theta$  (degrees)')
ax.set_ylabel('$\\dot{\\theta}$  (rad/s)')
ax.set_title('Phase space ($\\theta$, $\\dot{\\theta}$)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# energy conservation
ax = axes[2]
ax.plot(t / T_lin, (E_nl  - E_nl[0])  / E_nl[0]  * 100, color='steelblue', lw=1.5, label='Nonlinear')
ax.plot(t / T_lin, (E_lin - E_lin[0]) / E_lin[0] * 100, color='tomato',    lw=1.5, ls='--', label='Linear')
ax.set_xlabel('Time  /  $T_{\\mathrm{lin}}$')
ax.set_ylabel('Relative energy drift  (%)')
ax.set_title('Energy conservation check')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.ticklabel_format(axis='y', style='sci', scilimits=(-6, -6))

# animation
ax_anim.set_aspect('equal')
ax_anim.set_xlim(-1.2 * L, 1.2 * L)
ax_anim.set_ylim(-1.2 * L, 0.2 * L)
ax_anim.set_title('Pendulum Simulation - Nonlinear vs Approximation')
ax_anim.grid(True, alpha=0.3)
ax_anim.plot([0], [0], 'ko', markersize=6) # origin or pivot point

rod_nl, = ax_anim.plot([], [], '-', color='steelblue', lw=3, label='Nonlinear')
bob_nl, = ax_anim.plot([], [], 'o', color='steelblue', markersize=10)

rod_lin, = ax_anim.plot([], [], '--', color='tomato', lw=2, label='Linear')
bob_lin, = ax_anim.plot([], [], 'o', color='tomato', markersize=8)

ax_anim.legend(loc='upper right', fontsize=10)

def init():
    rod_nl.set_data([], [])
    bob_nl.set_data([], [])
    rod_lin.set_data([], [])
    bob_lin.set_data([], [])
    return rod_nl, bob_nl, rod_lin, bob_lin

def update(frame):
    # Coordinates for Nonlinear pendulum
    x_nl = L * np.sin(theta_nl[frame])
    y_nl = -L * np.cos(theta_nl[frame])

    # Coordinates for Linear pendulum
    x_lin = L * np.sin(theta_lin[frame])
    y_lin = -L * np.cos(theta_lin[frame])

    # Update data for rod and bob
    rod_nl.set_data([0, x_nl], [0, y_nl])
    bob_nl.set_data([x_nl], [y_nl])

    rod_lin.set_data([0, x_lin], [0, y_lin])
    bob_lin.set_data([x_lin], [y_lin])

    return rod_nl, bob_nl, rod_lin, bob_lin

frame_stride = 4
frames_to_draw = np.arange(0, len(t), frame_stride)

ani = animation.FuncAnimation(
    fig, update, frames=frames_to_draw, init_func=init,
    blit=True, interval=20, repeat=True
)

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust to fit the suptitle
plt.show()
ani.save('animations/simple-pendulum.gif', writer='pillow', fps=30)
