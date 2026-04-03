# import matplotlib
# matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

#arrays
q, m = -1.0, 1.0
E = np.array([0.0, 0.1, 0.0])
B = np.array([0.0, 0.0, 1.0])

def lorentz_deriv(t, y):
    pos = y[:3]
    v = y[3:]
    acc = (q / m) * (E + np.cross(v, B))
    return np.concatenate((v, acc))

y0 = [0, 0, 0, 0.5, 0.0, 0.1]
t_eval = np.linspace(0, 30, 3000)

solution = solve_ivp(lorentz_deriv, [0, 30], y0, t_eval=t_eval, method='RK45')
x, y, z = solution.y[0], solution.y[1], solution.y[2]


fig = plt.figure(figsize=(14, 6))
fig.suptitle('Lorentz Force: Cyclotron Motion and E-B Drift', fontsize=16, fontweight='bold')

# Panel 1
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.set_xlim(0, 3)
ax1.set_ylim(0, 3)
ax1.set_zlim(0, 3)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Trajectory')

# Panel 2
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_aspect('equal')
ax2.set_xlim(0, 4)
ax2.set_ylim(-1, 1)
ax2.set_xlabel('X Position')
ax2.set_ylabel('Y Position')


ax2.set_title(r'XY Plane Projection ($\vec{E} \times \vec{B}$ Drift)')
ax2.grid(True)


trail_3d, = ax1.plot([], [], [], 'b-', alpha=0.6)
particle_3d, = ax1.plot([], [], [], 'bo', markersize=6)

trail_2d, = ax2.plot([], [], 'r-', lw=2, alpha=0.6)
particle_2d, = ax2.plot([], [], 'ro', markersize=6)

def init():
    trail_3d.set_data([], [])
    trail_3d.set_3d_properties([])
    particle_3d.set_data([], [])
    particle_3d.set_3d_properties([])

    trail_2d.set_data([], [])
    particle_2d.set_data([], [])
    return trail_3d, particle_3d, trail_2d, particle_2d


def update(frame):

    trail_3d.set_data(x[:frame], y[:frame])
    trail_3d.set_3d_properties(z[:frame])
    particle_3d.set_data([x[frame]], [y[frame]])
    particle_3d.set_3d_properties([z[frame]])


    trail_2d.set_data(x[:frame], y[:frame])
    particle_2d.set_data([x[frame]], [y[frame]])

    # Slowly rotate the 3D frame
    ax1.view_init(elev=20, azim=frame * 0.3)

    return trail_3d, particle_3d, trail_2d, particle_2d

ani = animation.FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=2)
plt.tight_layout()
plt.show()
# ani.save('lorentz.gif', writer='pillow', fps=30)
