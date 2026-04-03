# import matplotlib
# matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# physics constant
m1, m2 = 2.0, 1.0
v1_i, v2_i = 3.0, -1.5
x1_0, x2_0 = -5.0, 5.0
e = 0.8 # Coefficient of restitution

t_collide = (x2_0 - x1_0) / (v1_i - v2_i)
x_collide = x1_0 + v1_i * t_collide

v1_f = ((m1 - e*m2)*v1_i + m2*(1 + e)*v2_i) / (m1 + m2)
v2_f = (m1*(1 + e)*v1_i + (m2 - e*m1)*v2_i) / (m1 + m2)

t_eval = np.linspace(0, t_collide + 2, 2000)
x1_data = np.where(t_eval < t_collide, x1_0 + v1_i*t_eval, x_collide + v1_f*(t_eval-t_collide))
x2_data = np.where(t_eval < t_collide, x2_0 + v2_i*t_eval, x_collide + v2_f*(t_eval-t_collide))
v1_data = np.where(t_eval < t_collide, v1_i, v1_f)
v2_data = np.where(t_eval < t_collide, v2_i, v2_f)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle(f'1D Collision Dynamics (e = {e})', fontsize=16, fontweight='bold')

# Panel 1
ax1.set_xlim(-6, 6)
ax1.set_ylim(-1, 1)
ax1.set_xlabel('Position (m)')
ax1.set_title('Physical View')
ax1.grid(True)
ax1.get_yaxis().set_visible(False) # Hide Y axis for 1D motion

# Panel 2
ax2.set_xlim(0, t_collide + 2)
ax2.set_ylim(-6, 6)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Position (m)')
ax2.set_title('Position vs. Time')
ax2.grid(True)


ball1, = ax1.plot([], [], 'bo', markersize=20, label='m1 (2.0 kg)')
ball2, = ax1.plot([], [], 'ro', markersize=14, label='m2 (1.0 kg)')
ax1.legend(loc='upper right')

line_x1, = ax2.plot([], [], 'b-', lw=2)
line_x2, = ax2.plot([], [], 'r-', lw=2)
dot_x1, = ax2.plot([], [], 'bo')
dot_x2, = ax2.plot([], [], 'ro')

hud = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, va='top', ha='left',
               fontfamily='monospace', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

def init():
    ball1.set_data([], [])
    ball2.set_data([], [])
    line_x1.set_data([], [])
    line_x2.set_data([], [])
    dot_x1.set_data([], [])
    dot_x2.set_data([], [])
    hud.set_text('')
    return ball1, ball2, line_x1, line_x2, dot_x1, dot_x2, hud


def update(frame):
    ball1.set_data([x1_data[frame]], [0])
    ball2.set_data([x2_data[frame]], [0])

    line_x1.set_data(t_eval[:frame], x1_data[:frame])
    line_x2.set_data(t_eval[:frame], x2_data[:frame])

    dot_x1.set_data([t_eval[frame]], [x1_data[frame]])
    dot_x2.set_data([t_eval[frame]], [x2_data[frame]])

    p_total = m1 * v1_data[frame] + m2 * v2_data[frame]
    ke_total = 0.5 * m1 * v1_data[frame]**2 + 0.5 * m2 * v2_data[frame]**2
    hud.set_text(f"System Momentum: {p_total:5.2f} kg*m/s\nSystem Kinetic E: {ke_total:5.2f} J")

    return ball1, ball2, line_x1, line_x2, dot_x1, dot_x2, hud

frames = int(len(t_eval))
ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=2)
plt.tight_layout()
plt.show()
# ani.save('collision.gif', writer='pillow', fps=30)
# print("Saved 'collision.gif'")
