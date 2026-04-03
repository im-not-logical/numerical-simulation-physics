# import matplotlib
# matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#creating meshgrid
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Dynamic Electric and Magnetic Fields', fontsize=16, fontweight='bold')

# Panel 1
ax1.set_xlim(-3, 3)
ax1.set_ylim(-3, 3)
ax1.set_aspect('equal')
ax1.set_title('Oscillating Electric Dipole')
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
ax1.grid(True)

# Panel 2
ax2.set_xlim(-3, 3)
ax2.set_ylim(-3, 3)
ax2.set_aspect('equal')
ax2.set_title('Alternating Current (AC) Magnetic Field')
ax2.set_xlabel('X Position')
ax2.grid(True)


q_E = ax1.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y), color='b', pivot='mid', scale=25)
dot_pos, = ax1.plot([], [], 'ro', markersize=10, label='+q')
dot_neg, = ax1.plot([], [], 'bo', markersize=10, label='-q')
ax1.legend(loc='upper right')

q_B = ax2.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y), color='g', pivot='mid', scale=25)
dot_I, = ax2.plot([0], [0], 'ko', markersize=12, label='Wire (I)')
ax2.legend(loc='upper right')

hud = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, va='top', ha='left',
               fontfamily='monospace', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

def init():
    q_E.set_UVC(np.zeros_like(X), np.zeros_like(Y))
    q_B.set_UVC(np.zeros_like(X), np.zeros_like(Y))
    dot_pos.set_data([], [])
    dot_neg.set_data([], [])
    hud.set_text('')
    return q_E, q_B, dot_pos, dot_neg, dot_I, hud


def update(frame):
    t = frame * 0.1

    d = 1.0 + 0.5 * np.sin(t)
    pos1, pos2 = [-d, 0], [d, 0]


    r1 = np.sqrt((X-pos1[0])**2 + (Y-pos1[1])**2) + 0.1
    r2 = np.sqrt((X-pos2[0])**2 + (Y-pos2[1])**2) + 0.1
    Ex = 1 * (X-pos1[0])/r1**3 - 1 * (X-pos2[0])/r2**3
    Ey = 1 * (Y-pos1[1])/r1**3 - 1 * (Y-pos2[1])/r2**3


    E_mag = np.sqrt(Ex**2 + Ey**2)
    q_E.set_UVC(Ex/E_mag, Ey/E_mag)
    dot_pos.set_data([pos1[0]], [pos1[1]])
    dot_neg.set_data([pos2[0]], [pos2[1]])


    I = np.sin(t)
    r_B = X**2 + Y**2 + 0.1
    Bx = -Y * I / r_B
    By = X * I / r_B
    B_mag = np.sqrt(Bx**2 + By**2) + 1e-5
    q_B.set_UVC(Bx/B_mag, By/B_mag)

    # Change wire colour
    dot_I.set_color('r' if I > 0 else 'b')

    hud.set_text(f"Dipole Separation: {2*d:.2f} m\nAC Current: {I:.2f} A")

    return q_E, q_B, dot_pos, dot_neg, dot_I, hud


ani = animation.FuncAnimation(fig, update, frames=1000, init_func=init, blit=True, interval=20)
plt.tight_layout()
plt.show()
# ani.save('fields.gif', writer='pillow', fps=30)
# print("Saved 'fields.gif'")
