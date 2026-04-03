# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# physics constant
wavelength = 500e-9
k = 2 * np.pi / wavelength
omega = 2 * np.pi * 2.0
slit_distance = 0.002
screen_width = 0.01
screen_distance = 1.0


y_screen = np.linspace(-screen_width/2, screen_width/2, 500)
r1_screen = np.sqrt(screen_distance**2 + (y_screen - slit_distance/2)**2)
r2_screen = np.sqrt(screen_distance**2 + (y_screen + slit_distance/2)**2)


x_2d = np.linspace(0.01, screen_distance, 200)
y_2d = np.linspace(-screen_width/2, screen_width/2, 200)
X_2d, Y_2d = np.meshgrid(x_2d, y_2d)
r1_2d = np.sqrt(X_2d**2 + (Y_2d - slit_distance/2)**2)
r2_2d = np.sqrt(X_2d**2 + (Y_2d + slit_distance/2)**2)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Young's Double Slit Interference", fontsize=16, fontweight='bold')

#panel 1
ax1.set_xlim(-5, 5)
ax1.set_ylim(-3, 3)
ax1.set_xlabel('Screen Position (mm)')
ax1.set_ylabel('Electric Field Amplitude')
ax1.set_title('Live Screen Amplitude')
ax1.grid(True)
wave_line, = ax1.plot([], [], 'b-', lw=2)

#panel 2
ax2.set_xlabel('Distance from Slits (m)')
ax2.set_ylabel('Vertical Position (m)')
ax2.set_title('2D Wave Propagation')
im = ax2.imshow(np.zeros_like(X_2d), extent=[0, screen_distance, -screen_width/2, screen_width/2],
                origin='lower', cmap='RdBu', aspect='auto', vmin=-2, vmax=2)

def init():
    wave_line.set_data([], [])
    im.set_array(np.zeros_like(X_2d))
    return wave_line, im


def update(frame):
    t = frame * 0.02


    wave1_s = np.exp(1j * (k * r1_screen - omega * t)) / r1_screen
    wave2_s = np.exp(1j * (k * r2_screen - omega * t)) / r2_screen
    wave_line.set_data(y_screen * 1000, np.real(wave1_s + wave2_s))


    wave1_2d = np.exp(1j * (k * r1_2d - omega * t)) / r1_2d
    wave2_2d = np.exp(1j * (k * r2_2d - omega * t)) / r2_2d
    im.set_array(np.real(wave1_2d + wave2_2d))

    return wave_line, im


ani = animation.FuncAnimation(fig, update, frames=100, init_func=init, blit=True, interval=20)
plt.tight_layout()
plt.show()
# ani.save('ds.gif', writer='pillow', fps=30)
# print("Saved 'ds.gif'")
