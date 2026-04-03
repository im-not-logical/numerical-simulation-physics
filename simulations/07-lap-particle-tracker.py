import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator

# importing field solver data
data = np.load("dipole_field.npz")
Ex_grid = data["Ex"]
Ey_grid = data["Ey"]
N  = int(data["N"])
h  = float(data["h"])

xs = np.linspace(0, (N - 1) * h, N)
ys = np.linspace(0, (N - 1) * h, N)

Ex_interp = RegularGridInterpolator((ys, xs), Ex_grid, method="linear", bounds_error=False, fill_value=0.0)
Ey_interp = RegularGridInterpolator((ys, xs), Ey_grid, method="linear", bounds_error=False, fill_value=0.0)

# parameters and initial conditions
q_over_m = 5e4       # C/kg
x0, y0   = 3 , 6  # m
vx0, vy0 = 20, 0 # m/s
t_start, t_end = 0.0, 0.2   # s

# equations of motion
def lorentz_ode(t, s):
    x, y, vx, vy = s
    pt = np.array([[y, x]]) # y,x to match interpolator

    field_scale = 2e3
    ax = q_over_m * (Ex_interp(pt)[0] * field_scale)
    ay = q_over_m * (Ey_interp(pt)[0] * field_scale)
    return [vx, vy, ax, ay]

# integrate
domain = [0.0, (N - 1) * h]

def hit_boundary(t, s):
    x, y = s[0], s[1]
    margin = h
    inside = (domain[0] + margin < x < domain[1] - margin and
              domain[0] + margin < y < domain[1] - margin)
    return float(inside) - 0.5

hit_boundary.terminal  = True
hit_boundary.direction = -1      # trigger on exit only

sol = solve_ivp(
    lorentz_ode,
    [t_start, t_end],
    [x0, y0, vx0, vy0],
    method="RK45",
    events=hit_boundary,
    dense_output=True,
    rtol=1e-8, atol=1e-10
)

t_frames = np.linspace(t_start, sol.t[-1], 400)
traj = sol.sol(t_frames)
px, py = traj[0], traj[1]

# matplotlib
fig, ax = plt.subplots(figsize=(7, 7))

stride = 5
X, Y = np.meshgrid(xs[::stride], ys[::stride])
U = Ex_grid[::stride, ::stride]
V = Ey_grid[::stride, ::stride]

E_mag = np.hypot(U, V)
ax.streamplot(
    X, Y, U, V,
    color=np.log1p(E_mag),   # log scale: avoids near-source saturation
    cmap="viridis",          # robust colormap for white backgrounds
    linewidth=0.8,
    density=1.4,
    arrowsize=0.8,
)

ax.set_xlim(*domain)
ax.set_ylim(*domain)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Particle Trajectory in Electric Field")

# animation
trail_len = 60 # length of trail

trail_line, = ax.plot([], [], color="crimson", lw=1.5, alpha=0.7)
particle_dot, = ax.plot([], [], "o", color="crimson", ms=6,
                        markeredgecolor="darkred", markeredgewidth=1.2, zorder=5)
time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                    color="black", fontsize=9, va="top")

def init():
    trail_line.set_data([], [])
    particle_dot.set_data([], [])
    time_text.set_text("")
    return trail_line, particle_dot, time_text

def animate(i):
    lo = max(0, i - trail_len)
    trail_line.set_data(px[lo:i+1], py[lo:i+1])
    particle_dot.set_data([px[i]], [py[i]])
    time_text.set_text("t = {:.4f} s".format(t_frames[i]))
    return trail_line, particle_dot, time_text

# create animation
ani = animation.FuncAnimation(
    fig, animate,
    frames=len(t_frames),
    init_func=init,
    interval=20,      # ms between frames — tweak to taste
    blit=True
)

plt.tight_layout()
plt.show()
# ani.save('animations/particle-tracker.gif', writer='pillow', fps=30)
