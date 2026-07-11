import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# parameters
N = 200               # grid points per side
h = 0.05              # grid spacing in metres
n_iter = 800          # sweeps per frame
n_frames = 100        # animation frames
r_blob = 2.5          # smear point charge over this radius (grid cells) to avoid singularity
q = 1.0               # charge magnitude

x = np.linspace(0, N * h, N)
y = np.linspace(0, N * h, N)
cx = N // 2           # grid-centre column
cy = N // 2           # grid-centre row
d_start = 0.7 * N     # in grid cells
d_end   = 0.12 * N

# blob
blob_r = int(4 * r_blob) + 1
bx, by = np.mgrid[-blob_r:blob_r+1, -blob_r:blob_r+1]
blob = np.exp(-(bx**2 + by**2) / (2 * r_blob**2))
blob /= blob.sum()    # normalise


def make_rho(sep):
    rho = np.zeros((N, N))
    for sign, col in [(+1, cx + sep // 2), (-1, cx - sep // 2)]:
        r0, c0 = cy, col
        r_lo = max(r0 - blob_r, 0);  r_hi = min(r0 + blob_r + 1, N)
        c_lo = max(c0 - blob_r, 0);  c_hi = min(c0 + blob_r + 1, N)
        br_lo = r_lo - (r0 - blob_r); br_hi = br_lo + (r_hi - r_lo)
        bc_lo = c_lo - (c0 - blob_r); bc_hi = bc_lo + (c_hi - c_lo)
        rho[r_lo:r_hi, c_lo:c_hi] += sign * q * blob[br_lo:br_hi, bc_lo:bc_hi]
    return rho


def gauss_seidel(phi, rho, n_iter):
    h2 = h * h
    phi = phi.copy()

    ii, jj = np.indices((N, N))
    red   = (ii + jj) % 2 == 0
    black = ~red

    interior = np.zeros((N, N), dtype=bool)
    interior[1:-1, 1:-1] = True

    red_int   = red   & interior
    black_int = black & interior

    for _ in range(n_iter):
        inner_red   = red_int[1:-1, 1:-1]
        inner_black = black_int[1:-1, 1:-1]

        nbr_sum = (phi[0:-2, 1:-1] + phi[2:, 1:-1] +
                   phi[1:-1, 0:-2] + phi[1:-1, 2:])

        update = (nbr_sum + rho[1:-1, 1:-1] * h2) / 4.0
        phi[1:-1, 1:-1] = np.where(inner_red, update, phi[1:-1, 1:-1])

        nbr_sum = (phi[0:-2, 1:-1] + phi[2:, 1:-1] +
                   phi[1:-1, 0:-2] + phi[1:-1, 2:])

        update = (nbr_sum + rho[1:-1, 1:-1] * h2) / 4.0
        phi[1:-1, 1:-1] = np.where(inner_black, update, phi[1:-1, 1:-1])

    return phi

# precomputing frames
separations = np.linspace(d_start, d_end, n_frames).astype(int)
phi_frames = []
rho_frames = []

phi = np.zeros((N, N))

print("Precomputing frames...")
for k, sep in enumerate(separations):
    rho = make_rho(sep)
    phi = gauss_seidel(phi, rho, n_iter)
    phi_frames.append(phi.copy())
    rho_frames.append(rho.copy())
    print(f"  Frame {k+1}/{n_frames}, separation = {sep * h:.2f} m")

print("Done. Launching animation.")

# matplotlib
fig, ax = plt.subplots(figsize=(7, 7))

phi0 = phi_frames[0]
vmax = np.percentile(np.abs(phi0), 98)

contourf_plot = ax.contourf(x, y, phi_frames[0],
                             levels=60, cmap='RdBu_r',
                             vmin=-vmax, vmax=vmax)

Ex0 = -np.gradient(phi_frames[0], h, axis=1)
Ey0 = -np.gradient(phi_frames[0], h, axis=0)

ds = 4
xd, yd = x[::ds], y[::ds]
stream_plot = ax.streamplot(xd, yd, Ex0[::ds, ::ds], Ey0[::ds, ::ds], # slicing so plot is not jumbled
                             color='k', linewidth=0.7, density=1.8,
                             arrowsize=0.8)

title = ax.set_title(f'Dipole Separation: {separations[0] * h:.2f} m', pad=10)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')

cbar = fig.colorbar(contourf_plot, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label('Electric Potential (V)')

# animation
def update(frame):
    global stream_plot   # streamplot must be removed and redrawn

    phi = phi_frames[frame]
    sep_m = separations[frame] * h

    Ex = -np.gradient(phi, h, axis=1)
    Ey = -np.gradient(phi, h, axis=0)

    for coll in ax.collections:
        coll.remove()
    vmax = np.percentile(np.abs(phi), 98)
    ax.contourf(x, y, phi, levels=60, cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    for patch in ax.patches:
        patch.remove()

    for line in ax.lines:
        line.remove()

    stream_plot = ax.streamplot(xd, yd, Ex[::ds, ::ds], Ey[::ds, ::ds],
                                 color='k', linewidth=0.7, density=1.8,
                                 arrowsize=0.8)

    x_pos = np.array([cx + separations[frame] // 2,
                       cx - separations[frame] // 2]) * h
    y_pos = np.array([cy, cy]) * h
    colors = ['#ff4444', '#4488ff']   # red = +q, blue = −q

    for xp, yp, c in zip(x_pos, y_pos, colors):
        ax.plot(xp, yp, 'o', markersize=11, color=c,
                markeredgecolor='k', markeredgewidth=0.8, zorder=5)

    title.set_text(f'Dipole Separation: {sep_m:.2f} m')
    return []

# creating animation
ani = anim.FuncAnimation(fig, update, frames=n_frames,
                         interval=120, blit=False, repeat=True)

plt.tight_layout()
plt.show()

# saving last frame state for particle tracker code
final_phi = phi_frames[-1]
Ex_final = -np.gradient(final_phi, h, axis=1)
Ey_final = -np.gradient(final_phi, h, axis=0)

print("Exporting data...")
np.savez_compressed('dipole_field.npz',
                    Ex=Ex_final,
                    Ey=Ey_final,
                    N=N,
                    h=h)
print("Saved to dipole_field.npz.")


# update(30)
# plt.savefig("image-output/field-solver.pdf", bbox_inches="tight")
