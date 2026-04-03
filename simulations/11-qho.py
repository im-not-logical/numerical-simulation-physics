import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# parameters
hbar = 1.0
m    = 1.0
omega = 1.0

x_left  = -8.0
x_right  = 8.0
n_points = 2000
x_eval = np.linspace(x_left, x_right, n_points)

# equations of motion
def schrodinger(x, y, E):
    psi, phi = y
    V = 0.5 * m * omega**2 * x**2
    dphi_dx = (2 * m / hbar**2) * (V - E) * psi
    return [phi, dphi_dx]

def shoot(E, n):
    if n % 2 == 0:
        y0 = [1e-6, 0.0]
    else:
        y0 = [0.0, 1e-6]

    sol = solve_ivp(
        schrodinger,
        [x_left, x_right],
        y0,
        args=(E,),
        method='RK45',
        t_eval=x_eval,
        dense_output=False,
        rtol=1e-10,
        atol=1e-12
    )
    return sol.y[0, -1]

def find_eigenstate(n):
    E_exact = n + 0.5
    dE = 0.4

    E_lo, E_hi = E_exact - dE, E_exact + dE

    E_eigen = brentq(shoot, E_lo, E_hi, args=(n,), xtol=1e-10) # finds root of shoot(E)

    if n % 2 == 0:
        y0 = [1e-6, 0.0]
    else:
        y0 = [0.0, 1e-6]

    sol = solve_ivp(
        schrodinger,
        [x_left, x_right],
        y0,
        args=(E_eigen,),
        method='RK45',
        t_eval=x_eval,
        rtol=1e-10,
        atol=1e-12
    )

    psi = sol.y[0]

    # normalize
    norm = np.trapezoid(psi**2, x_eval)
    psi /= np.sqrt(norm)

    return E_eigen, psi

# matplotlib
n_states = 5
fig, ax = plt.subplots(figsize=(8, 10))

V_plot = 0.5 * m * omega**2 * x_eval**2  # potential for reference
ax.plot(x_eval, V_plot, 'k--', lw=1.5, alpha=0.6, label='V(x)')

for n in range(n_states):
    E_n, psi_n = find_eigenstate(n)
    scale = 0.6 # scaling for clarity
    ax.plot(x_eval, psi_n * scale + E_n, lw=2.0, label=f'$\\psi_{n}$ (E={E_n:.1f})')
    ax.axhline(E_n, color='gray', lw=1.0, ls=':', alpha=0.5) # indicates enerly level

    print(f"n={n}: E_numerical={E_n:.6f}, E_exact={n + 0.5:.6f}, error={abs(E_n - (n+0.5)):.2e}")

ax.set_xlim(-5, 5)
ax.set_ylim(-0.5, n_states + 0.5)

ax.set_xlabel('x (natural units)', fontsize=12)
ax.set_ylabel('Energy E', fontsize=12)
ax.set_title('QHO Eigenstates', fontsize=14)
ax.legend(loc='upper right', framealpha=0.9)
plt.tight_layout()
# plt.savefig('qho_eigenstates.png', dpi=150)
plt.show()
