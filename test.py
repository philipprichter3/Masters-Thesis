
import matplotlib.pyplot as plt
import numpy as np
from qutip import basis, destroy, mcsolve, mesolve

kappa = 1.0
a = destroy(2)
n_op = a.dag() * a

H = n_op
psi0 = basis(2, 1)
tlist = np.linspace(0, 5, 300)
c_ops = [np.sqrt(kappa) * a]

# Deterministic master equation
me = mesolve(H, psi0, tlist, c_ops, e_ops=[n_op])

# Stochastic quantum trajectories
mc = mcsolve(
    H, psi0, tlist, c_ops,
    e_ops=[n_op],
    ntraj=200,
    options={"keep_runs_results": True},
    seeds=42
)

# Plot some individual trajectories
for trajectory in mc.runs_expect[0][:8]:
    plt.plot(tlist, trajectory, alpha=0.4)

plt.plot(tlist, mc.average_expect[0],
         label="200-trajectory average", linewidth=2)

plt.plot(tlist, me.expect[0], "--",
         label="Lindblad result", linewidth=2)

plt.xlabel("Time")
plt.ylabel("Cavity photon number")
plt.legend()
plt.show()
