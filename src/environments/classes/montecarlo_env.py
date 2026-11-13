
import numpy as np
import qutip as qt

from environments.qm_env import Environment


class MonteCarlo_Environment(Environment):

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(
        self,
        params
    ):

        self.ntraj = params["ntraj"]
        self.seed = params["seed"]

        super().__init__(
            params
        )

    def reset(self):

        psi0 = self.init_state()

        self.rng = np.random.default_rng(self.seed)

        self.psis = [
            psi0.copy() for _ in range(self.ntraj)
        ]

        self.rho = qt.ket2dm(psi0)

        self.jump_times = [
            [] for _ in range(self.ntraj)
        ]

        self.t = 0.0

# ======================================================================
# ============================= RUNTIME ============================
# ======================================================================

    def evolve_state(self, u, duration):
        """Monte Carlo quantum trajectory evolution"""

        H = self.H0 + u * self.Hint

        updated_psis = []

        if self.kappa > 0:

            for i, psi in enumerate(self.psis):

                result = qt.mcsolve(
                    H,
                    psi,
                    [0.0, duration],
                    c_ops=[self.c_op],
                    ntraj=1,
                    seeds=int(self.rng.integers(0, 2**32)),
                    options={
                        "keep_runs_results": True,
                        "store_states": True,
                        "progress_bar": False
                    },
                )

                psi_final = result.runs_states[0][-1]
                updated_psis.append(psi_final)

                self.jump_times[i].extend(
                    self.t + float(t_jump)
                    for t_jump in result.col_times[0]
                )

        else:

            for psi in self.psis:

                result = qt.sesolve(
                    H,
                    psi,
                    [0.0, duration],
                )

                updated_psis.append(result.states[-1])

        self.psis = updated_psis

        rho = qt.ket2dm(self.psis[0])
        for psi in self.psis[1:]:
            rho += qt.ket2dm(psi)

        self.rho = rho / self.ntraj

    def step(self, u):

        result = super().step(u)

        for space in ("symmetric", "full"):

            Wd = self.daemonic_ergotropy(space=space)

            result[f"daemonic_ergotropy_{space}"] = Wd

            result[f"daemonic_gain_{space}"] = (
                Wd - result[f"ergotropy_{space}"]
            )

        return result

# ============================================================
# ================= DAEMONIC ERGOTROPY ========================
# ============================================================

    def daemonic_ergotropy(self, space: str = "symmetric"):
        """average ergotropy of conditional battery states"""

        return float(np.mean([
            self.ergotropy(space=space, rho_b=psi.ptrace(1))
            for psi in self.psis
        ]))

# ============================================================
# ================= RL UTILS ========================
# ============================================================

    def observe(self, type : str = "default"):

        return