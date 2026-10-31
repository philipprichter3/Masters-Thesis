
import numpy as np
import qutip as qt
from qm_env import Environment


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

# ============================================================
# ================= DAEMONIC ERGOTROPY ========================
# ============================================================

    def _ergotropy_of_battery_state(self, rho_b, space="symmetric"):
        """Ergotropy of an individual reduced battery state."""

        eigenvalues = np.linalg.eigvalsh(rho_b.full())
        eigenvalues = np.sort(
            np.clip(eigenvalues.real, 0, None)
        )[::-1]

        eigenvalues /= eigenvalues.sum()

        # Battery energy in this conditional state
        stored_energy = float(
            np.real(qt.expect(self.Hb_bare, rho_b))
        )

        if space == "symmetric":
            energies = self.wb * np.arange(self.Nb + 1)
            passive_energy = np.dot(eigenvalues, energies)

        elif space == "full":
            passive_energy = self.wb * (1 - eigenvalues[0])

        else:
            raise ValueError("Unknown ergotropy space")

        return max(0.0, float(stored_energy - passive_energy))

    def daemonic_ergotropy(self, space="symmetric"):
        """Average ergotropy of conditional battery states."""

        ergotropies = []

        for psi in self.psis:

            # Trace out the cavity for this trajectory
            rho_b = psi.ptrace(1)

            W = self._ergotropy_of_battery_state(
                rho_b, space=space
            )

            ergotropies.append(W)

        return float(np.mean(ergotropies))

# ======================================================================
# ============================= RUNTIME ============================
# ======================================================================

    def evolve_state(self, u, duration):
        """Monte Carlo quantum trajectory evolution."""

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