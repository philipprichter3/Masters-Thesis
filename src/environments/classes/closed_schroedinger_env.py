
import numpy as np
import qutip as qt

from environments.qm_env import Environment


class CLosed_Schroedinger_Env(Environment):

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def reset(self):

        self.psi = self.init_state()

        self.rho = qt.ket2dm(self.psi)
        self.t = 0.0

# ======================================================================
# ============================= RUNTIME ============================
# ======================================================================

    def evolve_state(self, u, duration):
        """unitary schroedinger evolution"""

        H = self.H0 + u * self.Hint

        result = qt.sesolve(
            H,
            self.psi,
            [0.0, duration],
        )

        self.psi = result.states[-1]

        self.rho = qt.ket2dm(self.psi)

# ============================================================
# ================= RL UTILS ========================
# ============================================================

    def observe(self, type : str = "default"):

        psi = self.psi.full().ravel()

        return np.concatenate((
            psi.real,
            psi.imag,
        )).astype(np.float32)