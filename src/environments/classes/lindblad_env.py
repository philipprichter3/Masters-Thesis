
import numpy as np
import qutip as qt

from environments.qm_env import Environment


class Lindblad_Environment(Environment):

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def reset(self):

        psi0 = self._init_state()

        self.rho = qt.ket2dm(psi0)
        self.t = 0.0

# ======================================================================
# ============================= RUNTIME ============================
# ======================================================================

    def _evolve_state(self, u, duration):
        """Lindblad evolution"""

        H = self.H0 + u * self.Hint
        c_ops = [self.c_op] if self.kappa > 0 else []

        result = qt.mesolve(
            H,
            self.rho,
            [0.0, duration],
            c_ops=c_ops,
        )

        self.rho = result.states[-1]
        
# ======================================================================
# ============================= RL UTILS ============================
# ======================================================================

    def observe(self):
        
        rho = self.rho.full().ravel()

        return np.concatenate((
            rho.real,
            rho.imag,
        )).astype(np.float32)