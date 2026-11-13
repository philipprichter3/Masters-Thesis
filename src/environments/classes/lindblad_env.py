
import numpy as np
import qutip as qt

from environments.qm_env import Environment


class Lindblad_Environment(Environment):

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def reset(self):

        psi0 = self.init_state()

        self.rho = qt.ket2dm(psi0)
        self.t = 0.0

# ======================================================================
# ============================= RUNTIME ============================
# ======================================================================

    def evolve_state(self, u, duration):
        """Lindblad evolution"""

        H = self.H0 + u * self.Hint
        c_ops = [self.c_op] if self.kappa > 0 else []

        result = qt.mesolve(
            H,
            self.rho,
            [0.0, duration],
            c_ops=c_ops,
            options={
                "matrix_form": True,
                
                "store_states": False,
                "store_final_state": True,
            }
        )

        self.rho = result.final_state

        return result.states
        
# ======================================================================
# ============================= RL UTILS ============================
# ======================================================================

    def observe(self, type : str = "default"):

        if type == "reduced":

            return self.observe_reduced()

        if type == "diagonal":

            return self.observe_diag()

        return self.observe_full()


    def observe_full(self):

        rho = self.rho.full().ravel()

        return np.concatenate((
            rho.real,
            rho.imag,
        )).astype(np.float32)

    def observe_reduced(self):

        rho = self.battery_state().full().ravel()

        return np.concatenate((
            rho.real, 
            rho.imag
        )).astype(np.float32)

    def observe_diag(self):

        rho = self.rho.full().ravel()

        diag_elements = np.diag(rho)

        return np.concatenate((
            diag_elements.real, 
            diag_elements.imag
        )).astype(np.float32)

    