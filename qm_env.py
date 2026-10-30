from abc import ABC, abstractmethod

import numpy as np
import qutip as qt


class Environment(ABC):

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(
        self,
        params
    ):
        # Physical parameters
        self.wc = params["wc"]
        self.wb = params["wb"]

        self.Nc = params["Nc"]            # Initial cavity photon number
        self.Nb = params["Nb"]            # Number of TLS

        self.Nc_max = (
            (20 if self.Nb < 5 else 4 * self.Nb)
            if not "Nc_max" in params else params["Nc_max"]
        )

        self.kappa = params["kappa"]
        self.dt = params["dt"]
        self.tau = params["tau"]

        self.init_operators()
        self.reset()

    def init_state(self):

        psi_c = qt.basis(self.Nc_max + 1, self.Nc)
        psi_b = qt.basis(self.Nb + 1, self.Nb)

        return qt.tensor(psi_c, psi_b)
    
    def init_operators(self):

        Ic = qt.qeye(self.Nc_max + 1)
        Ib = qt.qeye(self.Nb + 1)

        a_c = qt.destroy(self.Nc_max + 1)

        Jx = qt.jmat(self.Nb / 2, "x")
        Jz = qt.jmat(self.Nb / 2, "z")

        a = qt.tensor(a_c, Ib)

        self.n_photons_op = a.dag() * a

        self.c_op = np.sqrt(2 * self.kappa) * a

        self.Hb_bare = self.wb * (
            Jz + (self.Nb / 2) * Ib
        )

        self.Hb = qt.tensor(Ic, self.Hb_bare)
        self.Hc = self.wc * a.dag() * a

        self.H0 = self.Hb + self.Hc
        self.Hint = 2 * qt.tensor(
            a_c + a_c.dag(), Jx
        )

    @abstractmethod
    def reset(self):

        pass

# ======================================================================
# ============================= UTILS ============================
# ======================================================================

    def battery_state(self):

        return self.rho.ptrace(1)

    def energy(self):

        return float(np.real(qt.expect(self.Hb, self.rho)))

    def ergotropy(self, space="symmetric"):
        """
        Ergotropy of the reduced battery.

        space:
            'symmetric' : arbitrary unitaries in j = Nb/2 sector
            'full'      : arbitrary unitaries on all Nb TLS
        """

        rho_b = self.battery_state()

        eigenvalues = np.linalg.eigvalsh(rho_b.full())
        eigenvalues = np.sort(
            np.clip(eigenvalues.real, 0, None)
        )[::-1]
        eigenvalues /= eigenvalues.sum()

        stored_energy = self.energy()

        if space == "symmetric":
            energies = self.wb * np.arange(self.Nb + 1)
            passive_energy = np.dot(eigenvalues, energies)

        elif space == "full":
            passive_energy = self.wb * (1 - eigenvalues[0])

        else:
            raise ValueError("Unknown ergotropy space")

        return max(0.0, float(stored_energy - passive_energy))


    def purity(self):

        rho_b = self.battery_state()

        return float(np.real((rho_b * rho_b).tr()))

    def joint_purity(self):

        return float(np.real((self.rho * self.rho).tr()))

    def photon_count(self):

        return float(np.real(qt.expect(self.n_photons_op, self.rho)))

# ======================================================================
# ============================= RUNTIME ============================
# ======================================================================

    @abstractmethod
    def evolve_state(self, u, duration):

        pass

    def step(self, u):
        """Evolve the system for one time step."""

        if self.t >= self.tau:
            raise RuntimeError("Simulation finished")

        duration = min(self.dt, self.tau - self.t)

        self.evolve_state(u, duration)
        self.t = min(self.t + duration, self.tau)

        return {
            "time": self.t,
            "energy": self.energy(),
            "ergotropy_symmetric": self.ergotropy(space = "symmetric"),
            "ergotropy_full": self.ergotropy(space = "full"),
            "purity": self.purity(),
            "joint_purity": self.joint_purity(),
            "photon_count": self.photon_count(),
        }

    def run(self, u):

        history = []

        while self.t < self.tau:
            result = self.step(u)
            history.append(result)

        return history