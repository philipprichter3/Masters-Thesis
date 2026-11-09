from abc import ABC, abstractmethod

import numpy as np
import qutip as qt


class Environment(ABC):

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(self, params : dict):
        
        self.rho = None
        
        # Physical parameters
        self.wc = params["wc"]
        self.wb = params["wb"]

        self.Nc = params["Nc"]            # Initial cavity photon number
        self.Nb = params["Nb"]            # Number of TLS

        self.Nc_max = \
            params["Nc_max"] if "Nc_max" in params \
            else \
                20 if self.Nc < 5 \
                else 4 * self.Nc

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
# ============================= RUNTIME ============================
# ======================================================================

    @abstractmethod
    def evolve_state(self, u, duration):

        pass

    def step(self, u):

        if self.t >= self.tau - 1e-12:
            raise RuntimeError("Charging episode already finished; call reset().")

        duration = min(self.dt, self.tau - self.t)

        E_i = self.energy();
        erg_sym_i = self.ergotropy(space = "symmetric")
        erg_full_i = self.ergotropy(space = "full")

        self.evolve_state(u, duration)

        E_f = self.energy();
        erg_sym_f = self.ergotropy(space = "symmetric")
        erg_full_f = self.ergotropy(space = "full")

        self.t = min(self.t + duration, self.tau)

        return {
            "time": self.t,

            "energy": E_f,
            "d_energy": E_f - E_i,

            "ergotropy_symmetric": erg_sym_f,
            "d_ergotropy_symmetric": erg_sym_f - erg_sym_i,

            "ergotropy_full": erg_full_f,
            "d_ergotropy_full": erg_full_f - erg_full_i,

            "battery_purity": self.battery_purity(),
            "purity": self.purity(),

            "photon_count": self.photon_count(),
        }

    def replay(self, u_values):

        self.reset()

        history = []

        for u in u_values:
            history.append(self.step(float(u)))

        return history


# ======================================================================
# ============================= RL UTILS ============================
# ======================================================================

    @abstractmethod
    def observe(self):

        pass

    def reward(self, info : dict, objective : str = "ergotropy_symmetric"):

        dE = info["d_energy"]
        dErg = info["d_" + objective]

        return np.array([dE, dErg], dtype=np.float32)

# ======================================================================
# ============================= UTILS ============================
# ======================================================================

    def battery_state(self):

        return self.rho.ptrace(1)

    def energy(self):

        return float(np.real(qt.expect(self.Hb, self.rho)))


    def ergotropy(self, space: str = "symmetric", rho_b=None):
        """
        ergotropy of a battery state

        space : str
            'symmetric': unitaries in the j = Nb/2 sector.
            'full': arbitrary unitaries on all Nb TLS.
        """

        if rho_b is None:
            rho_b = self.battery_state()

        eigenvalues = np.linalg.eigvalsh(rho_b.full())
        eigenvalues = np.sort(np.clip(eigenvalues.real, 0, None))[::-1]
        eigenvalues /= eigenvalues.sum()

        stored_energy = float(np.real(qt.expect(self.Hb_bare, rho_b)))

        if space == "symmetric":
            passive_energy = self.wb * np.dot(
                eigenvalues, np.arange(self.Nb + 1)
            )

        elif space == "full":
            passive_energy = self.wb * (1 - eigenvalues[0])

        else:
            raise ValueError(f"Unknown space: {space}")

        return max(0.0, stored_energy - passive_energy)


    def battery_purity(self):

        rho_b = self.battery_state()

        return float(np.real((rho_b * rho_b).tr()))

    def purity(self):

        return float(np.real((self.rho * self.rho).tr()))

    def photon_count(self):

        return float(np.real(qt.expect(self.n_photons_op, self.rho)))

