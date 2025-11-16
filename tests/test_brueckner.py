import pytest
from pyscf import gto, scf
from shadow_ci.utils import make_hydrogen_chain
from shadow_ci.brueckner import BruecknerSolver
from shadow_ci.solvers import FCISolver
from shadow_ci.utils import get_hf_reference, get_single_excitations
import numpy as np
from scipy.linalg import expm

@pytest.fixture(scope="module", params=[2, 4, 6, 8, 10, 12])
def n_hydrogen_mf(request):
    """N hydrogen chain for benchmark scaling."""
    mol = gto.Mole()
    atom = make_hydrogen_chain(request.param)
    mol.build(atom, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    return mf.run()


class TestOrbitalRotation:

    def test_taylor_expansion(self, n_hydrogen_mf: scf.hf.RHF):
        """
        Assert that the rotatation via first-order taylor expansion is equal 
        up to order ||t1|| to direct exponentiation.
        """

        brueckner = BruecknerSolver(n_hydrogen_mf)
        solver = FCISolver(n_hydrogen_mf)
        state, _ = solver.solve()

        ref_idx = get_hf_reference(n_hydrogen_mf).to_int()

        excitations = get_single_excitations(n_hydrogen_mf)
        singles_idx = [s.bitstring.to_int() for s in excitations]
        amplitudes = [state.data[i] for i in singles_idx]

        nocc, _ = n_hydrogen_mf.mol.nelec
        norb = n_hydrogen_mf.mo_coeff.shape[0]
        nvirt = norb - nocc
        c1 = np.empty((nocc, nvirt), dtype=np.float64)
        for c, e in zip(amplitudes, excitations):
            i = e.occ
            a = e.virt
            c1[i,a] = c.real

        ovlp = n_hydrogen_mf.get_ovlp()
        if np.allclose(ovlp, np.eye(ovlp.shape[-1])):
            ovlp = None

        mo_coeff = n_hydrogen_mf.mo_coeff
        bmo_occ, bmo_vir = brueckner._update_mo_coeff(mo_coeff, c1, ovlp)
        approx = np.hstack((bmo_occ, bmo_vir))

        kappa = np.zeros((norb, norb))
        kappa[nocc:, :nocc] = c1.T
        kappa[:nocc, nocc:] = -c1

        assert np.allclose(kappa + kappa.T, np.zeros((norb, norb))), "Kappa is not skew symettric!"

        exact = expm(kappa) @ mo_coeff

        error = np.linalg.norm(exact - approx)
        kappa_norm_squared = np.linalg.norm(kappa)**2
        normalized_error = error / kappa_norm_squared

        assert error < 0.1 * np.linalg.norm(kappa), (
            f"Error should be O(||κ||²), got ||error||={error:.2e}, ||κ||={np.linalg.norm(kappa):.2e}"
        )



    def test_exponentiation


