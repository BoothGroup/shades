import pytest
from pyscf import gto, scf
from shadow_ci.utils import make_hydrogen_chain
from shadow_ci.brueckner import brueckner_cycle, rotate_mf, rotate_mo_coeffs
from shadow_ci.solvers import FCISolver
from shadow_ci.utils import get_hf_reference, get_single_excitations
import numpy as np
from scipy.linalg import expm
from copy import copy

@pytest.fixture(scope="module", params=[2, 4, 6, 8])
def n_hydrogen_mf(request):
    """N hydrogen chain for benchmark scaling."""
    mol = gto.Mole()
    atom = make_hydrogen_chain(request.param, bond_length=2.0)
    mol.build(atom=atom, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    return mf.run()

class TestOrbitalRotation:

    def test_fci_energy_invarience(self, n_hydrogen_mf: scf.hf.RHF):

        fci = FCISolver(n_hydrogen_mf)
        E, _, c1, _ = fci.get_configuration_interaction()
        update_mf(n_hydrogen_mf, c1)
        rotated_fci = FCISolver(n_hydrogen_mf)
        E_new, _, _, _ = rotated_fci.get_configuration_interaction()

        assert np.allclose(E, E_new), "The FCI energy wasnt invarient to the MO rotation!"


    def test_smooth_rotation(self, n_hydrogen_mf: scf.hf.RHF):

        fci = FCISolver(n_hydrogen_mf)
        _, c0, c1, _ = fci.get_configuration_interaction()

        norms = [np.linalg.norm(c1)]

        for i in np.arange(0.001, 1.0, 0.001):
            mf = copy(n_hydrogen_mf)
            update_mf(mf, -c1*i/c0)
            fci = FCISolver(mf)
            _, _, _c1, _ = fci.get_configuration_interaction()
            norms.append(np.linalg.norm(_c1))

        print(norms)

    def test_single_rotation(self, n_hydrogen_mf: scf.hf.RHF):

        fci = FCISolver(n_hydrogen_mf)
        _, c0, c1, _ = fci.get_configuration_interaction()
        kappa = get_kappa(c1)

        update_mf(n_hydrogen_mf, c1)
        rotated_fci = FCISolver(n_hydrogen_mf)
        _, rotated_c0, rotated_c1, _ = rotated_fci.get_configuration_interaction()

        assert np.allclose(rotated_c1, transform_c1(c1, kappa))



    def test_taylor_expansion(self, n_hydrogen_mf: scf.hf.RHF):
        """
        Assert that the rotatation via first-order taylor expansion is equal 
        up to order ||t1|| to direct exponentiation.
        """

        def f(mf: scf.hf.RHF):
            solver = FCISolver(mf)
            return solver.get_configuration_interaction()

        brueckner = BruecknerSolver(n_hydrogen_mf)
        brueckner.solve(f)
        
        


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



    def test_exponentiation(self, n_hydrogen_mf: scf.hf.RHF):
        pass


