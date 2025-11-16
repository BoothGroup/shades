

import pytest
from pyscf import gto, scf
from shadow_ci.utils import make_hydrogen_chain

@pytest.fixture(scope="module", params=[2, 6, 10, 14, 18, 22, 26, 30])
def n_hydrogen_mf(request):
    """N hydrogen chain for benchmark scaling."""
    mol = gto.Mole()
    atom = make_hydrogen_chain(request.param)
    mol.build(atom, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    return mf

class TestOrbitalRotation:

    def test_expm(self, n_hydrogen_mf):
        pass

    def test_taylor_expansion(self, n_hydrogen_mf):
        pass
