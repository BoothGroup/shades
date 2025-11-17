import logging
from copy import copy
from typing import Callable, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray
from pyscf import lib, scf
from scipy.linalg import eigh, expm

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)  # Default level

def make_kappa(t1: NDArray[np.float64]) -> NDArray[np.float64]:

    nocc, nvirt = t1.shape
    norb = nocc + nvirt

    kappa = np.zeros((norb, norb))
    kappa[-nvirt:, :nocc] = t1.conj().T
    kappa[:nocc, -nvirt:] = -t1

    return kappa

def rotate_mo_coeffs(
        C: NDArray[np.float64],
        t1: NDArray[np.float64],
        ovlp: Optional[NDArray[np.float64]],
        damping: float = 0.0,
        diis: Optional[lib.diis.DIIS] = None,
        method: Literal["expn", "taylor"] = "taylor"
    ):

    nocc, nvirt = t1.shape
    norb = C.shape[-1]

    if not nocc + nvirt == norb:
        raise ValueError("The shape of molecular orbital coefficients and t1 do not match!")

    if method == "expn":
        kappa = make_kappa(t1)
        bmo = C @ expm(kappa*(1-damping))

        bmo_occ = bmo[:, :nocc]
        bmo_vir = bmo[:, nocc:]

    elif method == "taylor":

        delta_occ = (1 - damping) * np.dot(C[:, nocc:], t1.T) # multiply virtuals by t1.T
        bmo_occ = C[:, :nocc] + delta_occ

        if ovlp is None:
            bmo_occ = np.linalg.qr(bmo_occ)[0]
        else:
            dm_occ = np.dot(bmo_occ, bmo_occ.T)
            _, v = eigh(dm_occ, b=ovlp, type=2)
            bmo_occ = v[:, -nocc:]

        if diis:
            dm_occ = np.dot(bmo_occ, bmo_occ.T)
            dm_occ = diis.update(dm_occ)
            _, v = eigh(dm_occ, b=ovlp, type=2)
            bmo_occ = v[:, -nocc:]

        if ovlp is None: # get virtuals by unitary completion
            dm_vir = np.eye(norb) - np.dot(bmo_occ, bmo_occ.T)
        else:
            dm_vir = np.linalg.inv(ovlp) - np.dot(bmo_occ, bmo_occ.T)

        _, v = eigh(dm_vir, b=ovlp, type=2)
        bmo_vir = v[:, -nvirt:]

        if not bmo_occ.shape[-1] == nocc:
            raise RuntimeError()
        if not bmo_vir.shape[-1] == nvirt:
            raise RuntimeError()

        bmo = np.hstack((bmo_occ, bmo_vir))

    else:
        raise ValueError(f"Unrecognised rotation method: {method}")

    if ovlp is None and not np.allclose(np.dot(bmo.T, bmo), np.eye(norb)):
        raise RuntimeError("Brueckner molecular orbitals no longer orthonormal!")
    else:
        assert np.allclose(np.linalg.multi_dot((bmo.T, ovlp, bmo)), np.eye(norb))

    return bmo_occ, bmo_vir

def rotate_mf(
        mf: Union[scf.hf.RHF, scf.uhf.UHF],
        t1: NDArray[np.float64],
        canonicalize: bool = True,
        damping: float = 0.0,
        diis: Optional[lib.diis.DIIS] = None,
        method: Literal["expn", "taylor"] = "taylor"
    ):

    mf = copy(mf)

    mo_coeff = mf.mo_coeff
    norb = mo_coeff.shape[-1]
    nocc, _ = mf.mol.nelec
    nvirt = norb - nocc

    if not t1.shape == (nocc, nvirt):
        raise ValueError("Incorrect shape for T1 amplitudes.")

    ovlp = mf.get_ovlp()
    if np.allclose(ovlp, np.eye(ovlp.shape[-1])):
        ovlp = None

    bmo_occ, bmo_vir = rotate_mo_coeffs(mo_coeff, t1, ovlp, damping, diis, method)

    if canonicalize:
        if canonicalize == "hcore":
            h1e = mf.get_hcore()
        else:
            h1e = mf.get_fock()
        _, r = np.linalg.eigh(np.linalg.multi_dot((bmo_occ.T, h1e, bmo_occ)))
        bmo_occ = np.dot(bmo_occ, r)
        _, r = np.linalg.eigh(np.linalg.multi_dot((bmo_vir.T, h1e, bmo_vir)))
        bmo_vir = np.dot(bmo_vir, r)

    bmo = np.hstack((bmo_occ, bmo_vir))
    if ovlp is None and not np.allclose(np.dot(bmo.T, bmo), np.eye(norb)):
        raise RuntimeError("Brueckner molecular orbitals no longer orthonormal!")
    else:
        assert np.allclose(np.linalg.multi_dot((bmo.T, ovlp, bmo)), np.eye(norb))

    mf.mo_coeff = bmo
    mf.e_tot = mf.energy_tot()
    return mf

def brueckner_cycle(
        mf: Union[scf.hf.RHF, scf.uhf.UHF],
        estimator_fn: Callable[
            [Union[scf.hf.RHF, scf.uhf.UHF]],
            tuple[np.float64, np.float64, NDArray, NDArray]
        ],
        canonicalize: bool = True,
        damping: float = 0.0,
        *,
        max_iter: int = 10,
        callback_fn: Optional[Callable[[np.float64, np.float64, np.float64], bool]] = None,
        verbose: int = 0
    ):

    if verbose == 0:
        logger.setLevel(logging.WARNING)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    else:  # verbose >= 2
        logger.setLevel(logging.DEBUG)

    converged = False

    diis = lib.diis.DIIS()

    logger.info("Starting Brueckner orbital optimization")
    logger.info(f"  max_iter={max_iter}, damping={damping}")
    logger.info("")
    logger.info(f"{'Iter':<6} {'Energy':<18} {'c0':<16} {'||c1||':<14}")
    logger.info("-" * 70)

    for iteration in range(max_iter):

        E, c0, c1, _ = estimator_fn(mf)
        norm = np.linalg.norm(c1)
        t1 = -c1

        logger.info(
            f"{iteration + 1:<6} {E:<18.10f} {np.abs(c0):<16.10f} {norm:<14.6e}"
        )
        logger.debug(f"  c1 max: {np.max(np.abs(c1)):.6e}")
        logger.debug(f"  t1 max: {np.max(np.abs(t1)):.6e}")

        if callback_fn:
            converged = callback_fn(E, c0, norm)

        if converged:
            logger.info("-" * 70)
            logger.info(f"Converged after {iteration + 1} iterations")
            break

        mf = rotate_mf(mf, t1, canonicalize, damping, diis=diis, method="taylor")

    if not converged:
        logger.info("-" * 70)
        logger.warning(f"Did not converge after {max_iter} iterations")

if __name__ == "__main__":

    from shadow_ci.solvers import FCISolver
    from shadow_ci.utils import make_hydrogen_chain
    from pyscf import gto, scf

    atom = make_hydrogen_chain(10, bond_length=1.5)
    mol = gto.Mole()
    mol.build(atom=atom, basis="sto-3g", verbose=0)
    mf = scf.RHF(mol)
    mf.run()

    fci = FCISolver(mf)

    brueckner_cycle(mf, fci.estimate, max_iter=1000, damping=0.3, verbose=3)