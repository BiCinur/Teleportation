from pyquil import Program, get_qc
from pyquil.api import WavefunctionSimulator
from pyquil.gates import *
import numpy as np

def random_unitary():
    """
    :return: array of shape (2, 2) representing random unitary matrix drawn from Haar measure
    """
    # draw complex matrix from Ginibre ensemble
    z = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    # QR decompose this complex matrix
    q, r = np.linalg.qr(z)
    # make this decomposition unique
    d = np.diagonal(r)
    l = np.diag(d) / np.abs(d)
    return np.matmul(q, l)


def random_wavefunc():
    """
    :return: Program for a quantum circuit creating a random 1-qubit state
    """
    p = Program()
    p.defgate("RandomUnitary", random_unitary())
    p.inst(("RandomUnitary", 2))
    return p
wfn_sim = WavefunctionSimulator()
prog += random_wavefunc()
wfn_orig = wfn_sim.wavefunction(prog)
print(wfn_orig)

def bell_state(q0, q1):
    """
    :param int q0: first (R-to-L) qubit to form part of the Bell state
    :param int q1: second (R-to-L) qubit to form part of the Bell state
    :return: Program creating a Bell state between input qubits
    """
    p = Program(H(q1), CNOT(q1, q0))

def bell_basis_circuit(q0, q1):
    """
    :param int q0: first (R-to-L) qubit that Alice will measure
    :param int q1: second (R-to-L) qubit that Alice will measure
    :return: Program preparing a measurement in the Bell basis
    """
    p = Program(CNOT(q1, q0), H(q1))
    return p

prog += bell_basis_circuit(1, 2)

def conditionally_apply_gate(p, q0, q1, q2, alice_regs):
    """
    NOTE: This function directly modifies the input Program,
        but does not return a new Program
    
    :param p: Program that performs the teleportation protocol
        upto conditional application of Bob's gate(s)
    :param int q0: only qubit that Bob possesses
    :param int q1: first (R-to-L) qubit that Alice measures
    :param int q2: second (R-to-L) qubit that Alice measures
    :param list alice_regs: classical registers holding Alice's
        measurements of her qubits
    """
    p.measure(q1, alice_regs[0]).if_then(alice_regs[0], Program(X(q0)), Program(I(q0)))
    p.measure(q2, alice_regs[1]).if_then(alice_regs[1], Program(Z(q0)), Program(I(q0)))

conditionally_apply_gate(prog, 0, 1, 2, alice_regs)
wfn = wfn_sim.wavefunction(prog)
print (wfn)


np.testing.assert_almost_equal(
    np.sum([v for k, v in wfn.get_outcome_probs().items()
            if k[-1] == '0']), 
    wfn_orig.get_outcome_probs()['0'])

np.testing.assert_almost_equal(
    np.sum([v for k, v in wfn.get_outcome_probs().items()
            if k[-1] == '1']), 
    wfn_orig.get_outcome_probs()['1'])









    
    return p
