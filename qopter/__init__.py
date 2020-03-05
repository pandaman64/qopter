from typing import Any, List, Tuple, Dict
from math import log, inf
from qiskit.dagcircuit import DAGCircuit
#from qiskit.unroll import DagUnroller, DAGBackend
from qiskit.transpiler import transpile
import re
import sys

from . import libqopter

_gatedef = re.compile('gate[^}]+?}')

class Optimizer:
    def __init__(self, calibration: Any) -> None:
        self.single_fidelity: List[float] = []
        self.readout_fidelity: List[float] = []
        self.cnot_fidelity = [(e['qubits'][0], e['qubits'][1], log(1 - e['gateError']['value'])) for e in calibration['multi_qubit_gates']]

        for (i, e) in enumerate(calibration['qubits']):
            assert(e['name'] == 'Q{}'.format(i))
            print(e)
            if e['gateError'] is None:
                self.single_fidelity.append(-inf)
            else:
                self.single_fidelity.append(log(1 - e['gateError']['value']))
            if e['readoutError'] is None:
                self.readout_fidelity.append(-inf)
            else:
                self.readout_fidelity.append(log(1 - e['readoutError']['value']))

    def compile(self, qasm: Any, num_initial_mappings: int, num_beams: int, paths: int, edge_to_edge: bool, random: bool) -> Tuple[str, float, Dict[Tuple[str, int], int]]:
        # unroll the given circuit into a collection of unitary + cx
        dag = DAGCircuit.fromQuantumCircuit(qasm)
        expanded_qasm = transpile(dag, format='qasm')
        # print('before:\n' + expanded_qasm)
        expanded_qasm = _gatedef.sub('', expanded_qasm) # remove gate definitions
        # print('after\n' + expanded_qasm)

        # current technology doesn't permit gates after measurements, so reorder them
        gates: List[str] = []
        measurements: List[str] = []

        for g in expanded_qasm.split('\n'):
            if g.startswith('measure'):
                measurements.append(g)
            else:
                gates.append(g)

        print('expanded length: {}'.format(len(gates) + len(measurements)))
        expanded_qasm = '\n'.join(gates + measurements)

        print(expanded_qasm, file=sys.stderr)

        # optimize
        (optimized_qasm, estimated_probability, variables_to_qopter) = libqopter.solve(expanded_qasm, self.single_fidelity, self.cnot_fidelity, self.readout_fidelity, num_initial_mappings, num_beams, paths, edge_to_edge, random)

        return (optimized_qasm, estimated_probability, variables_to_qopter)

