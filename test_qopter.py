import qiskit
import qopter
from math import exp
from time import time


BACKEND_NAME = 'backend'
QASM_FILE_NAME = '/path/to/.qsam'


qiskit.IBMQ.load_accounts()
backend = qiskit.IBMQ.get_backend(BACKEND_NAME)
properties = backend.properties()

optimizer = qopter.Optimizer(properties)
circuit = qiskit.load_qasm_file(QASM_FILE_NAME)
start_time = time()
(qasm, esp, mapping) = optimizer.compile(circuit, 1000, 10000, 1, True)
end_time = time()

print(qasm)
print("esp:", exp(esp))
print("time: ", end_time - start_time)

