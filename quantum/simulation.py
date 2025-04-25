import numpy as np
from qiskit_aer import Aer
from qiskit.compiler import transpile

def quantum_simulation(qc, params_values):
    backend = Aer.get_backend('qasm_simulator')
    qc_copy = qc.copy()
    qc_copy.measure_all()

    param_dict = dict(zip(qc_copy.parameters, params_values))
    bound_qc = qc_copy.assign_parameters(param_dict)

    transpiled_qc = transpile(bound_qc, backend)
    job = backend.run(transpiled_qc, shots=1024)
    result = job.result()

    counts = result.get_counts()
    num_states = 2 ** qc_copy.num_qubits

    probs = np.array([
        counts.get(format(i, f'0{qc_copy.num_qubits}b'), 0) 
        for i in range(num_states)
    ])
    
    total = np.sum(probs)
    probs = probs / total if total > 0 else np.ones(num_states) / num_states

    return probs