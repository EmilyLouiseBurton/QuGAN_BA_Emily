import pennylane as qml
import torch
from quantum.random_curcuit import create_random_qugan_generator  

def create_qugan_generator(num_qubits=6, layers=5, gate_type="RX_Y"):
    if gate_type == "RANDOM":
        return create_random_qugan_generator(num_qubits, layers)

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(params, latent_input):
        # Step 1: Angle embedding of latent input
        qml.AngleEmbedding(latent_input, wires=range(num_qubits), rotation='X')

        # Step 2: Quantum circuit based on ansatz
        idx = 0
        for _ in range(layers):
            if gate_type == "RX_Y":
                for i in range(num_qubits):
                    qml.RX(params[idx], wires=i)
                    idx += 1
                for i in range(num_qubits):
                    qml.PauliY(wires=i)
            elif gate_type == "RXRY":
                for i in range(num_qubits):
                    qml.RX(params[idx], wires=i)
                    idx += 1
                for i in range(num_qubits):
                    qml.RY(params[idx], wires=i)
                    idx += 1
            else:
                raise ValueError(f"Unsupported gate type: {gate_type}")

            for i in range(num_qubits):
                qml.CNOT(wires=[i, (i + 1) % num_qubits])

        return qml.probs(wires=range(num_qubits)) 

    # Calculate parameter count based on ansatz
    if gate_type == "RX_Y":
        param_count = layers * num_qubits
    elif gate_type == "RXRY":
        param_count = layers * num_qubits * 2

    return circuit, param_count