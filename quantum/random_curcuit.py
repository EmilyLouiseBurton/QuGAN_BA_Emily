import random
import pennylane as qml
import torch

def create_random_qugan_generator(num_qubits=6, layers=5):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def circuit(params, latent_input):
        qml.AngleEmbedding(latent_input, wires=range(num_qubits), rotation='X')
        idx = 0
        for _ in range(layers):
            # Random order of RX, RY, RZ gates per qubit
            for wire in range(num_qubits):
                gate_sequence = random.sample(["RX", "RY", "RZ"], k=3)
                for gate in gate_sequence:
                    if gate == "RX":
                        qml.RX(params[idx], wires=wire)
                    elif gate == "RY":
                        qml.RY(params[idx], wires=wire)
                    elif gate == "RZ":
                        qml.RZ(params[idx], wires=wire)
                    idx += 1
            # Random entangling pattern
            qubit_list = list(range(num_qubits))
            random.shuffle(qubit_list)
            for k in range(0, num_qubits - 1, 2):
                qml.CNOT(wires=[qubit_list[k], qubit_list[k + 1]])

        return qml.probs(wires=range(num_qubits))

    # Each wire gets 3 params per layer (RX, RY, RZ)
    param_count = layers * num_qubits * 3

    return circuit, param_count