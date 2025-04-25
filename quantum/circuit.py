import pennylane as qml
import torch

def create_qugan_circuit(num_qubits=6, layers=1, gate_type="RXRY"):
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev, interface="torch")
    def circuit(params):
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

        # returns a torch.Tensor 
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]
    
    if gate_type == "RX_Y":
        param_count = layers * num_qubits
    elif gate_type == "RXRY":
        param_count = layers * num_qubits * 2
    else:
        raise ValueError(f"Unsupported gate type: {gate_type}")

    return circuit, param_count