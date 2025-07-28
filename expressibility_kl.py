import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import unitary_group, entropy
import random

# Set random seeds
random.seed(None)
np.random.seed(None)

# Structured Generator
def create_qugan_generator(num_qubits=6, layers=5, gate_type="RX_Y"):
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(params, latent_input):
        qml.AngleEmbedding(latent_input, wires=range(num_qubits), rotation='X')
        qml.AngleEmbedding(latent_input, wires=range(num_qubits), rotation='Y')
        qml.AngleEmbedding(latent_input, wires=range(num_qubits), rotation='Z')
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
            for i in range(num_qubits):
                qml.CNOT(wires=[i, (i + 1) % num_qubits])
        return qml.state()
    param_count = layers * num_qubits if gate_type == "RX_Y" else layers * num_qubits * 2
    return circuit, param_count

# Random Generator
def create_random_qugan_generator(num_qubits=6, layers=5):
    dev = qml.device("default.qubit", wires=num_qubits)
    @qml.qnode(dev)
    def circuit(params, latent_input):
        qml.AngleEmbedding(latent_input, wires=range(num_qubits), rotation='X')
        idx = 0
        for _ in range(layers):
            for wire in range(num_qubits):
                for gate in random.sample(["RX", "RY", "RZ"], k=3):
                    getattr(qml, gate)(params[idx], wires=wire)
                    idx += 1
            qubits = list(range(num_qubits))
            random.shuffle(qubits)
            for i in range(0, num_qubits - 1, 2):
                qml.CNOT(wires=[qubits[i], qubits[i + 1]])
        return qml.state()
    param_count = layers * num_qubits * 3
    return circuit, param_count

# Fidelity computation
def generate_fidelity_distribution(qnode, n_params, num_samples=5000, num_qubits=6):
    fidelities = []
    for _ in range(num_samples):
        params = np.random.uniform(0, 2 * np.pi, n_params)
        latent = np.random.rand(num_qubits)
        state = qnode(params, latent)
        state /= np.linalg.norm(state)
        haar = unitary_group.rvs(2 ** num_qubits)[..., 0]
        haar /= np.linalg.norm(haar)
        fidelities.append(np.abs(np.vdot(state, haar)) ** 2)
    return np.array(fidelities)

# Haar–Haar baseline
def generate_haar_fidelity_distribution(num_samples=5000, num_qubits=6):
    fidelities = []
    for _ in range(num_samples):
        psi = unitary_group.rvs(2 ** num_qubits)[..., 0]
        phi = unitary_group.rvs(2 ** num_qubits)[..., 0]
        fidelities.append(np.abs(np.vdot(psi, phi)) ** 2)
    return np.array(fidelities)

# --- Main ---
MODEL_CONFIGS = [
    {"model_type": "QuGAN(36)", "num_qubits": 6, "layers": 5, "gate_type": "RX_Y"},
    {"model_type": "QuGAN(66)", "num_qubits": 6, "layers": 10, "gate_type": "RX_Y"},
    {"model_type": "QuGAN(72)", "num_qubits": 6, "layers": 5, "gate_type": "RXRY"},
    {"model_type": "QuGAN(132)", "num_qubits": 6, "layers": 10, "gate_type": "RXRY"},
    {"model_type": "RQC(90)", "num_qubits": 6, "layers": 5, "gate_type": "RANDOM"},
    {"model_type": "RQC(180)", "num_qubits": 6, "layers": 10, "gate_type": "RANDOM"}
]

results = {}
for config in MODEL_CONFIGS:
    if config["gate_type"] == "RANDOM":
        qnode, n_params = create_random_qugan_generator(config["num_qubits"], config["layers"])
    else:
        qnode, n_params = create_qugan_generator(config["num_qubits"], config["layers"], config["gate_type"])
    print(f"Generating fidelities for {config['model_type']}...")
    fidelities = generate_fidelity_distribution(qnode, n_params)
    results[config["model_type"]] = fidelities

# Generate Haar baseline
haar_fids = generate_haar_fidelity_distribution()

# --- KL Divergence from Haar baseline ---
print("\nKL Divergences from Haar baseline:")
bins = np.linspace(0, 1, 51)
hist_haar, _ = np.histogram(haar_fids, bins=bins, density=True)
hist_haar += 1e-10

for label, fids in results.items():
    hist_model, _ = np.histogram(fids, bins=bins, density=True)
    hist_model += 1e-10
    kl = entropy(hist_model, hist_haar)
    print(f"KL({label} || Haar) = {kl:.5f}")