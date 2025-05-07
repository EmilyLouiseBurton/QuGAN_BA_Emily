# Config
MODEL_CONFIGS = [
    {"model_type": "QuGAN(36)", "num_qubits": 6, "layers": 5, "gate_type": "RX_Y"},
    {"model_type": "QuGAN(66)", "num_qubits": 6, "layers": 10, "gate_type": "RX_Y"},
    {"model_type": "QuGAN(72)", "num_qubits": 6, "layers": 5, "gate_type": "RXRY"},  
    {"model_type": "QuGAN(132)", "num_qubits": 6, "layers": 10, "gate_type": "RXRY"} 
]
HYPERPARAMS = {
    "learning_rate_disc": 0.3,
    "learning_rate_gen": 0.001,
    "num_seeds": 5,
    "batch_size": 32,
    "epochs": 10,
    "gate_type": "PauliY",
    "noise_std": 0.4,              # For parameter noise
    "output_noise_std": 0.01       # For output/noise after circuit
}


MODEL_ARCH = {
    "discriminator_input_size": 6,
    "generator_input_size": 10,
    "generator_hidden_size": 10,
    "generator_output_size": 6,
}

EVAL_SETTINGS = {
    "num_graphs": 1000
}

# Init metrics dictionary
MODEL_DATA = {
    36: {"valid_graphs": [], "std_dev": [], "std_dev_per_step": [], "edge_weights": [], "gen_loss": [], "steps": [], "losses": []},
    66: {"valid_graphs": [], "std_dev": [], "std_dev_per_step": [], "edge_weights": [], "gen_loss": [], "steps": [], "losses": []},
    72: {"valid_graphs": [], "std_dev": [], "std_dev_per_step": [], "edge_weights": [], "gen_loss": [], "steps": [], "losses": []},
    132: {"valid_graphs": [], "std_dev": [], "std_dev_per_step": [], "edge_weights": [], "gen_loss": [], "steps": [], "losses": []},
}
