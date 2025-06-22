MODEL_CONFIGS = [
    {"model_type": "QuGAN(36)", "num_qubits": 6, "layers": 5, "gate_type": "RX_Y"},
    {"model_type": "QuGAN(66)", "num_qubits": 6, "layers": 10, "gate_type": "RX_Y"},
    {"model_type": "QuGAN(72)", "num_qubits": 6, "layers": 5, "gate_type": "RXRY"},
    {"model_type": "QuGAN(132)", "num_qubits": 6, "layers": 10, "gate_type": "RXRY"},
    {"model_type": "RQC(90)", "num_qubits": 6, "layers": 6, "gate_type": "RANDOM"},
    {"model_type": "RQC(180)", "num_qubits": 6, "layers": 10, "gate_type": "RANDOM"}
]

HYPERPARAMS = {
    "learning_rate_gen": 1e-3,   
    "learning_rate_disc": 0.3,   
    "num_seeds": 1,
    "batch_size": 32,
    "epochs": 2,
    "gate_type": "PauliY",  
    "share_latent_vector": False,
    "use_minibatch_discriminator": False,
}

MODEL_ARCH = {
    "discriminator_input_size": 6,
    "generator_input_size": 10,
    "generator_hidden_size": 10,
    "generator_output_size": 6

}

EVAL_SETTINGS = {
    "num_graphs": 1000
}

MODEL_DATA = {
    36: {
        "valid_graphs": [], "std_dev": [], "std_dev_per_step": [],
        "edge_weights": [], "gen_loss": [], "steps": [], "losses": [],
        "valid_graphs_all_seeds": [], "std_dev_all_seeds": [], "gen_loss_all_seeds": []
    },
    66: {
        "valid_graphs": [], "std_dev": [], "std_dev_per_step": [],
        "edge_weights": [], "gen_loss": [], "steps": [], "losses": [],
        "valid_graphs_all_seeds": [], "std_dev_all_seeds": [], "gen_loss_all_seeds": []
    },
    72: {
        "valid_graphs": [], "std_dev": [], "std_dev_per_step": [],
        "edge_weights": [], "gen_loss": [], "steps": [], "losses": [],
        "valid_graphs_all_seeds": [], "std_dev_all_seeds": [], "gen_loss_all_seeds": []
    },
    132: {
        "valid_graphs": [], "std_dev": [], "std_dev_per_step": [],
        "edge_weights": [], "gen_loss": [], "steps": [], "losses": [],
        "valid_graphs_all_seeds": [], "std_dev_all_seeds": [], "gen_loss_all_seeds": []
    },
    90: {
        "valid_graphs": [], "std_dev": [], "std_dev_per_step": [],
        "edge_weights": [], "gen_loss": [], "steps": [], "losses": [],
        "valid_graphs_all_seeds": [], "std_dev_all_seeds": [], "gen_loss_all_seeds": []
    },
    180: {
        "valid_graphs": [], "std_dev": [], "std_dev_per_step": [],
        "edge_weights": [], "gen_loss": [], "steps": [], "losses": [],
        "valid_graphs_all_seeds": [], "std_dev_all_seeds": [], "gen_loss_all_seeds": []
    },
    "train": {
        "edge_weights": []
    }
}