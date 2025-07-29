MODEL_CONFIGS = [
    {"model_type": "QuGAN(36)", "num_qubits": 6, "layers": 5, "gate_type": "RX_Y"},
    {"model_type": "QuGAN(66)", "num_qubits": 6, "layers": 10, "gate_type": "RX_Y"},
    {"model_type": "QuGAN(72)", "num_qubits": 6, "layers": 5, "gate_type": "RXRY"},
    {"model_type": "QuGAN(132)", "num_qubits": 6, "layers": 10, "gate_type": "RXRY"},
    {"model_type": "RQC(90)", "num_qubits": 6, "layers": 5, "gate_type": "RANDOM"},
    {"model_type": "RQC(180)", "num_qubits": 6, "layers": 10, "gate_type": "RANDOM"}
]

HYPERPARAMS = {
    "learning_rate_gen": 0.001,
    "learning_rate_disc": 0.3,
    "num_seeds": 1,
    "batch_size": 32,
    "epochs": 1000,
    "share_latent_vector": False,
    "use_minibatch_discriminator": False,
    "minibatch_weight": 0.1
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
        "valid_graphs_all_seeds": [],
        "std_dev_all_seeds": [],
        "intra_graph_std_all_seeds": [],
        "graph_means_all_seeds": [],
        "gen_loss_all_seeds": [],
        "edge_weights_all_seeds": [],
        "steps_all_seeds": [],
        "expressibility_kl": None
    },
    66: {
        "valid_graphs_all_seeds": [],
        "std_dev_all_seeds": [],
        "intra_graph_std_all_seeds": [],
        "graph_means_all_seeds": [],
        "gen_loss_all_seeds": [],
        "edge_weights_all_seeds": [],
        "steps_all_seeds": [],
        "expressibility_kl": None
    },
    72: {
        "valid_graphs_all_seeds": [],
        "std_dev_all_seeds": [],
        "intra_graph_std_all_seeds": [],
        "graph_means_all_seeds": [],
        "gen_loss_all_seeds": [],
        "edge_weights_all_seeds": [],
        "steps_all_seeds": [],
        "expressibility_kl": None
    },
    132: {
        "valid_graphs_all_seeds": [],
        "std_dev_all_seeds": [],
        "intra_graph_std_all_seeds": [],
        "graph_means_all_seeds": [],
        "gen_loss_all_seeds": [],
        "edge_weights_all_seeds": [],
        "steps_all_seeds": [],
        "expressibility_kl": None
    },
    90: {
        "valid_graphs_all_seeds": [],
        "std_dev_all_seeds": [],
        "intra_graph_std_all_seeds": [],
        "graph_means_all_seeds": [],
        "gen_loss_all_seeds": [],
        "edge_weights_all_seeds": [],
        "steps_all_seeds": [],
        "expressibility_kl": None
    },
    180: {
        "valid_graphs_all_seeds": [],
        "std_dev_all_seeds": [],
        "intra_graph_std_all_seeds": [],
        "graph_means_all_seeds": [],
        "gen_loss_all_seeds": [],
        "edge_weights_all_seeds": [],
        "steps_all_seeds": [],
        "expressibility_kl": None
    },
    "train": {
        "edge_weights": []
    }
}