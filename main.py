import torch
import numpy as np
from models.Discriminator import Discriminator
from training.QuGAN_training import train_discriminator, train_qugan
from quantum.circuit import create_qugan_circuit
from vizualisation.plots import run_plots
from data.ports import real_graphs_tensor
from training.config import *
from training.evaluation import evaluate_generator
from training.evaluation import calculate_standard_deviation_and_edges_from_qugan


def main():
    # Extract hyperparameters
    learning_rate_disc = HYPERPARAMS["learning_rate_disc"]
    batch_size = HYPERPARAMS["batch_size"]
    epochs = HYPERPARAMS["epochs"]
    param_noise_std = HYPERPARAMS["noise_std"]
    output_noise_std = HYPERPARAMS["output_noise_std"]

    # Loop through each QuGAN config
    for config in MODEL_CONFIGS:
        print(f"\n=== Training {config['model_type']} ===")

        model_type = config["model_type"]
        num_qubits = config["num_qubits"]
        layers = config["layers"]
        gate_type = config.get("gate_type", HYPERPARAMS["gate_type"])

        # Create quantum circuit
        qnode, num_params = create_qugan_circuit(
            num_qubits=num_qubits,
            layers=layers,
            gate_type=gate_type
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        param_tensor = torch.nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(num_params,), device=device))

        discriminator = Discriminator(input_size=MODEL_ARCH["discriminator_input_size"])

        gen_losses = []

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs} for {model_type}")

            # Train discriminator
            disc_loss = train_discriminator(
                discriminator=discriminator,
                real_graphs_tensor=real_graphs_tensor,
                qnode=qnode,
                param_tensor=param_tensor,
                batch_size=batch_size
            )

            # Train generator
            g_losses = train_qugan(
                discriminator=discriminator,
                qnode=qnode,
                param_tensor=param_tensor,
                epochs=1
            )
            gen_losses.extend(g_losses)
            avg_gen_loss = np.mean(gen_losses)
            steps = list(range(len(gen_losses)))
            losses = gen_losses

            # Evaluation (with noise handling)
            valid_graphs, mean_valid, std_dev, all_weights = evaluate_generator(
                param_tensor=param_tensor,
                qnode=qnode,
                num_graphs=EVAL_SETTINGS["num_graphs"],
                num_seeds=HYPERPARAMS["num_seeds"],
                param_noise_std=param_noise_std,
                output_noise_std=output_noise_std
            )

            std_qugan, edge_weights = calculate_standard_deviation_and_edges_from_qugan(
                qnode=qnode,
                param_tensor=param_tensor,
                num_samples=EVAL_SETTINGS["num_graphs"],
                param_noise_std=param_noise_std,
                output_noise_std=output_noise_std
            )

            model_number = int(model_type.split("(")[1].split(")")[0])

            if model_number not in MODEL_DATA:
                MODEL_DATA[model_number] = {
                    "valid_graphs": [],
                    "std_dev": [],
                    "edge_weights": [],
                    "gen_loss": [],
                    "steps": [],
                    "losses": []
                }

            # Log results
            mean_valid = np.mean(valid_graphs)
            MODEL_DATA[model_number]["valid_graphs"].append(mean_valid)
            MODEL_DATA[model_number]["std_dev"].append(std_qugan)
            MODEL_DATA[model_number]["edge_weights"].extend(edge_weights)
            MODEL_DATA[model_number]["gen_loss"].append(avg_gen_loss)
            MODEL_DATA[model_number]["steps"] = steps
            MODEL_DATA[model_number]["losses"] = losses

            print(f"[Model {model_number}] Epoch {epoch + 1}")
            print(f"  Valid Graphs (mean): {mean_valid}")
            print(f"  Generator Loss: {avg_gen_loss:.4f}")
            print(f"  STD: {std_qugan:.4f}")
            print(f"  Num Edge Weights This Epoch: {len(edge_weights)}")
            print(f"  Total Valid Graphs So Far: {MODEL_DATA[model_number]['valid_graphs']}")
            print("-" * 40)

    print("\n=== Training complete. Plotting results... ===")
    run_plots()


if __name__ == "__main__":
    main()
