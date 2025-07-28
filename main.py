import torch
from training.QuGAN_training import train_model, MODEL_DATA
from vizualisation.plots import run_plots
from training.config import EVAL_SETTINGS

def main():
    torch.set_num_threads(1) 
    print(f"Using {torch.get_num_threads()} threads.")

    train_model()
    run_plots(MODEL_DATA, max_graphs=EVAL_SETTINGS["num_graphs"])

if __name__ == "__main__":
    main()