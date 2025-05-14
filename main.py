from training.QuGAN_training import train_model
from vizualisation.plots import run_plots
from training.config import MODEL_DATA

def main():
    train_model()
    run_plots(MODEL_DATA)

if __name__ == "__main__":
    main()