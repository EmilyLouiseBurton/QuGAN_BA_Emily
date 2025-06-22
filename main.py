from training.QuGAN_training import train_model, MODEL_DATA
from vizualisation.plots import run_plots
from training.config import EVAL_SETTINGS

def main():
   train_model()
   run_plots(MODEL_DATA, max_graphs=EVAL_SETTINGS["num_graphs"])

if __name__ == "__main__":
   main()
