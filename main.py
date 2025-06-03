from training.QuGAN_training import train_model
from vizualisation.plots import run_plots
from quantum.graph_generation import generate_graph_from_qugan




def main():
   train_model()
   run_plots()

if __name__ == "__main__":
   main()
