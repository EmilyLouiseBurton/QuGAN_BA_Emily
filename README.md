QuGAN Graph Generation

This project trains quantum generative models (QuGANs) to generate small graphs that satisfy the triangle inequality.

Getting Started

1. Create and activate your Python environment:
   python -m venv myenv
   source myenv/bin/activate

2. Install dependencies:
   pip install -r requirements.txt

3. Run training and evaluation:
   python main.py

Output

After training and evaluation, results are saved in the checkpoints/ folder:

- basis_model_<id>_summary.json: JSON summary of each model’s performance across epochs
- real_edge_weights.npy: Preprocessed real graph data
- real_data_edge_weight_distribution.png: KDE plot of edge weights from real data
- Visualization plots are automatically shown (valid graphs, std, loss, KDE)
