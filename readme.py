# MC-PINN
MC-PINN (Monte Carlo–Physics-Informed Neural Network) combines Monte Carlo sampling with physical constraints to predict fatigue crack growth life. It augments limited a–N data using statistical porosity distributions and embeds Paris’ law and monotonicity constraints into the loss, improving accuracy, interpretability, and generalization.
1. System Requirements
  (1) Software Dependencies:
      Python ≥ 3.9
      PyTorch ≥ 1.12
      NumPy ≥ 1.23
      SciPy ≥ 1.10
      Matplotlib ≥ 3.7
      Pandas ≥ 1.5
      Scikit-learn ≥ 1.2
  (2) Operating Systems:
      Windows 11
  (3) Versions Tested:
      Windows 11 + Python 3.10 + PyTorch 2.5.1 + CUDA124
2. Installation Guide
   git clone https://github.com/fengshengzhou/MC-PINN.git
   cd MC-PINN
3. Demo
  (1) Run Demo Example:
      A demo dataset (a-N.csv) is provided.
      Run: python MC_model.py
      Run: python train.py
  (2) Expected Output:
      Random Sampling C and m Scatter Plot（PNG）
      FCGR Curve and a-N Curve After Data Augmentation(PNG)
      Nend Frequency Histogram(PNG)
      Extended a, N, C, and m data tables(CSV)
      Scatter Plot of Predicted Values vs. Actual Values(PNG)
4. Instructions for Use
  (1) Prepare your dataset in CSV format with the following columns:
      a, ρ, N
      where:
      a = crack length (mm)
      ρ = porosity (%)
      N = fatigue life (cycles)
  (2) Modify the configuration file (config/config.yaml) to specify:
      Dataset path
      Network structure (layers, neurons, learning rate, etc.)
      Loss weights for data fitting, Paris’ law, and monotonicity constraints
  (3) Run python MC_model.py
  (4) Run: python train.py
  (5) Run: python evalute.py