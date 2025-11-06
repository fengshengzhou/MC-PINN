import torch
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from data_utils import min_max_scale, inverse_min_max_scale
from sklearn.metrics import r2_score
import numpy as np
from datetime import datetime
import os


INPUT_NUMBER = 4
OUTPUT_NUMBER = 5
Crack_Length = 0
Sigma_max = 1
Porosity_COL_INDEX = 2
Fatigue_Life = 3


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


device = torch.device("cpu")


model_loaded = torch.load('model.pth', map_location=device)
model_loaded.eval()

csv_file = "a-N.csv"
test_data = pd.read_csv(csv_file).values


inputs, input_min, input_max = min_max_scale(test_data[:, :INPUT_NUMBER])


outputs, output_min, output_max = min_max_scale(test_data[:, INPUT_NUMBER:OUTPUT_NUMBER])


X_test = torch.tensor(inputs, dtype=torch.float32).to(device)
Y_true = torch.tensor(outputs, dtype=torch.float32).to(device)


with torch.no_grad():
    Y_pred = model_loaded(X_test)


mse_norm = torch.nn.MSELoss()(Y_pred, Y_true)
mae_norm = torch.nn.L1Loss()(Y_pred, Y_true)


Y_pred_original = inverse_min_max_scale(Y_pred.cpu().numpy(), output_min, output_max)
Y_true_original = test_data[:, INPUT_NUMBER:OUTPUT_NUMBER]


mse_orig = torch.nn.MSELoss()(
    torch.tensor(Y_pred_original, dtype=torch.float32),
    torch.tensor(Y_true_original, dtype=torch.float32)
)
mae_orig = torch.mean(
    torch.abs(
        torch.tensor(Y_pred_original, dtype=torch.float32) -
        torch.tensor(Y_true_original, dtype=torch.float32)
    )
).item()


r2 = r2_score(Y_true_original, Y_pred_original)


print(f"Normalized-space MSE: {mse_norm.item():.6f}")
print(f"Normalized-space MAE: {mae_norm.item():.6f}")
print(f"Original-scale MSE: {mse_orig.item():.4f}")
print(f"Original-scale MAE: {mae_orig:.4f}")
print(f"Test R²: {r2:.4f}")


plt.figure(figsize=(10, 6))
plt.scatter(Y_true_original, Y_pred_original, label="Predicted vs True", alpha=0.5)
min_val = min(Y_true_original.min(), Y_pred_original.min())
max_val = max(Y_true_original.max(), Y_pred_original.max())
plt.plot([min_val, max_val], [min_val, max_val],
         label="Ideal (y = x)", linestyle='--', color='red')
plt.text(min_val, max_val - 0.1*(max_val-min_val),
         f"Orig MSE: {mse_norm.item():.4f}\nOrig MAE: {mae_norm:.4f}\nR²: {r2:.4f}",
         fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.legend()
plt.title("True vs Predicted (Original Scale)")
plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.show()


export_folder = "exported_results"
os.makedirs(export_folder, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
prediction_filename = os.path.join(export_folder, f"{timestamp}_prediction_results.csv")
metrics_filename   = os.path.join(export_folder, f"{timestamp}_metrics_results.csv")


export_data = np.column_stack((Y_true_original, Y_pred_original))
export_df = pd.DataFrame(export_data, columns=["True_Value", "Predicted_Value"])
export_df.to_csv(prediction_filename, index=False)


metrics_data = {
    "mse_norm": mse_norm.item(),
    "mae_norm": mae_norm.item(),
    "mse_orig": mse_orig.item(),
    "mae_orig": mae_orig,
    "r2": r2
}
metrics_df = pd.DataFrame([metrics_data])
metrics_df.to_csv(metrics_filename, index=False)

print(f"Results exported to:\n  {prediction_filename}\n  {metrics_filename}")

