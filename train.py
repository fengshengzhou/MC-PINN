import torch
import numpy as np
import random
import pandas as pd
from network import Network
from data_utils import min_max_scale
from datetime import datetime
import os
import matplotlib.pyplot as plt
from losses import combined_loss

INPUT_NUMBER = 4
OUYPUT_NUMBER = 5

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PINN:
    def __init__(self, csv_file):
        self.device = torch.device("cpu")
        self.model = Network(
            input_size=4,
            hidden_size=64,
            output_size=1,
            depth=8,
            # act=torch.nn.GELU
            act=torch.nn.Tanh
            # act = torch.nn.LeakyReLU
        ).to(self.device)
        self.model.apply(self.init_weights)


        data = pd.read_csv(csv_file).values

        inputs, self.input_min, self.input_max = min_max_scale(data[:, :INPUT_NUMBER])


        outputs, self.output_min, self.output_max = min_max_scale(data[:, INPUT_NUMBER:OUYPUT_NUMBER])

        self.X_train = torch.tensor(inputs, dtype=torch.float32, requires_grad=True).to(self.device)
        self.Y_train = torch.tensor(outputs, dtype=torch.float32).to(self.device)

        # 记录迭代次数
        self.iter = 1

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)

        self.losses_history = pd.DataFrame(columns=[
            "Epoch", "Total_Loss", "MSE_Loss", "LA_Area_Loss", "Paris_Loss"
        ])

    def init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def loss_func(self):
        self.optimizer.zero_grad()

        y_pred = self.model(self.X_train)


        total_loss, mse_val, la_area_val, paris_val , rank_val= combined_loss(

            y_pred, self.Y_train, self.X_train,
            self.input_min[2], self.input_max[2],
            self.input_min[0], self.input_max[0],
            clamp_min=0, alpha=0.01,
            mse_weight=1,
            paris_weight=0.01,
            rank_weight=0.2,
            margin=0.00001,
            # paris_weight=0,
            # rank_weight=0,
            # margin=0,
            rho_idx=3,
            verbose=False
        )

        total_loss.backward()

        self.optimizer.step()


        self.losses_history.loc[len(self.losses_history)] = [
            self.iter, total_loss.item(), mse_val.item(), la_area_val.item(), paris_val.item()
        ]

        if self.iter % 100 == 0:
            print(f"Iteration {self.iter}, Loss: {total_loss.item():.6f}, "
                  f"MSE: {mse_val.item():.6f}, Area: {la_area_val.item():.6f}, "
                  f"Paris: {paris_val.item():.6f}, Rank: {rank_val.item():.6f}")

        self.iter += 1
        return total_loss

    def train(self, epochs=1000):

        self.model.train()
        for epoch in range(epochs):
            self.optimizer.step(self.loss_func)


        export_folder = "losses_history"
        os.makedirs(export_folder, exist_ok=True)


        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳，格式为YYYYMMDD_HHMMSS
        losses_filename = os.path.join(export_folder, f"{timestamp}_losses_history.csv")  # 文件名


        self.losses_history.to_csv(losses_filename, index=False)
        print(f": {losses_filename}")


set_seed(42)
csv_file = "MC-expand data.csv"
pinn = PINN(csv_file)
pinn.train(epochs=12000)
torch.save(pinn.model, 'model.pth')


plt.figure(figsize=(10, 6))
plt.plot(pinn.losses_history["Epoch"], pinn.losses_history["Total_Loss"], label="Total Loss")
plt.plot(pinn.losses_history["Epoch"], pinn.losses_history["MSE_Loss"], label="MSE Loss")
plt.plot(pinn.losses_history["Epoch"], pinn.losses_history["LA_Area_Loss"], label="LA Area Loss")
plt.plot(pinn.losses_history["Epoch"], pinn.losses_history["Paris_Loss"], label="Paris Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.show()