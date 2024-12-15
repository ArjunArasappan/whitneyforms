import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from torchdiffeq import odeint_adjoint as odeint

# 1. Generate synthetic data (Moons and Circles)
def generate_data(n_samples=1000, noise=0.05, data_type="moons"):
    if data_type == "moons":
        data, _ = make_moons(n_samples=n_samples, noise=noise)
    elif data_type == "circles":
        data, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    return torch.tensor(data, dtype=torch.float32)

# 2. Define a simple MLP for the vector field
class FlowNet(nn.Module):
    def __init__(self):
        super(FlowNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, t):
        t = t.expand(x.shape[0], 1)
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

# 3. Flow Matching Loss
def flow_matching_loss(flow_net, x0, xt, x1, t):
    with torch.no_grad():
        x_diff = x1 - x0  # The direction vector (x1 - x0)
    flow_pred = flow_net(xt, t)  # Predict the flow at xt, not x0
    loss = torch.mean((flow_pred - x_diff)**2)
    return loss

# 4. Training Loop
def train(flow_net, data, optimizer, n_steps=5000, sigma=0.1):
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Sample x0 from the normal distribution
        x0 = torch.randn((64, 2))

        # Sample a random t ∈ [0, 1]
        t = torch.rand((64, 1))

        # Generate xt by interpolating between x0 and target data points with noise
        idx = torch.randint(0, data.shape[0], (64,))
        x1 = data[idx] # target

        # Add Gaussian noise during interpolation
        noise = torch.randn_like(x0)
        xt = (1 - t) * x0 + t * x1 + sigma * torch.sqrt(t * (1 - t)) * noise

        # Compute loss and optimize
        loss = flow_matching_loss(flow_net, x0, xt, x1, t)
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

# 5. ODE Function for Inference
class ODEFunc(nn.Module):
    def __init__(self, flow_net):
        super(ODEFunc, self).__init__()
        self.flow_net = flow_net

    def forward(self, t, x):
        t_tensor = torch.full((x.shape[0], 1), t, dtype=torch.float32)
        return self.flow_net(x, t_tensor)

# 6. Visualization function
def visualize(flow_net, data, n_trials=64):
    with torch.no_grad():
        # Sample initial points from the normal distribution
        x0 = torch.randn((n_trials, 2))

        # Define the ODE function
        ode_func = ODEFunc(flow_net)

        # Integrate the ODE from t=0 to t=1
        t = torch.linspace(0, 1, 100)
        x_trajectory = odeint(ode_func, x0, t)

        # Plot the results
        plt.figure(figsize=(8, 6))
        plt.scatter(data[:, 0], data[:, 1], alpha=0.3, label='Target Data')

        plt.scatter(x_trajectory[-1, 0, 0], x_trajectory[-1, 0, 1], color='red', marker='o', s=25, label="x1")
        plt.scatter(x_trajectory[0, 0, 0], x_trajectory[0, 0, 1], color='black', marker='o', s=25, label="x0")
        for i in range(1, n_trials):
            plt.scatter(x_trajectory[-1, i, 0], x_trajectory[-1, i, 1], color='red', marker='o', s=25)
            plt.scatter(x_trajectory[0, i, 0], x_trajectory[0, i, 1], color='black', marker='o', s=25)

        plt.title("Flow Matching with ODE Integration")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Generate target double half-moon data
    data = generate_data(data_type="moons", n_samples=5000, noise=0.05)

    # Initialize the flow network and optimizer
    flow_net = FlowNet()
    optimizer = optim.Adam(flow_net.parameters(), lr=1e-2)

    # Train the flow network
    train(flow_net, data, optimizer, n_steps=4000, sigma=0.05)

    # Visualize the results with ODE integration
    visualize(flow_net, data, n_trials=64)