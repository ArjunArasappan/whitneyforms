import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Model configuration")

    parser.add_argument('--latent_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    
    return parser.parse_args()




data = torch.load('/Users/arjunarasappan/Documents/PIMI/whitneyforms/point_poisson_lloyd_data_50k_1x.pt')






parameters = data['parameters']
solutions = data['fine_scale_mesh_solns']
coordinates = data['point_source_coords']


test_split = 0.8

args = parse_args()
latent_size = args.latent_size
EPOCHS = args.epochs




parameters = (parameters - parameters.mean(dim=0)) / parameters.std(dim=0)
coordinates = (coordinates - coordinates.mean(dim=0)) / coordinates.std(dim=0)
solutions = (solutions ) / solutions.std(dim=0)

parameters = torch.nan_to_num(parameters, nan=0.0)
coordinates = torch.nan_to_num(coordinates, nan=0.0)
solutions = torch.nan_to_num(solutions, nan=0.0)


torch.manual_seed(10)
num_samples = parameters.size(0)
split_idx = int(num_samples * test_split)
indices = torch.randperm(num_samples)


train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

print("Train Size", split_idx)
print("Val Size", num_samples -  split_idx)

param_train, param_val = parameters[train_indices], parameters[val_indices]
coord_train, coord_val = coordinates[train_indices], coordinates[val_indices]
sol_train, sol_val = solutions[train_indices], solutions[val_indices]

param_train = param_train.float()
param_val = param_val.float()
coord_train = coord_train.float()
coord_val = coord_val.float()
sol_train = sol_train.float()
sol_val = sol_val.float()



batch_size = 512
num_workers = 0

param_soln_trainloader = DataLoader(TensorDataset(param_train, sol_train), batch_size=batch_size, shuffle=True, num_workers=num_workers)
param_cord_trainloader = DataLoader(TensorDataset(param_train, coord_train), batch_size=batch_size, shuffle=True, num_workers=num_workers)
coord_soln_trainloader = DataLoader(TensorDataset(coord_train, sol_train), batch_size=batch_size, shuffle=True, num_workers=num_workers)

ssl_trainloader = DataLoader(TensorDataset(sol_train), batch_size=batch_size, shuffle=True, num_workers=num_workers)



    
    
class solution_enc(nn.Module):
    def __init__(self):
        super(solution_enc, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(solutions.shape[1],  128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_size)
        )
    
    def forward(self, x):
        return self.layers(x)

class solution_dec(nn.Module):
    def __init__(self):
        super(solution_dec, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, solutions.shape[1])
        )
    
    def forward(self, x):
        return self.layers(x)
    
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.enc = solution_enc()
        self.dec = solution_dec()
    
    def forward(self, x):
        latent = self.enc(x)
        output = self.dec(latent)
        return latent, output
    
    
autoencoder = AE()
criterion = nn.MSELoss()
optimizerAE = optim.Adam(autoencoder.parameters(), lr = 0.001)



def joint_train(AE, opt, criterion, trainloader, val_data, val_labels, epochs=100):
    train_losses = []
    val_losses = []
    val_latents = None
    
    
    for epoch in range(epochs):
        AE.train()
        loss, count = 0, 0
        for batch in trainloader:
            soln_train = batch[0]

            opt.zero_grad()
            latents, outputs = AE(soln_train)
            train_loss = criterion(outputs, soln_train)
            train_loss.backward()
            opt.step()
            loss += train_loss.item()
            count += 1


        train_losses.append(loss / count)
        
        if epoch % 10 == 0:
            AE.eval()
            with torch.no_grad():
                lat, val_outputs = AE(val_data)
                val_loss = criterion(val_outputs, val_labels)
                
                val_latents = lat
                
                coord_val 
                val_losses.append(val_loss.item())
        else:
            val_losses.append(val_losses[-1])
        
        if epoch % 1 == 0:
            print(f"Epoch [{epoch}/{epochs}], Train Loss: {train_loss.item():.4f}, Val Loss: {val_losses[-1]}")
              
            
    
    return train_losses, val_losses, val_latents




def linear_convert(x, y):

    X = torch.cat([torch.ones(x.shape[0], 1), x], dim=1)
    
    
    XtX = X.T @ X 
    XtX_inv = torch.linalg.inv(XtX) 
    W = XtX_inv @ X.T @ y 
    
    y_pred = X @ W
    return y_pred, W

def polynomial_convert(x, y, degree=2):
    # Generate polynomial features
    poly_features = [torch.ones(x.shape[0], 1)]  # Start with the bias term (intercept)
    for d in range(1, degree + 1):
        poly_features.append(x ** d)  # Add polynomial terms up to the specified degree
        print(poly_features[-1].shape)


    poly_features.append((x[:, 0] * x[:, 1]).reshape((x.shape[0], 1)))
    print(poly_features[-1].shape)
    
    
    # Concatenate the features into a single matrix
    X = torch.cat(poly_features, dim=1)
    
    # Compute the closed-form solution
    XtX = X.T @ X
    XtX_inv = torch.linalg.inv(XtX)
    W = XtX_inv @ X.T @ y
    
    # Predict the output
    y_pred = X @ W
    return y_pred, W






    


print(f"Joint training of AE (solutions -> latent{latent_size} -> solutions):")
train_losses3, val_losses3, latents = joint_train(autoencoder, optimizerAE, criterion, ssl_trainloader, sol_val, sol_val, epochs = EPOCHS)


savepath = f'weights/ssl_ae{latent_size}.pth'
torch.save(autoencoder.state_dict(), savepath)


plt.figure(figsize=(6 * (3), 5))
plt.subplot(1, 3 + latent_size, 1)
plt.plot(train_losses3, label='Train Loss')
plt.plot(val_losses3, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('MLP1 -> MLP 2 (params -> solutions)')
plt.yscale('log')
plt.xscale('log')
plt.legend()

y_pred, _ = polynomial_convert(latents, coord_val)

y_pred_np = y_pred
coord_val_np = coord_val

# for i in range(latent_size):
#     plt.subplot(1, 3 + latent_size, 2 + i)
#     plt.scatter(latents[:, 0], coord_val_np[:, 0], label="x", alpha=0.5)
#     plt.xlabel(f"latent coordinate {i}")
#     plt.ylabel("ground truth coordinate")
#     plt.yscale('log')
#     plt.xscale('log')
#     plt.legend()
#     plt.title("Predicted vs True Coordinates")



plt.subplot(1, 3, 2)
plt.scatter(y_pred[:, 0], coord_val_np[:, 0], label="x", alpha=0.5)
plt.xlabel("linear prediction")
plt.ylabel("ground truth coordinate")
plt.legend()
plt.title("Predicted vs True Coordinates")

plt.subplot(1, 3, 3 )
plt.scatter(y_pred[:, 1], coord_val_np[:, 1], label="y", alpha=0.5)
plt.xlabel("linear prediction")
plt.ylabel("ground truth coordinate")
plt.legend()
plt.title("Predicted vs True Coordinates")

plt.savefig(f'./plots/latent{latent_size}.png')



