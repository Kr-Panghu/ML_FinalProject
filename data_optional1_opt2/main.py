import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from dataset import get_data, normalize

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256 * 15 * 15, self.latent_dim)
        self.fc_logvar = nn.Linear(256 * 15 * 15, self.latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)         # 输入x通过编码器的卷积层处理，得到一系列特征图
        x = x.view(x.size(0), -1)   # 通过view方法将特征图展平为一维向量
        mu = self.fc_mu(x)          # 通过全连接层fc_mu计算潜在向量的均值mu
        logvar = self.fc_logvar(x)  # 通过全连接层fc_logvar计算潜在向量的方差logvar
        return mu, logvar
    
    def reparameterize(self, mu, logvar):   # 重参数化
        std = torch.exp(0.5 * logvar)       # 计算标准差std
        eps = torch.randn_like(std)         # 从标准正态分布中采样随机噪声eps
        z = mu + eps * std                  # 使用这个随机噪声和标准差计算出潜在向量z
        return z
    
    def decode(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1)    # 输入的潜在向量z被重新调整为适合解码器的形状
        x_recon = self.decoder(z)                       # 通过解码器的卷积转置层将潜在向量解码为重构输出x_recon
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)         # 输入的数据x经过编码器部分得到潜在向量的均值mu和方差logvar
        z = self.reparameterize(mu, logvar) # 调用重参数化方法获得潜在向量z
        x_recon = self.decode(z)            # 将潜在向量z通过解码器部分解码为重构输出x_recon
        return x_recon, mu, logvar

# Define the training loop
def train_vae(model, dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    
    for batch_idx, data in enumerate(dataloader):
        inputs = data[0].to(device)
        
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(inputs)
        loss = criterion(recon_batch, inputs, mu, logvar)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
    train_loss /= len(dataloader.dataset)
    return train_loss

# Define the loss function for VAE
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

if __name__ == '__main__':
    ######################## Get train dataset ########################
    X_train = get_data('dataset')
    ########################################################################
    ######################## Implement your code here #######################
    ########################################################################

    # Convert data to torch.Tensor and create DataLoader
    X_train = torch.from_numpy(X_train).permute(0, 1, 2, 3)  # Reshape to (N, C, H, W)
    train_dataset = TensorDataset(X_train)
    batch_size = 128
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create VAE model and move it to the device
    latent_dim = 128
    model = VAE(latent_dim).to(device)

    # Set optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train_vae(model, train_dataloader, optimizer, vae_loss, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")
        scheduler.step()

    # Save the trained model
    # save_path = "vae_model.pth"
    # torch.save(model.state_dict(), save_path)
