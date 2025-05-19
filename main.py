import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import torch.nn.utils
import pandas as pd
import numpy as np
from Models.Diffusion_TS import Diffusion_TS
from tqdm import tqdm
import time
import os
import argparse
from sklearn.preprocessing import MinMaxScaler

# Configurations
SEQ_LENGTH = 799
FEATURE_SIZE = 1
SF_DIM = 1
TEMP_DIM = 7
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-5
GRADIENT_ACCUMULATE_EVERY = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description="Train or sample from the Diffusion-TS model")
    parser.add_argument("--train", type=int, default=0, help="1 for training, 0 otherwise")
    parser.add_argument("--sample", type=int, default=0, help="1 for sampling, 0 otherwise")
    parser.add_argument("--test", type=int, default=0, help="1 for sampling, 0 otherwise")
    parser.add_argument("--test_dir", type=str, default="TEST", help="Directory to save results")
    parser.add_argument("--output_dir", type=str, default="OUTPUT", help="Directory to save results")
    return parser.parse_args()

args = parse_args()

# Load Data
sf_data = pd.read_csv('Data/StreamFlow.csv').values 
valid_data = sf_data[sf_data != -999.0]
mean_value = valid_data.mean()
sf_data[sf_data == -999.0] = mean_value
temp_data = pd.read_csv('Data/TEMP.csv').values 

# Use separate scalers for streamflow and temperature data
sf_scaler = MinMaxScaler(feature_range=(0, 1))  # For streamflow data
temp_scaler = MinMaxScaler(feature_range=(0, 1))  # For temperature data

# Scale the data separately
sf_data = sf_scaler.fit_transform(sf_data)
temp_data = temp_scaler.fit_transform(temp_data)

print('Size of sf before reshaping:', len(sf_data))
print('Size of temp before reshaping:', len(temp_data))

# Reshape Data
sf_tensor = torch.tensor(sf_data, dtype=torch.float32).reshape(-1, SEQ_LENGTH, SF_DIM)
temp_tensor = torch.tensor(temp_data, dtype=torch.float32).reshape(-1, SEQ_LENGTH, TEMP_DIM)

print('\nSize of sf after converting to tensor:', sf_tensor.shape)
print('Size of temp after converting to tensor:', temp_tensor.shape)

# Create Dataset and DataLoader
dataset = TensorDataset(sf_tensor, temp_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Instantiate Model
model = Diffusion_TS(
    seq_length=SEQ_LENGTH,
    feature_size=FEATURE_SIZE,
    sf_dim=SF_DIM,
    temp_dim=TEMP_DIM,
    n_layer_enc=3,
    n_layer_dec=6,
    d_model=96,
    timesteps=1000,
    sampling_timesteps=1000,
    loss_type='l1',
    beta_schedule='cosine',
    n_heads=4,
    mlp_hidden_times=4,
    reg_weight=0.1,
).to(DEVICE)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1000, min_lr=1e-6
)

# Start training
if args.train:
    print("Starting Training...")
    step = 0
    tic = time.time()
    output_model_path = os.path.join(args.output_dir, "trained_model.pth")
    os.makedirs(args.output_dir, exist_ok=True)

    with tqdm(initial=step, total=EPOCHS, desc="Training Progress", unit="batch") as pbar:
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for batch_idx, (sf, temp) in enumerate(dataloader):
                sf, temp = sf.to(DEVICE), temp.to(DEVICE)
                # Forward pass and calculate loss
                loss = model(sf, temp)
                loss = loss / GRADIENT_ACCUMULATE_EVERY  # Normalize loss for gradient accumulation
                loss.backward()

                if (batch_idx + 1) % GRADIENT_ACCUMULATE_EVERY == 0 or (batch_idx + 1) == len(dataloader):
                    # Clip gradients for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Optimizer step and clear gradients
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item() * GRADIENT_ACCUMULATE_EVERY  # Scale back loss for logging
            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item() * GRADIENT_ACCUMULATE_EVERY:.4f}")
            pbar.update(1)
            step += 1
            scheduler.step(epoch_loss)

            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, output_model_path)

    print(f"Training complete. Total time: {time.time() - tic:.2f}s")

# Sampling function
def sample(model, num, size_every, temp, seq_length, feature_size):
    print("Starting Sampling...")
    samples = np.empty([0, seq_length, feature_size])  # Predefine the structure of output samples
    num_cycle = int(num // size_every) + 1

    for c in range(num_cycle - 1):
        temp_batch = temp[c * size_every:(c + 1) * size_every].to(model.betas.device)
        batch_samples = model.generate_mts(
            temp=temp_batch,
            batch_size=size_every
        )

        # Convert to numpy and reshape for inverse scaling
        batch_np = batch_samples.detach().cpu().numpy()
        original_shape = batch_np.shape  # (B, SEQ_LEN, 1)

        # Flatten to 2D for inverse transform
        batch_flat = batch_np.reshape(-1, 1)

        # Inverse scaling using the streamflow scaler
        batch_inv = sf_scaler.inverse_transform(batch_flat)  # Now in original units

        # Reshape back and clamp PHYSICAL negatives (0 in original space)
        batch_inv = batch_inv.reshape(original_shape)
        batch_inv = np.clip(batch_inv, 0, None)  # Physical non-negativity

        samples = np.row_stack([samples, batch_inv])  # Store descaled, clamped values
        torch.cuda.empty_cache()

    print("Sampling complete.")
    return samples

if args.sample:
    # Load the trained model
    checkpoint = torch.load(os.path.join(args.output_dir, "trained_model.pth"), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode

    NUM_SAMPLES = len(dataset)
    BATCHES = 16

    generated_samples = sample(
        model=model,
        num=NUM_SAMPLES,
        size_every=BATCHES,
        temp=temp_tensor,
        seq_length=SEQ_LENGTH,
        feature_size=FEATURE_SIZE
    )

    # Save Generated Samples
    np.save(os.path.join(args.output_dir, "generated_samples.npy"), generated_samples)
    print(f"Generated Samples Shape: {generated_samples.shape}")

if args.test:
    # Load the trained model
    checkpoint = torch.load(os.path.join(args.output_dir, "trained_model.pth"), map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode

    l = os.listdir('Data/test/temp_test')
    os.makedirs(args.test_dir, exist_ok=True)
    for i in range(len(l)):
        temp_data = pd.read_csv('Data/test/temp_test/' + l[i]).values
        temp_data = temp_scaler.transform(temp_data)  # Use temp_scaler for consistency
        temp_tensor = torch.tensor(temp_data, dtype=torch.float32).reshape(-1, SEQ_LENGTH, TEMP_DIM)
        dataset = TensorDataset(temp_tensor)
        NUM_SAMPLES = len(dataset)
        BATCHES = 4
        generated_samples = sample(
            model=model,
            num=NUM_SAMPLES,
            size_every=BATCHES,
            temp=temp_tensor,
            seq_length=SEQ_LENGTH,
            feature_size=FEATURE_SIZE
        )

        # Save Generated Samples
        np.save(os.path.join(args.test_dir, f"test_basin{i + 1}.npy"), generated_samples)
        print(f"Generated Samples Shape: {generated_samples.shape}")