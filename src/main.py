from torch.utils.data import DataLoader
import torch
import numpy as np
from torch import nn,optim
from bestmodel import MyModule
from Dataset import MyDataset
import sklearn.metrics as m
from tqdm import tqdm
import os

def RMSE(y_true, y_pred):
    return np.sqrt(m.mean_squared_error(y_true, y_pred))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 32
epochs = 50
path = os.path.abspath(os.path.dirname(os.getcwd()))
data_path = path + r'/data'

max_seq_len = 1024
max_smi_len = 256

seed = 990721

# GPU uses cudnn as backend to ensure repeatable by setting the following (in turn, use advances function to speed up training)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(seed)
np.random.seed(seed)

best_rmse = 10000
train_loss = val_loss = 0


from sklearn.model_selection import KFold

# Define the number of folds for k-fold cross validation
k_folds = 10

# Create a KFold object to partition the number of folds for k-fold cross validation
kf = KFold(n_splits=k_folds, shuffle=True)


train_dataset = MyDataset('kfold',data_path, max_seq_len, max_smi_len)
# Loop through each fold
for fold, (train_index, val_index) in enumerate(kf.split(train_dataset)):
    print(f"Fold {fold + 1}/{k_folds}")

    # Divide the training set and validation set based on fold number
    train_subset = torch.utils.data.Subset(train_dataset, train_index)
    val_subset = torch.utils.data.Subset(train_dataset, val_index)

    # Create a data loader for training and validation datasets
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Reinitialize the model before each fold begins
    model = MyModule().to(device)

    # use the corresponding data loader during training and validation
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    best_rmse = 10000
    train_loss = val_loss = 0

    # Define parameters related to early stop method
    early_stopping_rounds = 3  # Stop training when there is no improvement for three consecutive epochs
    best_val_loss = float('inf')  # The initial best validation set loss is infinite
    rounds_without_improvement = 0  # Number of epochs without continuous improvement

    for epoch in range(epochs):
        model.train()
        for id_name, smi_encode, seq_encode, pocket_encode, affinity in tqdm(train_dataloader,
                                                                    desc=f"Training Epoch {epoch + 1}"):
        # train
            final_output = model(seq_encode, smi_encode, pocket_encode)
            loss = criterion(final_output, affinity)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate the training loss and validation loss of the fold number
        train_loss /= len(train_dataloader.dataset)

        model.eval()
        pre = []
        target = []
        for id_name, smi_encode, seq_encode, pocket_encode, affinity in val_dataloader:
            with torch.no_grad():
                final_output = model(seq_encode, smi_encode, pocket_encode)
            loss = criterion(final_output, affinity)
            val_loss += loss.item()
            pre.extend(final_output.cpu().numpy().reshape(-1))
            target.extend(affinity.cpu().numpy().reshape(-1))

        rmse = RMSE(pre, target)
        val_loss /= len(val_dataloader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_rmse = rmse
            rounds_without_improvement = 0
            # Store the current best model weight
            torch.save(model.state_dict(), f'./model/Fold {fold + 1} best_model.ckpt')
            print('saving model with RMSE {:.3f}'.format(best_rmse))
            print('Saving model with improved validation loss')
            
        elif train_loss > val_loss:
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= early_stopping_rounds:
                print('Early stopping: validation loss has not improved in {} rounds'.format(rounds_without_improvement))
                break

        # Training loss and validation loss of printed book fold
        print(f'Fold %d - Train Loss: %.4f, Val Loss: %.4f' % (fold + 1, train_loss, val_loss))

