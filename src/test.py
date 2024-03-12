from torch.utils.data import DataLoader
import torch
import numpy as np
from torch import nn,optim
from bestmodel import MyModule
from Dataset import MyDataset
import sklearn.metrics as m
from numba import njit
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

def RMSE(y_true, y_pred):
    return np.sqrt(m.mean_squared_error(y_true, y_pred))
def MAE(y_true, y_pred):
    return m.mean_absolute_error(y_true, y_pred)
def MSE(y_true, y_pred):
    return m.mean_squared_error(y_true, y_pred)


@njit
def c_index(y_true, y_pred):
    summ = 0
    pair = 0

    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1

    if pair != 0:
        return summ / pair
    else:
        return 0

device = "cuda:0" if torch.cuda.is_available() else "cpu"

path = os.path.abspath(os.path.dirname(os.getcwd()))
data_path = path + r'/data'

batch_size = 1
max_seq_len = 1024
max_smi_len = 256

test_dataset = MyDataset('core2016', data_path, max_seq_len, max_smi_len)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = MyModule().to(device)

## Model fusion testing module
final_pre = []
target = []
id = []
for i in range(10):

    model.load_state_dict(torch.load(path + r"/src/model" + f'/Fold {i+1} best_model.ckpt'))
    model.eval()
    pre = []

    for id_name, smi_encode, seq_encode, pocket_encode, affinity in test_dataloader:
        with torch.no_grad():
            final_output = model(seq_encode, smi_encode, pocket_encode)
        pre.extend(final_output.cpu().numpy().reshape(-1)/10)
        if i == 0:
            id.extend(id_name)
            target.extend(affinity.cpu().numpy().reshape(-1))
    if i == 0:
        final_pre = pre
    else:
        final_pre = [i + j for i,j in zip(final_pre,pre)]


rmse = RMSE(final_pre, target)
mae = MAE(final_pre, target)
mse = MSE(final_pre, target)
cl = c_index(final_pre, target)
print(f"total RMSE:{rmse}total MAE:{mae}total MSE:{mse}total CL:{cl}")
df = pd.DataFrame({"id_name":id, 'true affinity': target, 'predict affinity': final_pre})
df.to_csv("kfold_result.csv", index = False)

## Multi model testing module
for i in range(10):

    model.load_state_dict(torch.load(path + r"/src/model" + f'/Fold {i+1} best_model.ckpt'))
    model.eval()
    pre = []
    target = []

    for id_name, smi_encode, seq_encode, pocket_encode, affinity in test_dataloader:
        with torch.no_grad():
            final_output = model(seq_encode, smi_encode, pocket_encode)
        pre.extend(final_output.cpu().numpy().reshape(-1))
        target.extend(affinity.cpu().numpy().reshape(-1))

    rmse = RMSE(pre, target)
    mae = MAE(pre, target)
    mse = MSE(pre, target)
    cl = c_index(pre, target)
    print(f"Fold {i+1} RMSE:" + str(rmse)+"MAE:" + str(mae)+"MSE:" + str(mse) + "CL:" + str(cl))


