import os
import pandas as pd
path = os.path.abspath(os.path.dirname(os.getcwd()))
train_path = path + r'\data\seq_data_train.csv'
test_path = path + r'\data\seq_data_test.csv'
val_path = path + r'\data\seq_data_validation.csv'
path = [train_path, test_path, val_path]
smi_char_frequency = {}
# get the frequency of alphabet
smi_char = {}
for data_path in path:
    df = pd.read_csv(data_path)
    smile = list(df["Smile"])
    for a in smile:
        for x in a:
            if x not in smi_char_frequency:
                smi_char_frequency[x] = 0
            else:
                smi_char_frequency[x] += 1
smi_char_sorted = sorted(zip(smi_char_frequency.values(), smi_char_frequency.keys()), reverse=True)
i = 1
for x in smi_char_sorted:
    smi_char[x[1]] = i
    i += 1
print(smi_char)