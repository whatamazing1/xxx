from torch.utils.data import Dataset
import pandas as pd
import torch
from pathlib import Path
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

smi_char = {'<MASK>': 0,'C': 1, ')': 2, '(': 3, 'c': 4, 'O': 5, ']': 6, '[': 7,
             '@': 8, '1': 9, '=': 10, 'H': 11, 'N': 12, '2': 13, 'n': 14,
             '3': 15, 'o': 16, '+': 17, '-': 18, 'S': 19, 'F': 20, 'p': 21,
             'l': 22, '/': 23, '4': 24, '#': 25, 'B': 26, '\\': 27, '5': 28,
             'r': 29, 's': 30, '6': 31, 'I': 32, '7': 33, '%': 34, '8': 35,
             'e': 36, 'P': 37, '9': 38, 'R': 39, 'u': 40, '0': 41, 'i': 42,
             '.': 43, 'A': 44, 't': 45, 'h': 46, 'V': 47, 'g': 48, 'b': 49,
             'Z': 50, 'T': 51, 'M': 52}
portein_char = {'<MASK>': 0,'A': 1, 'C': 2, 'D': 3, 'E': 4,
                'F': 5, 'G': 6, 'H': 7, 'K': 8,
                'I': 9, 'L': 10, 'M': 11, 'N': 12,
                'P': 13, 'Q': 14, 'R': 15, 'S': 16,
                'T': 17, 'V': 18, 'Y': 19, 'W': 20}

# Return an array with the same dimension as seq and all<MASK>except for position
def position_seq(seq, position):
    res = ['<MASK>']*len(seq)
    for i in position:
        res[i-1] = seq[i-1]
    return res
# Return an array with a length of max_ SMI_ Len, obtain the SMILES encoding format of the input sequence (1 * 256), use 0 as padding for sentence length filling
def label_smiles(line, max_smi_len):
    label = np.zeros(max_smi_len)
    for i, lab in enumerate(line[:max_smi_len]):
        label[i] = smi_char[lab]
    return label
# Return an array with a length of max_ Seq_ Len, obtain the seq encoding format of the input sequence (1 * 1024)
def label_seq(line, max_seq_len):
    label = np.zeros(max_seq_len)
    for i, lab in enumerate(line[:max_seq_len]):
        label[i] = portein_char[lab]
    return label

device = "cuda:0" if torch.cuda.is_available() else "cpu"
class MyDataset(Dataset):
    def __init__(self, type, data_path, max_seq_len, max_smi_len):
        super().__init__()
        data_path = Path(data_path)
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len
        self.type = type

        affinity_path = data_path / 'affinity.csv'
        affinity_data = pd.read_csv(affinity_path, index_col=0)
        affinity = {}
        for _, row in affinity_data.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity

        seq_path = data_path / f'seq_data_{type}.csv'
        seq_data = pd.read_csv(seq_path)
        smile = {}
        sequence = {}
        position = {}
        idx = {}
        i = 0
        for _, row in seq_data.iterrows():
            # if len(row[3]) < 1536 and eval(row[5])[0] != 1:
            idx[i] = row[1]
            smile[row[1]] = row[2]
            sequence[row[1]] = row[3]
            # 读取的position是字符串格式的，需要eval函数进行转换
            position[row[1]] = eval(row[5])
            i += 1
        self.position = position
        self.smile = smile
        self.sequence = sequence
        self.idx = idx
        assert len(sequence) == len(smile)
        self.len = len(self.sequence)

    def __getitem__(self, index):
        global device
        id_name = self.idx[index]
        pos = self.position[id_name]

        smi = self.smile[id_name]
        seq = self.sequence[id_name]
        pocket = position_seq(seq, pos)

        smi_encode = torch.tensor(label_smiles(smi, self.max_smi_len), device=device).long()
        seq_encode = torch.tensor(label_seq(seq, self.max_seq_len), device=device).long()
        pocket_encode = torch.tensor(label_seq(pocket, self.max_seq_len), device=device).long()
        affinity = torch.tensor(np.array(self.affinity[id_name], dtype=np.float32), device=device)

        return id_name, smi_encode, seq_encode, pocket_encode, affinity

    def __len__(self):
        return self.len
