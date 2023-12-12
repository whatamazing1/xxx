# DEAttentionDTA
DEAttentionDTA is a deep learning which based on dynamic word embeddings and a self-attention mechanism, for predicting protein-target binding affinity.
The raw data can be found at [PDBbind](http://pdbbind.org.cn/) including three datasets: version 2020,core2016 and core2014.
Data preprocessing can be referred to `./precode/`. The processed data is stored in `./data/`.
version 1 has active site information, the version 1 code and result can be found in `./src/`, model can be found in `./src/model/`.
version 2 has no active site information, the version 2 code and result can be found in `./src-v2/`, model can be found in `./src-v2/model/`.
# Requirements
- python 3.9.12

- numpy 1.21.5

- pytorch 1.12.1

- pandas 1.4.3

- sklearn 1.1.2

- tqdm 4.64.0

- numba 0.55.1

- Cuda 11.4.100



You can get the DEAttentionDTA code by:
```
git clone https://github.com/whatamazing1/DEAttentionDTA
```
# Training & Testing
Select a version and enter its root directory
```
cd ./src/
```
If you want to train this model, try:
```
python main.py
```
The train result can be found in `./src/kfold-result.csv`

If you want to test the existed model, try:
```
python test.py
```
The test result can be found in `./src/test-result.csv`
