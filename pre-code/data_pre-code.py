# Batch acquisition of protein sequence, pocket sequence, and absolute position information of pockets
import os
import pandas as pd
from get_pocket import get_poc_seq


path = os.path.abspath(os.path.dirname(os.getcwd()))
seq_path = path + f'\pre-data\core2016'
seq_data_path = path + r'\data\seq_data_core2016.csv'
files = os.listdir(seq_path)
pdb_id = []
pdb_smi = []
pdb_protein = []
pdb_pocket = []
pdb_position = []
for pdb_file in files:
    # Obtain the path for each pdbid
    total_path = os.path.join(seq_path, pdb_file)

    # Obtain smile information for each ligand
    mol_path = os.path.join(total_path, pdb_file + r'_ligand.smi')
    mol = open(mol_path)
    line = mol.readline()
    pdb_smi.append(line.split()[0])

    # Obtain information on each protein sequence, pocket sequence, and pocket location
    protein_path = os.path.join(total_path, pdb_file + r'_protein.pdb')
    pocket_path = os.path.join(total_path, pdb_file + r'_pocket.pdb')
    protein_seq, pocket_seq, position = get_poc_seq(pocket_path, protein_path)
    pdb_id.append(pdb_file)
    pdb_protein.append(protein_seq)
    pdb_pocket.append(pocket_seq)
    pdb_position.append(position)
data = {"PDBname":pdb_id, "Smile":pdb_smi, "Sequence":pdb_protein, "Pocket":pdb_pocket, "Position":pdb_position}
frame = pd.DataFrame(data)
frame.to_csv(seq_data_path)