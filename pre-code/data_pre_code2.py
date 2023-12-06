# Batch acquisition of protein sequence, pocket sequence, and absolute position information of pockets
import os
import pandas as pd


aa_codes = {
     'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
     'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
     'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
     'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W'}

def get_pro_seq1(path):
    seq = ''
    for line in open(path):
        if line[0:6] == "SEQRES":
            columns = line.split()
            for resname in columns[3:]:
                if resname in aa_codes:
                    seq = seq + aa_codes[resname]
    return seq

path = os.path.abspath(os.path.dirname(os.getcwd()))
seq_path = path + r'\pre-data\core2013'
seq_data_path = path + r'\data\seq_data_core2013.csv'
files = os.listdir(seq_path)
pdb_id = []
pdb_smi = []
pdb_protein = []
for pdb_file in files:
    if pdb_file == 'index':
        continue
    # Obtain the path for each pdbid
    total_path = os.path.join(seq_path, pdb_file)

    # Obtain smile information for each ligand
    mol_path = os.path.join(total_path, pdb_file + r'_ligand.smi')
    mol = open(mol_path)
    line = mol.readline()
    pdb_smi.append(line.split()[0])

    # Obtain information on each protein sequence, pocket sequence, and pocket location
    protein_path = os.path.join(total_path, pdb_file + r'_protein.pdb')
    protein_seq = get_pro_seq1(protein_path)

    pdb_id.append(pdb_file)
    pdb_protein.append(protein_seq)

data = {"PDBname":pdb_id, "Smile":pdb_smi, "Sequence":pdb_protein}
frame = pd.DataFrame(data)
frame.to_csv(seq_data_path)