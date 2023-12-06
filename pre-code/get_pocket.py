# This module is used to obtain the absolute position of the product, pocket sequence, and pocket
import os
from get_protein import get_pro_seq

def get_poc_seq(pocket_path, protein_path):
    aa_codes = {
        'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E',
        'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
        'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N',
        'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S',
        'THR':'T', 'VAL':'V', 'TYR':'Y', 'TRP':'W'}
    poc_seq = ''
    i = ''  # locator
    position = []
    protein_seq, order = get_pro_seq(protein_path)
    for line in open(pocket_path):
        if line[0:4] == "ATOM":
            columns = line.split()
            index1 = columns[4]
            index2 = columns[5]
            if len(columns[4]) > 1: # When the residue sequence exceeds 1000, there will be no spaces between the chain and sequence, and manual separation is required
                index1 = columns[4][0]
                index2 = columns[4][1:]
            if index2 != i:
                i = index2
                position.append(order[(index1, index2)])
                poc_seq += aa_codes[columns[3]]
            else:
                continue

    return protein_seq, poc_seq, position