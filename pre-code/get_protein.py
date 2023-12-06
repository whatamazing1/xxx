# This module is used to obtain the absolute position information of the entire protein sequence and pockets
def get_pro_seq(path):
    aa_codes = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
        'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W'}
    seq = ''
    order = {}
    i = 0  # locator
    for line in open(path):
        if line[0:4] == "ATOM":
            columns = line.split()
            index1 = columns[4]
            index2 = columns[5]
            if len(columns[4]) > 1: # When the residue sequence exceeds 1000, there will be no spaces between the chain and sequence, and manual separation is required
                index1 = columns[4][0]
                index2 = columns[4][1:]
            if (index1, index2) not in order:
                i = i + 1
                order[(index1, index2)] = i
                seq += aa_codes[columns[3]]
            else:
                continue
    return seq, order

