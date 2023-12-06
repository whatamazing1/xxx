# Obtain the affinity corresponding to each protein
import os
import pandas as pd
# Program root directory
path = os.path.abspath(os.path.dirname(os.getcwd()))
# index directory
index_path = path + r'\pre-data\PDBbind_index\index\INDEX_general_PL_data.2020'
# Affinity storage directory
affinity_data_path = path + r'\data\affinity.csv'
with open(index_path,'r') as file:
    file = file.readlines()
    name = []
    affinity = []
# The first six lines are not valid messages
    for i in range(6,len(file)):
        _content = file[i].split('  ')
        name.append(str(_content[0]))
        affinity.append(_content[3])
    data = {'PDBname':name,'affinity':affinity}
    frame = pd.DataFrame(data)
    frame.to_csv(affinity_data_path)

