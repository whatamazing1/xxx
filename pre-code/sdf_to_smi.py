from openbabel.pybel import readfile,Outputfile
import os

def MolFormatConversion(input_file,output_file,input_format="mol2",output_format="smi"):
    molecules = readfile(input_format,input_file)
    output_file_writer = Outputfile(output_format,output_file, overwrite=True)
    for i,molecule in enumerate(molecules):
        output_file_writer.write(molecule)
    output_file_writer.close()


path = os.path.abspath(os.path.dirname(os.getcwd()))
seq_path = path + f'\pre-data\core2016'
files = os.listdir(seq_path)
for pdb_file in files:
    # Obtain the path for each pdbid
    total_path = os.path.join(seq_path, pdb_file)
    # Obtain sdf information and smi file storage path for each ligand
    mol_path = os.path.join(total_path, pdb_file + r'_ligand.mol2')
    smi_path = os.path.join(total_path, pdb_file + r'_ligand.smi')
    # Convert to SMI format
    MolFormatConversion(mol_path, smi_path)