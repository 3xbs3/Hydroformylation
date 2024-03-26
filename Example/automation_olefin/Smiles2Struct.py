from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import ComputeSignedDihedralAngle
import warnings
import pandas as pd
import numpy as np
import argparse
import re

warnings.filterwarnings("ignore")

def gen_smiles(ligand, metal):
    '''This definition is to generate SMILES of transition metal complexes.
    
    Args:
        ligand(str): Ligand.
        metal(str): Metal.
    Return: 
        str: SMILES of Transition metal complexes.
    '''
    comb = 0
    num_P = len(re.findall(r'P', ligand))
    if num_P == 1:
        result = re.split(r"P", ligand)
        if metal == 'Rh':
            comb = result[0] + 'P->%16' + result[1] + '.' + result[0] + 'P->%17' + result[1] + '.[RhH]%16%17' \
                                '%18.[C-]->%18#[O+]'
        if metal == 'Co':
            comb = result[0] + 'P->%16' + result[1] + '.' + result[0] + 'P->%17' + result[1] + '.[CoH]%16%17' \
                                '%18%19.[C-]->%18#[O+].[C-]->%19#[O+]'
        if metal == 'Pt':
            comb = result[0] + 'P->%16' + result[1] + '.' + result[0] + 'P->%17' + result[1] + '.[PtH]%16%17' \
                                '[Sn](Cl)(Cl)Cl'
        if metal == 'Ir':
            comb = result[0] + 'P->%16' + result[1] + '.' + result[0] + 'P->%17' + result[1] + '.[IrH]%16%17%18' \
                                '.[C-]->%18#[O+]'
        if metal == 'Pd':
            comb = result[0] + 'P->%16' + result[1] + '.' + result[0] + 'P->%17' + result[1] + '.[PdH]%16%17I'
    if num_P == 2:
        result = re.split(r"P", ligand)
        if metal == 'Rh':
            comb = result[0] + 'P->%16' + result[1] + 'P->%17' + result[2] + '.[RhH]%16%17%18.' \
                                '[C-]->%18#[O+]'
        if metal == 'Co':
            comb = result[0] + 'P->%16' + result[1] + 'P->%17' + result[2] + '.[CoH]%16%17%18%19.' \
                                '[C-]->%18#[O+].[C-]->%19#[O+]'
        if metal == 'Pt':
            comb = result[0] + 'P->%16' + result[1] + 'P->%17' + result[2] + '.[PtH]%16%17' \
                                '[Sn](Cl)(Cl)Cl'
        if metal == 'Ir':
            comb = result[0] + 'P->%16' + result[1] + 'P->%17' + result[2] + '.[IrH]%16%17%18' \
                                '.[C-]->%18#[O+]'
        if metal == 'Pd':
            comb = result[0] + 'P->%16' + result[1] + 'P->%17' + result[2] + '.[PdH]%16%17I'
    return comb
    
def deal_di_axial_chirality(mol2, patt, index):

    '''This definition is to assess the dihedral angle of specific transition metal complexes.
    
    Args:
        mol2(rdkit.Chem.rdchem.Mol):The structures required an assessment..
        patt(str): Smart.
        index(array): Substructures' index.
    Return: 
        float: dihedral angle.
    '''
    AllChem.EmbedMultipleConfs(mol2, numConfs=1, params=params)

    if patt == patt1: # calculate the dihedral angle
        di_angle_1 = ComputeSignedDihedralAngle(mol2.GetConformer().GetAtomPosition(index[0][0])
                                                , mol2.GetConformer().GetAtomPosition(index[0][9])
                                                , mol2.GetConformer().GetAtomPosition(index[0][10])
                                                , mol2.GetConformer().GetAtomPosition(index[0][11])
                                                )
        
        if len(index) == 2:
            di_angle_2 = ComputeSignedDihedralAngle(mol2.GetConformer().GetAtomPosition(index[1][0])
                                                    , mol2.GetConformer().GetAtomPosition(index[1][9])
                                                    , mol2.GetConformer().GetAtomPosition(index[1][10])
                                                    , mol2.GetConformer().GetAtomPosition(index[1][11])
                                                    )
            return di_angle_1, di_angle_2
        
        if len(index) == 1:
            return di_angle_1
    
    if patt == patt2:
        di_angle_1 = ComputeSignedDihedralAngle(mol2.GetConformer().GetAtomPosition(index[0][2])
                                                , mol2.GetConformer().GetAtomPosition(index[0][8])
                                                , mol2.GetConformer().GetAtomPosition(index[0][9])
                                                , mol2.GetConformer().GetAtomPosition(index[0][14])
                                                )
        
        if len(index) == 2:
            di_angle_2 = ComputeSignedDihedralAngle(mol2.GetConformer().GetAtomPosition(index[1][2])
                                                    , mol2.GetConformer().GetAtomPosition(index[1][8])
                                                    , mol2.GetConformer().GetAtomPosition(index[1][9])
                                                    , mol2.GetConformer().GetAtomPosition(index[1][14])
                                                    )
            return di_angle_1, di_angle_2
        
        if len(index) == 1:
            return di_angle_1

def gen_confor(mol2, patt):

    '''This definition is to generate the initial 3D structures of specific transition metal complexes.
    
    Args:
        mol2(rdkit.Chem.rdchem.Mol):The structures required an assessment.
        patt(str): Smart.
    '''
    result = re.findall(r".*?\[(C@*)\].*?", smile) # match chiral sign
    index = mol2.GetSubstructMatches(patt) # the matched structures' index
    
    if len(result) == 4:
        if result[0] == 'C@' and result[2] == 'C@':
            while True:
                di_angle_1, di_angle_2 = deal_di_axial_chirality(mol2, patt, index, number)
                if di_angle_1 < 0 and di_angle_2 < 0:
                    break
        if result[0] == 'C@' and result[2] == 'C@@':
            while True:
                di_angle_1, di_angle_2 = deal_di_axial_chirality(mol2, patt, index, number)
                if di_angle_1 < 0 and di_angle_2 > 0:
                    break
        if result[0] == 'C@@' and result[2] == 'C@@':
            while True:
                di_angle_1, di_angle_2 = deal_di_axial_chirality(mol2, patt, index, number)
                if di_angle_1 > 0 and di_angle_2 > 0:
                    break
        if result[0] == 'C@@' and result[2] == 'C@':
            while True:
                di_angle_1, di_angle_2 = deal_di_axial_chirality(mol2, patt, index, number)
                if di_angle_1 > 0 and di_angle_2 < 0:
                    break
    
    elif len(result) == 2:
        if result[0] == 'C@':
            while True:
                di_angle = deal_di_axial_chirality(mol2, patt, index, number)
                if di_angle < 0:
                    break
        if result[0] == 'C@@':
            while True:
                di_angle = deal_di_axial_chirality(mol2, patt, index, number)
                if di_angle > 0:
                    break
    
    else:
        AllChem.EmbedMultipleConfs(mol2, numConfs=1, params=params)
        
if __name__ == '__main__':
    
    # Acquire data
    data = pd.read_excel('Example.xlsx', header=2, index_col=0)
    
    # Add args
    parser = argparse.ArgumentParser(description='ArgparseTry') # 0 denote olefin, 1 denote ligand-M
    parser.add_argument('--type', required=True, type=int)
    args = parser.parse_args()
    cal_type = args.type  

    # Padding Ligand-M
    for j in range(0,len(data['Ligand-M'])):
        ligand = data.loc[j, 'ligand']
        metal = data.loc[j, 'metal']
        comb = gen_smiles(ligand, metal)
        data.loc[j,'Ligand-M'] = comb
    
    # Store the error
    gen_error_list = [] 
    
    # Smart match structure
    patt1 = Chem.MolFromSmarts('c3ccc4ccccc4c3c3c(ccc4ccccc43)') 
    patt2 = Chem.MolFromSmarts('p3oc4ccccc4c4ccccc4o3')
    
    # Column in xlsx corresponding to smiles, which 0 represents olefin and 2 represents Ligand-M
    if cal_type == 0:    
    	index = [0]
    elif cal_type == 1:
    	index = [2]

    # Preprocess for smiles
    for i in index:
        col_name = data.columns[i]
        
        for j in range(0,len(data[col_name])):
            params = AllChem.ETKDGv3()  
            params.useSmallRingTorsions = True
            smiles = data.iloc[j, i]
            can_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) # Sanitize SMILES
            mol = Chem.MolFromSmiles(can_smile)
            mol2 = Chem.AddHs(mol)
            
            # Record the number of failed conformation
            for k in range(5):
                L = AllChem.EmbedMolecule(mol2, params=params)
                if L == 0:
                    break
                if L == -1:
                    gen_error_list.append(f"{j}-{i}-{col_name}")

            # Preprocess the axial chirality 
            if mol.HasSubstructMatch(patt1):
                gen_confor(mol2, patt1)
            elif mol.HasSubstructMatch(patt2):
                gen_confor(mol2, patt2)
            else:
                AllChem.EmbedMultipleConfs(mol2, numConfs=1, params=params)

            # Save as xyz files
            Chem.MolToXYZFile(mol2, f"./xyz/{j}-{i}-{col_name}.xyz")

    print(gen_error_list)
