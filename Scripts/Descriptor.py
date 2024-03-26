from rdkit import Chem
from morfeus import BiteAngle, read_xyz, BuriedVolume, ConeAngle
import numpy as np
import pandas as pd
import os

def get_bv_and_ba(num):
    angle_set = []
    bv_set = []
    metal_set = ['Rh', 'Ir', 'Co', 'Pd', 'Pt']
    for i in range(num):
        # Acquire bite angle
        elements, coordinates = read_xyz(f'{i}-Ligand-M.xyz')
        p1_index = np.where(elements == 'P')[0][0] + 1
        p2_index = np.where(elements == 'P')[0][1] + 1
        for i in metal_set:
            if i in elements:
                rh_index = np.where(elements == i)[0][0] + 1
        ba = BiteAngle(coordinates, rh_index, p1_index, p2_index)
        angle_set.append(ba.angle)
        
        # Acquire buried volume
        mol = Chem.MolFromMolFile(f'{i}-Ligand-M.mol')
        if 'Rh' in elements:
            patt = Chem.MolFromSmarts('[Rh](CO)')
            m_index = np.where(elements == 'Rh')[0][0] + 1
        elif 'Co' in elements:
            patt = Chem.MolFromSmarts('[Co](CO)(CO)')
            m_index = np.where(elements == 'Co')[0][0] + 1
        elif 'Pt' in elements:
            patt = Chem.MolFromSmarts('[Pt][Sn](Cl)(Cl)Cl')
            m_index = np.where(elements == 'Pt')[0][0] + 1
        elif 'Pd' in elements:
            patt = Chem.MolFromSmarts('[Pd]I')
            m_index = np.where(elements == 'Pd')[0][0] + 1
        elif 'Ir' in elements:
            patt = Chem.MolFromSmarts('[Ir](CO)')
            m_index = np.where(elements == 'Ir')[0][0] + 1
        hit_ats = [i + 1 for i in list(mol.GetSubstructMatch(patt))]
        bv = BuriedVolume(elements, coordinates, m_index, excluded_atoms=hit_ats)
        bv_set.append(bv.fraction_buried_volume)
        
        df = pd.DataFrame({'bite angle': angle_set, 'buried volume': bv_set})
    return df
    
if __name__ == '__main__':

    df = get_bv_and_ba(1167)
    df.to_csv('ba-bv.csv')
    print('done!')