from ase.io import read
from dscribe.descriptors.mbtr import MBTR
from dscribe.descriptors.soap import SOAP
from dscribe.descriptors.lmbtr import LMBTR
from dscribe.descriptors.acsf import ACSF
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def recognize_center(mol):
    atom_num = mol.get_atomic_numbers()
    for i in [27,45,46,77,78]:
        if i in atom_num:
            return list(atom_num).index(i)
            
def get_descr(type, df):

    '''
    Arg:
        type:different descriptor(include mbtr,lmbtr,ascf,soap)
        df:dataframe
    Return:
        1D array descriptor of the database
    '''

    # get the row number
    row_num = df.shape[0]
    # chemical substance index
    rea_descr = []
    chem_descr = []

    for i in range(0, row_num):
#             print(i)
            mol = read(f'./ood_olefin/{i}-olefin.pdb')

            if type == 'mbtr':
                k1 = {"geometry": {"function": "atomic_number"},
                      "grid": {"min": 0, "max": 8, "n": 10, "sigma": 0.1}}
                k2 = {"geometry": {"function": "inverse_distance"},
                      "grid": {"min": 0, "max": 4, "n": 10, "sigma": 0.1},
                      "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3}}
                k3 = {"geometry": {"function": "cosine"},
                      "grid": {"min": -1, "max": 4, "n": 10, "sigma": 0.1},
                      "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3}}
                setup = MBTR(geometry={"function": "inverse_distance"}, grid={"min": 0, "max": 4, "n": 10, "sigma": 0.1}
                             ,weighting= {"function": "exp", "scale": 0.5, "threshold": 1e-3}
                    ,species=['Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'S', 'Si']
                             ,normalization='l2', periodic=False, sparse=False)
                descr = setup.create(system=mol)
                chem_descr.append(descr)

            if type == 'lmbtr':
                k2 = {"geometry": {"function": "inverse_distance"},
                      "grid": {"min":0, "max":1, "n":10,"sigma":0.1},
                      "weighting": {"function":"exp", "scale": 0.5, "threshold": 1e-3}}
                k3 = {"geometry": {"function": "cosine"},
                      "grid": {"min": -1, "max": 1, "n": 10, "sigma":0.1},
                      "weighting": {"function": "exp", "scale": 0.5,"threshold": 1e-3}}
                setup = LMBTR(geometry={"function": "inverse_distance"}, grid={"min": 0, "max": 1, "n": 10, "sigma": 0.1}
                             ,weighting= {"function": "exp", "scale": 0.5, "threshold": 1e-3}
                    ,species=['Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'S', 'Si']
                              ,normalization='l2', periodic=False, sparse=False)
                descr = setup.create(system=mol, centers=[0])
                chem_descr.append(descr)

            if type == 'soap':
                rcut = 6.0
                nmax = 4
                lmax = 3
                setup = SOAP(r_cut=rcut, n_max=nmax, l_max=lmax, species=['Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'S', 'Si'],
                             periodic=False, sparse=False)
#                 descr = np.average(setup.create(system=mol), axis=0)
                descr = setup.create(system=mol, centers=[0])
                chem_descr.append(descr)

            if type == 'acsf':
                rcut = 6.0,
                g2_params = [[1, 1], [1, 2], [1, 3]]
                g4_params = [[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]]
                setup = ACSF(6.0, species=['Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'S', 'Si'],
                             g2_params=g2_params, g4_params=g4_params, periodic=False, sparse=False)
                descr = setup.create(system=mol, centers=[0])
                chem_descr.append(descr)

    return chem_descr
    
def scale(data, method):

    '''
    Arg:
        data:data to be processed
        method: scaler's method(include standard, minmax)
    Return:
        processed data
    '''

    sca_data = []

    if method == 'standard':
        scaler = StandardScaler()
        sca_data.append(scaler.fit_transform(data))

    if method == 'minmax':
        scaler = MinMaxScaler()
        sca_data.append(scaler.fit_transform(data))

    return sca_data
    
if __name__ == '__main__':
    df = pd.read_excel('./Ligand-M.xlsx', index_col=0,header=2)  # index_col去行号
    types = [ 'acsf','soap','lmbtr','mbtr']
    # preprocess the input and output
    for type in types:
        descr = get_descr(type=type,df=df)
        descr = np.array(descr).reshape(1167,-1)
        dataset = pd.DataFrame(descr)
        dataset.to_csv(f"{type}-olefin.csv")
        print(f'{type}-olefin.csv is generate')