import pandas as pd
import numpy as np
import argparse

def readxyz(path):
    atom_type = []
    with open(path,'r') as f:
        content = f.read()
        contact = content.split('\n')
        for line in contact[2:]:
            if line == '' or line.isdigit():
                continue
            else:
                atom = line.split()
                atom_type.append(atom[0])

        f.close()   
    return atom_type
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ArgparseTry')
    parser.add_argument('--path',required=True,type=str)
    args = parser.parse_args()
    atom_type=set(readxyz(path=args.path))
    sdd_elements = ['Fe','Co','Ir','Pd','Pt','Rh','Sn','I']
    main_group_elements = ['Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'S', 'Si','P','Na']
    for i in atom_type:
        if i not in sdd_elements:
            print(i, end=' ')
    print(0)
    print('6-31G**')
    for i in atom_type:
        if i in sdd_elements:
            print(f'****\n{i} 0\nSDD\n****\n\n{i} 0\nSDD')
    print('\n\n')