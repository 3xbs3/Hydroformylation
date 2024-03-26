from __future__ import print_function
import argparse

def readxyz(path):
    atom_type = []
    with open(path,'r') as f:
        content=f.read()
        contact = content.split('\n')
        for line in contact[2:]:
            if line == '' or line.isdigit():
                continue
            else:
                atom = line.split()
                atom_type.append(atom[0])

        f.close()
    return atom_type

parser = argparse.ArgumentParser(description='ArgparseTry')
parser.add_argument('--path',required=True,type=str)
args = parser.parse_args()
# atom_type=set(readxyz(path='./ml_gaussian/49-Ligand-M.xyz'))
atom_type=set(readxyz(path=args.path))
# atom_type=set(readxyz(path='./ml_gaussian/0-olefin.xyz'))
sdd_elements = ['Fe','Co','Ir','Pd','Pt','Rh','Sn','I']
main_group_elements = ['Br', 'C', 'Cl', 'F', 'H', 'N', 'O', 'S', 'Si','P','Na']
print('\n', end='')
for i in atom_type:
    if i not in sdd_elements:
        print(i, end=' ')
print(0)
print('6-31G**')
print('****')
for i in atom_type:
    if i in sdd_elements:
        print(i, end=' ') 
print(0)
print('SDD')
print('****\n')  
for i in atom_type:
    if i in sdd_elements:
        print(i, end=' ') 
print(0)
print('SDD')
print('\n')

