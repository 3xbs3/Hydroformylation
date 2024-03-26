import glob
import time
import os
import shutil
import pandas as pd
import numpy as np

class Atom():
    def __init__(self, t: str, r):
        '''
        每一个分子所具有的属性
        r: 原子坐标
        t: 原子类型
        '''
        self.r = r
        self.t = t

def getMolecule(path:str):
    '''
    从pdb文件获取原子类型和坐标
    path: pdb路径
    return: list(原子实例)
    '''
    mol = []
    with open(path, 'r') as f:
        for lines in f.readlines():
            items = lines.split()
            if len(items) == 11:
                mol.append(Atom(items[10], np.asarray([float(i) for i in items[5:8]])))
    return mol

def euclideanDistance(r1, r2):
    '''
    计算两点之间的欧式距离
    r1/r2: 两原子坐标
    '''
    return np.sqrt((r1[0] - r2[0]) ** 2 + (r1[1] - r2[1]) ** 2 + (r1[2] - r2[2]) ** 2)

def getIndexH(mol):
    '''
    判断离末端C最近的H的索引，来构造平面
    mol: 末端烯烃分子
    '''
    rC = mol[0].r #末端C的坐标
    min_dist = 1000
    min_dist_index = 0
    for i in range(1,len(mol)):
        r_ = mol[i].r
        ed = euclideanDistance(r_, rC)
        if ed < min_dist:
            min_dist = ed
            min_dist_index = i 
    return min_dist_index

def getPlaneFunction(r1, r2, r3):
    '''
    三点确定平面系数
    r1/r2/r3: 三个点的坐标
    return: 平面方程的四个系数
    '''
    a = (r2[1] - r1[1]) * (r3[2] - r1[2]) - (r2[2] - r1[2]) * (r3[1] - r1[1])
    b = (r3[0] - r1[0]) * (r2[2] - r1[2]) - (r2[0] - r1[0]) * (r3[2] - r1[2])
    c = (r2[0] - r1[0]) * (r3[1] - r1[1]) - (r3[0] - r1[0]) * (r2[1] - r1[1])
    d = - (a * r1[0] + b * r1[1] + c * r1[2])
    return a, b, c, d

def dummyAtom(mol, dist=2.25):
    '''
    确定虚拟原子的坐标
    mol: 末端烯烃分子
    dist: 金属到双键距离
    return: 虚拟原子坐标、双键垂直平分点坐标
    '''
    r1, r2 = mol[0].r, mol[1].r
    index = getIndexH(mol)
    r3 = mol[index].r
    r_half = (r1 + r2) / 2
    a, b, c, d = getPlaneFunction(r1, r2, r3)
    k = (dist * np.sqrt(a ** 2 + b ** 2 + c ** 2) - (a * r_half[0] + b * r_half[1] + c * r_half[2] + d)) \
        / (a ** 2 + b ** 2 + c ** 2)
    x0, y0, z0 = r_half[0] - a * k, r_half[1] - b * k, r_half[2] - c * k
    return np.asarray([x0, y0, z0]), r_half

def getMaxConeAngle(mol, scale=1, std='radian'):
    '''
    获得最大包络圆锥角度
    mol: 末端烯烃分子
    scale: 半径伸缩尺度
    std: 角度标准——支持弧度制'radian'和角度值'degree'
    return: 角度
    '''
    radius_atom_map = {'Br':1.90, 'C':1.70, 'Cl':1.80, 'F':1.50, 'H':1.20, 'N':1.60, 'O':1.55, 'S':1.80, 'Si':2.10}
    rM, r_half = dummyAtom(mol)
    vec1 = r_half - rM
    mod_vec1 = euclideanDistance(r_half, rM)
    max_theta = 0
    for i in range(len(mol)):
        vec2 = mol[i].r - rM
        mod_vec2 = euclideanDistance(mol[i].r, rM)
        theta1 = np.arccos((vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]) / (mod_vec1 * mod_vec2))
        theta2 = np.arctan(radius_atom_map[mol[i].t] * scale / mod_vec2)
        theta = theta1 + theta2
        if theta > max_theta:
            max_theta = theta
    return max_theta if std == 'radian' else max_theta * 180 / np.pi
    
if __name__ == '__main__':

    angle_set = []
    for i in range(1167):
        mol = getMolecule(f'{i}-olefin.pdb')
        angle = getMaxConeAngle(mol, scale=1.17, std='degree')
        angle_set.append(angle)
    df = pd.DataFrame(angle_set)
    df.to_csv('coneAngle.csv')
    print('done!')