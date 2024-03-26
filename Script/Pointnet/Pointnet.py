import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPool2D, Reshape, concatenate, ReLU
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import warnings
import time
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from pymatgen.core.periodic_table import Element

# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.des_dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            # add index to feature
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            
            if len(s) == 0:
                self.des_dim += 1
            else:
                # acquire total dimensions
                self.dim += len(s)
            
        
    def encode(self, inputs):
        output_onehot = np.zeros((self.dim,))
        output_des = np.zeros((self.des_dim,))
        for ids, (name_feature, feature_mapping) in enumerate(self.features_mapping.items()):
            feature = getattr(self, name_feature)(inputs)
            if len(feature_mapping) == 0:
                output_des[ids] = feature
            else:
                if feature not in feature_mapping:
                    continue
                output_onehot[feature_mapping[feature]] = 1.0
        
        return np.concatenate([output_onehot, output_des])
        
class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def atomic_num(self, atom):
        return Element(atom.GetSymbol()).Z

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def radius(self, atom):
        return Element(atom.GetSymbol()).atomic_radius

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()  
        
    def chi(self, atom):
        return Element(atom.GetSymbol()).X    
        
    def group(self, atom):
        return Element(atom.GetSymbol()).group 
        
    def molar_volume(self, atom):
        return Element(atom.GetSymbol()).molar_volume
        
    def first_ion_energy(self, atom):
        return Element(atom.GetSymbol()).ionization_energies[0]
    
def molecule_from_molfile(path):
    molecule = Chem.MolFromMolFile(path, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule
    
def point_cloud_from_molecule(molecule):
    atom_features = []
    center = 0 
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == ('Rh' or 'Pt' or 'Co' or 'Pd' or 'Ir'):
            center = np.array(molecule.GetConformer().GetAtomPosition(atom.GetIdx()))
    for atom in molecule.GetAtoms():   
        xyz = np.array(molecule.GetConformer().GetAtomPosition(atom.GetIdx()))
        new_xyz = xyz - center
        atom_features.append(np.concatenate([new_xyz, atom_featurizer.encode(atom)]))
       
    return np.array(atom_features)

def getPointCloud(df, normaliztion=True, substance='Ligand-M'):
    point_cloud_set = []
    len = df.shape[0]
    for i in range(len):
        if substance == 'Ligand-M':
            mol = molecule_from_molfile(f'../mywork/hydroformylation/mol/Ligand-M/{i}-Ligand-M.mol')
            point_cloud_set.append(point_cloud_from_molecule(mol))
        elif substance == 'olefin':
            mol = molecule_from_molfile(f'../mywork/hydroformylation/mol/olefin/{i}-olefin.mol')
            point_cloud_set.append(point_cloud_from_molecule(mol))
    return np.array(point_cloud_set)

def point_cloud_padding(point_cloud):
    num_atoms = [i.shape[0] for i in point_cloud]
    max_num_atoms = np.max(num_atoms)
    pc_stacked = np.stack(
            [
                np.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(point_cloud, num_atoms)
            ],
            axis=0,
        )
    return pc_stacked, max_num_atoms

def scale(data, method):

    sca_data = []

    if method == 'standard':
        scaler = StandardScaler()
        sca_data.append(scaler.fit_transform(data))

    if method == 'minmax':
        scaler = MinMaxScaler()
        sca_data.append(scaler.fit_transform(data))

    return sca_data
    
def prepare_batch(x_batch_1, x_batch_2, x_batch_3, y_batch):
    return (x_batch_1, x_batch_2, x_batch_3), y_batch
    
def PointnetDataset(X1, X2, X3, y, batch_size=64, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X1, X2, X3, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch).prefetch(-1)

def conv_bn_relu(inputs, num_ouput_channels, kernel_size, stride=[1,1], padding='same', activation_fn=ReLU):
    outputs = Conv2D(num_ouput_channels, kernel_size, stride, padding=padding)(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = ReLU()(outputs)
    return outputs
    
def pointnet(input, K, **kwargs):
    num_point = input.get_shape()[1]
    input_image = tf.expand_dims(input, -1)
    output = conv_bn_relu(input_image, 64, (1,K), (1,1), padding='valid')
    output = conv_bn_relu(output, 128, (1,1), (1,1), padding='valid')
    output = conv_bn_relu(output, hidden_dim, (1,1), (1,1), padding='valid')
    output = MaxPool2D(pool_size=(num_point, 1))(output)
    output = Reshape([hidden_dim])(output)
    weight = Dense(K * K, kernel_initializer=tf.keras.initializers.Zeros()
                   , bias_initializer=tf.keras.initializers.constant(np.eye(K).flatten().astype("float32")))(output) #(None, 225)
    transform = Reshape([K,K])(weight) #(None, 15, 15)
    transform_input = tf.matmul(input, transform) #(None, 258, 15)
    input_image = tf.expand_dims(transform_input, -1)
    output = conv_bn_relu(input_image, 64, (1,K), (1,1), padding='valid')
    output = conv_bn_relu(output, 128, (1,1), (1,1), padding='valid')
    output = conv_bn_relu(output, hidden_dim, (1,1), (1,1), padding='valid')
    output = MaxPool2D(pool_size=(num_point, 1))(output)
    output = Reshape([hidden_dim])(output)
    output = Dense(512, activation="relu")(output)
    output = Dropout(rate)(output)
    output = Dense(256, activation="relu")(output)
    output = Dropout(rate)(output)
    output = Dense(embedding_dim, activation="relu")(output)
    return output

def PointnetModel(
    atom_dim,
    max_num_ligand,
    max_num_olefin,
):

    pc_1 = layers.Input((max_num_ligand, atom_dim), dtype="float32", name="atom_features_1")
    pc_2 = layers.Input((max_num_olefin, atom_dim), dtype="float32", name="atom_features_2")
    rc = layers.Input((26), dtype="float32", name="reaction_constant")

    X_pointnet0 = pointnet(pc_1, K=atom_dim)
    X_pointnet1 = pointnet(pc_2, K=atom_dim)
    
    X = concatenate([X_pointnet0, X_pointnet1, rc])
    output = Dense(1024, activation="relu")(X)
    output = Dropout(rate)(output)
    output = Dense(512, activation="relu")(output)
    output = Dropout(rate)(output)
    output = Dense(256, activation="relu")(output)
    output = Dropout(rate)(output)
    output = Dense(1)(output)

    model = keras.Model(
        inputs=[pc_1, pc_2, rc],
        outputs=[output],
    )
    return model

if __name__ == '__main__':
    # Feature engineer
    atom_featurizer = AtomFeaturizer(
        allowable_sets={
            "atomic_num": {},#1, 6, 7, 8, 9, 11, 14, 15, 16, 17, 26, 27, 35, 45, 46, 50, 53, 77, 78
            "n_valence": {},
            "radius": {},
            "chi": {},
            "group": {},
            "molar_volume": {},
            "first_ion_energy": {},
            "hybridization": {"s", "sp", "sp2", "sp3", "sp3d"}
            
        }
    )
    dim = 15

    # Hyperparameter
    rate = 0.2
    random_state = 1
    hidden_dim = 256
    embedding_dim = 32
    EPOCH = 200
    batch_size = 64
    learning_rate = 1E-3
    aug = 10
    model_perform = []
    name = []

    # Raw data
    df = pd.read_csv('descriptor.csv', index_col=0)
    df2 = pd.read_csv('hydroformylation.csv', index_col=0)
    df_new = df
    
    # Get the point cloud
    pc_ligand = getPointCloud(df_new, substance='Ligand-M')
    pc_olefin = getPointCloud(df_new, substance='olefin')
    pc_ligand, max_num_ligand = point_cloud_padding(pc_ligand)
    pc_olefin, max_num_olefin = point_cloud_padding(pc_olefin)
    pc_ligand = scale(pc_ligand.astype("float32").reshape([-1,dim]), 'standard')[0].reshape([len(df_new),-1,dim])
    pc_olefin = scale(pc_olefin.astype("float32").reshape([-1,dim]), 'standard')[0].reshape([len(df_new),-1,dim])
    rc = scale(np.array(df_new.iloc[:, :26]).astype("float32"), 'standard')[0]
    y = np.array(df_new['L/B']).astype("float32").reshape(-1,1)

    # Record the time
    start = time.perf_counter()
    count = 0
    model_mae = []
    model_r2 = []
    name = []

    count += 1
    print(f'执行第{count}轮参数调试')

    np.random.seed(random_state)
    permuted_indices = np.random.permutation(np.arange(df_new.shape[0]))
    # Train set: 80 % of data
    train_index = permuted_indices[: int(df_new.shape[0] * 0.6)]
    x_train_1 = pc_ligand[train_index]
    x_train_2 = pc_olefin[train_index]
    x_train_3 = rc[train_index]
    y_train = y[train_index]

    # Valid set: 10 % of data
    valid_index = permuted_indices[int(df_new.shape[0] * 0.6) : int(df_new.shape[0] * 0.8)]
    x_valid_1 = pc_ligand[valid_index]
    x_valid_2 = pc_olefin[valid_index]
    x_valid_3 = rc[valid_index]
    y_valid = y[valid_index]

    # Test set: 10 % of data
    test_index = permuted_indices[int(df_new.shape[0] * 0.8) :]
    x_test_1 = pc_ligand[test_index]
    x_test_2 = pc_olefin[test_index]
    x_test_3 = rc[test_index]
    y_test = y[test_index]

    train_dataset = PointnetDataset(x_train_1, x_train_2, x_train_3, y_train, batch_size=batch_size)
    valid_dataset = PointnetDataset(x_valid_1, x_valid_2, x_valid_3, y_valid, batch_size=batch_size)
    test_dataset = PointnetDataset(x_test_1, x_test_2, x_test_3, y_test, batch_size=batch_size)

    # Model construct
    model = PointnetModel(atom_dim=dim, max_num_ligand=max_num_ligand, max_num_olefin=max_num_olefin)

    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[keras.metrics.MeanAbsoluteError(name="MAE")],
    )

    # Record the time
    start = time.perf_counter()

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=EPOCH,
        verbose=2,
        callbacks=[
        tf.keras.callbacks.ModelCheckpoint("./1-octene/model_weights_{epoch}.h5", save_best_only=False, save_weights_only=True, verbose=2, monitor="val_loss"),
        ]
    )

    plt.figure(figsize=(10, 6))
    plt.plot(history.history["MAE"], label="train MAE")
    plt.plot(history.history["val_MAE"], label="valid MAE")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("MAE", fontsize=16)
    plt.legend(fontsize=16)
        
                
    end = time.perf_counter()
    print(f"model train time consumed: {np.round(end - start, decimals=3)} sec.")