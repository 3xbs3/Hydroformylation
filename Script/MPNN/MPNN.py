import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time
from rdkit import Chem
from rdkit import RDLogger
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pymatgen.core.composition import Element

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

   
class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()
        

def molecule_from_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule
    
def graph_from_molecule(molecule):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))

        # Add self-loops
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)

def graphs_from_smiles(smiles_list):
    # Initialize graphs
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []

    for smiles in smiles_list:
        molecule = molecule_from_smiles(smiles)
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

        atom_features_list.append(atom_features)
        bond_features_list.append(bond_features)
        pair_indices_list.append(pair_indices)

    # Convert lists to ragged tensors for tf.data.Dataset later on
    return (
        tf.ragged.constant(atom_features_list, dtype=tf.float32),
        tf.ragged.constant(bond_features_list, dtype=tf.float32),
        tf.ragged.constant(pair_indices_list, dtype=tf.int64),
    )

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

def mergeIntoGlobal(x_batch):
    atom_features, bond_features, pair_indices = x_batch
    
    # Obtain number of atoms and bonds for each graph (molecule)
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    # Obtain partition indices (molecule_indicator), which will be used to gather (sub)graphs from global graph in model later on
    molecule_indices = tf.range(len(num_atoms))
    molecule_indicator = tf.repeat(molecule_indices, num_atoms) # 索引对应smiles,相同数字代表有几个原子

    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to 'pair_indices' (and merging ragged tensors) actualizes the global graph
    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    
    return atom_features, bond_features, pair_indices, molecule_indicator
    
def prepare_batch(x_batch_1, x_batch_2, x_batch_3, x_batch_4, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph
    """

    atom_features_1, bond_features_1, pair_indices_1, molecule_indicator_1 = mergeIntoGlobal(x_batch_1)
    atom_features_2, bond_features_2, pair_indices_2, molecule_indicator_2 = mergeIntoGlobal(x_batch_2)
    atom_features_3, bond_features_3, pair_indices_3, molecule_indicator_3 = mergeIntoGlobal(x_batch_3)

    return (atom_features_1, bond_features_1, pair_indices_1, molecule_indicator_1,
           atom_features_2, bond_features_2, pair_indices_2, molecule_indicator_2,
           atom_features_3, bond_features_3, pair_indices_3, molecule_indicator_3, x_batch_4), y_batch

def MPNNDataset(X1, X2, X3, X4, y, batch_size=64, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X1, X2, X3, X4, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)

class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim), initializer="zeros", name="bias",
        )
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Apply linear transformation to bond features
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias

        # Reshape for neighborhood aggregation later
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))

        # Obtain atom features of neighbors
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)

        # Apply neighborhood aggregation
        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices[:, 0],
            num_segments=tf.shape(atom_features)[0],
        )
        return aggregated_features

class MessagePassing(layers.Layer):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Pad atom features if number of desired units exceeds atom_features dim.
        # Alternatively, a dense layer could be used here.
        atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # Aggregate information from neighbors
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )

            # Update node state via a step of GRU
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated

class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):

        atom_features, molecule_indicator = inputs

        # Obtain subgraphs
        atom_features_partitioned = tf.dynamic_partition(
            atom_features, molecule_indicator, self.batch_size
        )

        # Pad and stack subgraphs
        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0,
        )

        # Remove empty subgraphs (usually for last batch in dataset)
        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)

class TransformerEncoderReadout(layers.Layer):
    def __init__(
        self, num_heads=8, embed_dim=64, dense_dim=256, batch_size=64, **kwargs
    ):
        super().__init__(**kwargs)

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D() 
    def call(self, inputs):
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        return self.average_pooling(proj_output)

def MPNNModel(
    atom_dim,
    bond_dim,
    batch_size=64,
    message_units=64, #64
    message_steps=4,
    num_attention_heads=8,
    dense_units=512, #512
):

    atom_features_1 = layers.Input((atom_dim), dtype="float32", name="atom_features_1")
    bond_features_1 = layers.Input((bond_dim), dtype="float32", name="bond_features_1")
    pair_indices_1 = layers.Input((2), dtype="int32", name="pair_indices_1")
    molecule_indicator_1 = layers.Input((), dtype="int32", name="molecule_indicator_1")
    atom_features_2 = layers.Input((atom_dim), dtype="float32", name="atom_features_2")
    bond_features_2 = layers.Input((bond_dim), dtype="float32", name="bond_features_2")
    pair_indices_2 = layers.Input((2), dtype="int32", name="pair_indices_2")
    molecule_indicator_2 = layers.Input((), dtype="int32", name="molecule_indicator_2")
    atom_features_3 = layers.Input((atom_dim), dtype="float32", name="atom_features_3")
    bond_features_3 = layers.Input((bond_dim), dtype="float32", name="bond_features_3")
    pair_indices_3 = layers.Input((2), dtype="int32", name="pair_indices_3")
    molecule_indicator_3 = layers.Input((), dtype="int32", name="molecule_indicator_3")
    reaction_constant = layers.Input((26), dtype="float32", name="reaction_constant")

    x1 = MessagePassing(message_units, message_steps)(
        [atom_features_1, bond_features_1, pair_indices_1]
    )
    x1 = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x1, molecule_indicator_1])
    x2 = MessagePassing(message_units, message_steps)(
        [atom_features_2, bond_features_2, pair_indices_2]
    )
    x2 = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x2, molecule_indicator_2])
    x3 = MessagePassing(message_units, message_steps)(
        [atom_features_3, bond_features_3, pair_indices_3]
    )
    x3 = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x3, molecule_indicator_3])
    
    x_raw = layers.Concatenate()([x1, x2, x3, reaction_constant]) # 3*32/64+26
    x = x_raw # 218
    # Attention layer
    x_ = layers.Dense(3*message_units+26, activation='softmax')(x)#+9
    x = layers.Multiply()([x, x_])
    x = layers.Dense(1, activation="sigmoid")(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(1)(x)

    model = keras.Model(
        inputs=[atom_features_1, bond_features_1, pair_indices_1, molecule_indicator_1,
               atom_features_2, bond_features_2, pair_indices_2, molecule_indicator_2,
               atom_features_3, bond_features_3, pair_indices_3, molecule_indicator_3, reaction_constant],
        outputs=[x],
    )
    return model


if __name__ == '__main__':
    # Feature engineer
    atom_featurizer = AtomFeaturizer(
        allowable_sets={
            "atomic_num": {}, #1, 6, 7, 8, 9, 11, 14, 15, 16, 17, 26, 27, 35, 45, 46, 50, 53, 77, 78
            "n_valence": {},
            "chi": {},
            "group": {},
            "molar_volume": {},
            "first_ion_energy": {},
            "hybridization": {"s", "sp", "sp2", "sp3", "sp3d"}
            
        }
    )
    
    bond_featurizer = BondFeaturizer(
        allowable_sets={
            "bond_type": {"single", "double", "triple", "aromatic"},
            "conjugated": {True, False},
        }
    )
    # Raw data
    df = pd.read_csv('hydroformylation.csv', index_col=0)
    df2 = pd.read_csv('descriptor-enlarge.csv', index_col=0)
    
    # Hyperparameter
    rate = 0.2
    split_radio = 0.6
    random_state = 1
    batch_size = 64
    learning_rate = 1E-3
    root_dir = 'C:/Users/wh/Desktop/graph_pred'
    
    # Record the cross-validation results
    y_pred_sum = []
    y_true_sum = []
    
    # Random index
    np.random.seed(random_state)
    permuted_indices = np.random.permutation(np.arange(df.shape[0]))
    # Train set: 70 % of data
    train_index = permuted_indices[: int(df.shape[0] * split_radio)]
    valid_index = permuted_indices[int(df.shape[0] * split_radio) : int(df.shape[0] * 0.8)]
    # Test set: 30 % of data
    test_index = permuted_indices[int(df.shape[0] * 0.8) :]
    
    # Log and ckpt's path
    model_ckpt = os.path.join(root_dir, f"best_model/{batch_size}-model_opt")
    rundir = os.path.join(root_dir, f"log/{batch_size}-log")

    # Train set: 80 % of data
    x_train_1 = graphs_from_smiles(df.iloc[train_index]['olefin'])
    x_train_2 = graphs_from_smiles(df.iloc[train_index]['Ligand-M'])
    x_train_3 = graphs_from_smiles(df.iloc[train_index]['solvent'])
    x_train_4 = scale(np.array(df2.iloc[train_index, :26]), 'standard')[0]  # scale the input
    y_train = df.iloc[train_index]['L/B']
    
    # Valid set: 10 % of data
    x_valid_1 = graphs_from_smiles(df.iloc[valid_index]['olefin'])
    x_valid_2 = graphs_from_smiles(df.iloc[valid_index]['Ligand-M'])
    x_valid_3 = graphs_from_smiles(df.iloc[valid_index]['solvent'])
    x_valid_4 = scale(np.array(df2.iloc[valid_index, :26]), 'standard')[0]
    y_valid = df.iloc[valid_index]['L/B']
    
    # Test set: 20 % of data
    x_test_1 = graphs_from_smiles(df.iloc[test_index]['olefin'])
    x_test_2 = graphs_from_smiles(df.iloc[test_index]['Ligand-M'])
    x_test_3 = graphs_from_smiles(df.iloc[test_index]['solvent'])
    x_test_4 = scale(np.array(df2.iloc[test_index, :26]), 'standard')[0]
    y_test = df.iloc[test_index]['L/B']
    
    # Model construct
    mpnn = MPNNModel(
        atom_dim=x_train_1[0][0][0].shape[0], bond_dim=x_train_1[1][0][0].shape[0], batch_size=batch_size
    )
    
    mpnn.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[keras.metrics.MeanAbsoluteError(name="MAE")],
    )
    train_dataset = MPNNDataset(x_train_1, x_train_2, x_train_3, x_train_4, y_train, batch_size=batch_size)
    valid_dataset = MPNNDataset(x_valid_1, x_valid_2, x_valid_3, x_valid_4, y_valid, batch_size=batch_size)
    test_dataset = MPNNDataset(x_test_1, x_test_2, x_test_3, x_test_4, y_test, batch_size=batch_size)
    
    # Record the time
    start = time.perf_counter()
    
    history = mpnn.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=300,
        verbose=2,
        callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=rundir, histogram_freq=5),
        tf.keras.callbacks.ModelCheckpoint("./stryene/model_weights_{epoch}.h5", save_best_only=False, save_weights_only=True, verbose=2, monitor="val_loss"),
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