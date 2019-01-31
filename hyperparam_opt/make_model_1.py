import pickle
import numpy as np
import h5py
from qml.aglaia.aglaia import ARMP
from random import shuffle


ref_ene = -133.1 * 2625.5

data_methane = h5py.File("/home/sa16246/repositories/NN-sq/data_sets/methane_cn_dft.hdf5", "r")
data_squal = h5py.File("/home/sa16246/repositories/NN-sq/data_sets/squalane_cn_dft_pruned.hdf5", "r")

# Getting the data
ene_methane = np.array(data_methane.get("ene")) * 2625.50
ene_methane = ene_methane - ref_ene
zs_methane = np.array(data_methane.get("zs"), dtype=np.int32)
xyz_methane = np.array(data_methane.get("xyz"), dtype=np.int32)

ene_squal = np.array(data_squal.get("ene")) * 2625.50
ene_squal = ene_squal - ref_ene
zs_squal = np.array(data_squal.get("zs"), dtype=np.int32)
xyz_squal = np.array(data_squal.get("xyz"), dtype=np.int32)
n_atoms_squal = len(zs_squal[0])

# Scaling the data
concat_ene = np.concatenate((ene_methane, ene_squal))
zs_for_scaler = list(zs_methane) + list(zs_squal)
scaling = pickle.load(open("./scaler.pickle", "rb"))
concat_ene_scaled = scaling.transform(zs_for_scaler, concat_ene)

# Padding the inputs
pad_xyz_methane = np.concatenate((xyz_methane, np.zeros((xyz_methane.shape[0], n_atoms_squal - xyz_methane.shape[1], 3))), axis=1)
pad_zs_methane = np.concatenate((zs_methane, np.zeros((zs_methane.shape[0], n_atoms_squal - zs_methane.shape[1]), dtype=np.int32)), axis=1)

# Taking 15000 random methane samples
all_idx = range(len(ene_methane))
shuffle(all_idx)
train_idx = all_idx[:15000]

# Concatenating the intputs
concat_xyz = np.concatenate((pad_xyz_methane[train_idx], xyz_squal))
concat_zs = np.concatenate((pad_zs_methane[train_idx], zs_squal))

# Create the estimator
acsf_params = {"nRs2":14, "nRs3":14, "nTs":14, "rcut":3.29, "acut":3.29, "zeta":100.06564927139748, "eta":39.81824764370754}
estimator = ARMP(representation_name='acsf', representation_params=acsf_params, hidden_layer_sizes=(150,))

estimator.set_properties(concat_ene_scaled)
estimator.generate_representation(concat_xyz, concat_zs, method="fortran")

pickle.dump(estimator, open('model.pickle', 'wb'))

with open('idx.csv', 'w') as f:
    for i in range(len(concat_ene)):
        f.write('%s\n' % i)
