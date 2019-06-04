from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py
import pickle
from random import shuffle


# Getting the dataset
data_methane = h5py.File("../../data_sets/methane_cn_dft.hdf5", "r")
data_isopentane = h5py.File("../../data_sets/isopentane_cn_dft.hdf5", "r")
data_squal = h5py.File("../../data_sets/squalane_cn_dft.hdf5", "r")

ref_ene = -133.1 * 2625.50

n_samples_methane = 7500
n_samples_isopentane = 7500


zs_squal = np.array(data_squal.get("zs"), dtype=np.int32)
n_atoms_squal = len(zs_squal[0])
print("The number of squalane samples is: %i" % len(zs_squal))

# Data for methane
traj_idx_methane = np.asarray(data_methane.get('traj_idx'), dtype=int)

idx_train_methane = np.where(traj_idx_methane != 14)[0]
shuffle(idx_train_methane)
idx_train_methane = idx_train_methane[:n_samples_methane]

print("The number of methane samples is: %i (train)" % (len(idx_train_methane)))

xyz_methane = np.array(data_methane.get("xyz"))[idx_train_methane]
zs_methane = np.array(data_methane.get("zs"), dtype=np.int32)[idx_train_methane]
ene_methane = np.array(data_methane.get("ene"))[idx_train_methane]* 2625.50 - ref_ene

pad_xyz_methane = np.concatenate((xyz_methane, np.zeros((xyz_methane.shape[0], n_atoms_squal - xyz_methane.shape[1], 3))), axis=1)
pad_zs_methane = np.concatenate((zs_methane, np.zeros((zs_methane.shape[0], n_atoms_squal - zs_methane.shape[1]), dtype=np.int32)), axis=1)

# Data for isopentane
traj_idx_isopentane = np.asarray(data_isopentane.get('traj_idx'), dtype=int)

idx_train_isopentane = np.where(traj_idx_isopentane != 22)[0]
shuffle(idx_train_isopentane)
idx_train_isopentane = idx_train_isopentane[:n_samples_isopentane]

print("The number of isopentane samples is: %i (train)" % (len(idx_train_isopentane)))

xyz_isopentane = np.array(data_isopentane.get("xyz"))[idx_train_isopentane]
zs_isopentane = np.array(data_isopentane.get("zs"), dtype=np.int32)[idx_train_isopentane]
ene_isopentane = np.array(data_isopentane.get("ene"))[idx_train_isopentane]* 2625.50 - ref_ene

pad_xyz_isopentane = np.concatenate((xyz_isopentane, np.zeros((xyz_isopentane.shape[0], n_atoms_squal - xyz_isopentane.shape[1], 3))), axis=1)
pad_zs_isopentane = np.concatenate((zs_isopentane, np.zeros((zs_isopentane.shape[0], n_atoms_squal - zs_isopentane.shape[1]), dtype=np.int32)), axis=1)

# Concatenating all the data
concat_xyz = np.concatenate((pad_xyz_methane, pad_xyz_isopentane))
concat_ene = np.concatenate((ene_methane, ene_isopentane))
concat_zs = np.concatenate((pad_zs_methane, pad_zs_isopentane))

zs_for_scaler = list(zs_methane) + list(zs_isopentane)

scaling = pickle.load(open("../../scaler/scaler.pickle", "rb"))
concat_ene_scaled = scaling.transform(zs_for_scaler, concat_ene)

# ACSF parameters
n_basis = 16
r_min = 0.8
r_cut = 3.24756528424191
tau = 1.3399321961520696 
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))

acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

# Generate estimator
estimator = ARMP(iterations=1130, l1_reg=9.6983542971344e-06, l2_reg=2.7847233160230936e-06, learning_rate=0.0009050324762982482, representation_name='acsf',
                 representation_params=acsf_params, tensorboard=True, store_frequency=10, hidden_layer_sizes=(371,94,), batch_size=73)

estimator.set_properties(concat_ene_scaled)
estimator.generate_representation(concat_xyz, concat_zs, method='fortran')

# Training and testing
idx_training = list(range(len(concat_ene_scaled)))
shuffle(idx_training)

estimator.fit(idx_training)

estimator.save_nn()
