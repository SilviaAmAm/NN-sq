from qml.qmlearn.preprocessing import AtomScaler
from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel
import pickle
from random import shuffle


# Getting the dataset
data_methane = h5py.File("../../data_sets/methane_cn_dft.hdf5", "r")
data_squal = h5py.File("../../data_sets/squalane_cn_dft.hdf5", "r")

ref_ene = -133.1 * 2625.50

n_samples = 15000

xyz_squal = np.array(data_squal.get("xyz"))
ene_squal = np.array(data_squal.get("ene")) * 2625.50
ene_squal = ene_squal - ref_ene
zs_squal = np.array(data_squal.get("zs"), dtype=np.int32)
n_atoms_squal = len(zs_squal[0])
print("The number of squalane samples is: %i" % len(zs_squal))

traj_idx_methane = np.asarray(data_methane.get('traj_idx'), dtype=int)

idx_train = np.where(traj_idx_methane != 14)[0]
shuffle(idx_train)
idx_train = idx_train[:n_samples]
idx_test = np.where(traj_idx_methane == 14)[0]

print("The number of methane samples is: %i (train) %i (trajectory)" % (len(idx_train), len(idx_test)))

xyz_methane = np.concatenate((np.array(data_methane.get("xyz"))[idx_train], np.array(data_methane.get("xyz"))[idx_test]))
zs_methane = np.concatenate((np.array(data_methane.get("zs"), dtype=np.int32)[idx_train], np.array(data_methane.get("zs"), dtype=np.int32)[idx_test]))
ene_methane = (np.concatenate((np.array(data_methane.get("ene"))[idx_train], np.array(data_methane.get("ene"))[idx_test]))* 2625.50) - ref_ene

pad_xyz_methane = np.concatenate((xyz_methane, np.zeros((xyz_methane.shape[0], n_atoms_squal - xyz_methane.shape[1], 3))), axis=1)
pad_zs_methane = np.concatenate((zs_methane, np.zeros((zs_methane.shape[0], n_atoms_squal - zs_methane.shape[1]), dtype=np.int32)), axis=1)

concat_xyz = np.concatenate((pad_xyz_methane, xyz_squal))
concat_ene = np.concatenate((ene_methane, ene_squal))
concat_zs = np.concatenate((pad_zs_methane, zs_squal))

zs_for_scaler = list(zs_methane) + list(zs_squal)

scaling = pickle.load(open("../../scaler/scaler.pickle", "rb"))
concat_ene_scaled = scaling.transform(zs_for_scaler, concat_ene)

# ACSF parameters
n_basis = 19
r_min = 0.8
r_cut = 4.179471213058485
tau = 1.4149304209115
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))

acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

# Generate estimator
estimator = ARMP(iterations=500, l1_reg=9.062776275956941e-06, l2_reg=3.093356834458892e-06, learning_rate=0.0015127649122759221, representation_name='acsf',
                 representation_params=acsf_params, tensorboard=True, store_frequency=10, hidden_layer_sizes=(300,67,), batch_size=73)

estimator.set_properties(concat_ene_scaled)
estimator.generate_representation(concat_xyz, concat_zs, method='fortran')

# Training and testing
idx_n_samples = list(range(n_samples))
shuffle(idx_n_samples)
idx_traj = list(range(n_samples, len(ene_methane)))
idx_squal = list(range(len(ene_methane), len(concat_ene)))

estimator.fit(idx_n_samples)

estimator.save_nn()

# Scoring the model
print("The score on the methane trajectory is:")
score_traj = estimator.score(idx_test)
print(score_traj)

print("The score on the squalane trajectory is:")
score_squal = estimator.score(idx_squal)
print(score_squal)

ene_pred_traj = estimator.predict(idx_test)
ene_pred_squal = estimator.predict(idx_squal)

np.savez("predictions.npz", ene_pred_traj, concat_ene_scaled[idx_test], ene_pred_squal, concat_ene_scaled[idx_squal])
