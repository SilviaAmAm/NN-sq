from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py
import pickle
from random import shuffle


# Getting the dataset
data_methane = h5py.File("../../data_sets/methane_cn_dft.hdf5", "r")
data_squal = h5py.File("../../data_sets/squalane_cn_dft.hdf5", "r")

ref_ene = -133.1 * 2625.50

n_samples = 15000

zs_squal = np.array(data_squal.get("zs"), dtype=np.int32)
n_atoms_squal = len(zs_squal[0])
print("The number of squalane samples is: %i" % len(zs_squal))

traj_idx_methane = np.asarray(data_methane.get('traj_idx'), dtype=int)

idx_train = np.where(traj_idx_methane != 14)[0]
shuffle(idx_train)
idx_train = idx_train[:n_samples]

print("The number of methane samples is: %i (train) " % (len(idx_train)))

xyz_methane = np.array(data_methane.get("xyz"))[idx_train]
zs_methane = np.array(data_methane.get("zs"), dtype=np.int32)[idx_train]
ene_methane = np.array(data_methane.get("ene"))[idx_train]* 2625.50 - ref_ene

pad_xyz_methane = np.concatenate((xyz_methane, np.zeros((xyz_methane.shape[0], n_atoms_squal - xyz_methane.shape[1], 3))), axis=1)
pad_zs_methane = np.concatenate((zs_methane, np.zeros((zs_methane.shape[0], n_atoms_squal - zs_methane.shape[1]), dtype=np.int32)), axis=1)

scaling = pickle.load(open("../../scaler/scaler.pickle", "rb"))
ene_scaled = scaling.transform(zs_methane, ene_methane)

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

estimator.set_properties(ene_scaled)
estimator.generate_representation(pad_xyz_methane, pad_zs_methane, method='fortran')

# Training and testing
idx_n_samples = list(range(n_samples))
shuffle(idx_n_samples)

estimator.fit(idx_n_samples)

estimator.save_nn()