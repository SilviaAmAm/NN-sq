import pickle
import numpy as np
import sklearn.pipeline
from qml import qmlearn
import h5py
from random import shuffle

# Create data
dataset_meth = h5py.File("../../data_sets/methane_cn_dft.hdf5")
dataset_ethane = h5py.File("../../data_sets/ethane_cn_dft.hdf5")
dataset_isobutane = h5py.File("../../data_sets/isobutane_cn_dft.hdf5")
dataset_isopentane = h5py.File("../../data_sets/isopentane_cn_dft.hdf5")


ref_energy = -133.1 * 2625.5
scaling = pickle.load(open("../../scaler/larger_scaler.pickle", "rb"))

# Data objects
data = qmlearn.Data()

# Keeping a trajectory for testing methane
traj_idx_meth = np.asarray(dataset_meth['traj_idx'], dtype=int)
idx_train_meth = np.where(traj_idx_meth != 14)[0]
n_samples_meth = 7500
shuffle(idx_train_meth)
idx_train_meth = idx_train_meth[:n_samples_meth]
xyz_meth = np.asarray(dataset_meth['xyz'])[idx_train_meth]
zs_meth = np.array(dataset_meth['zs'])[idx_train_meth]
energies_meth = np.asarray(dataset_meth['ene'])[idx_train_meth] * 2625.50
energies_meth -= ref_energy
ene_scaled_meth = scaling.transform(zs_meth, energies_meth)
traj_idx_meth = traj_idx_meth[idx_train_meth]

# Keeping a trajectory for testing ethane
traj_idx_ethane = np.asarray(dataset_ethane['traj_idx'], dtype=int)
idx_train_ethane = np.where(traj_idx_ethane != 1)[0]
n_samples_ethane = 3500
shuffle(idx_train_ethane)
idx_train_ethane = idx_train_ethane[:n_samples_ethane]
xyz_ethane = np.asarray(dataset_ethane['xyz'])[idx_train_ethane]
zs_ethane = np.array(dataset_ethane['zs'])[idx_train_ethane]
energies_ethane = np.asarray(dataset_ethane['ene'])[idx_train_ethane] * 2625.50
energies_ethane -= ref_energy
ene_scaled_ethane = scaling.transform(zs_ethane, energies_ethane)
traj_idx_ethane = traj_idx_ethane[idx_train_ethane]

# Keeping a trajectory for testing Isobutane
traj_idx_isobutane = np.asarray(dataset_isobutane['traj_idx'], dtype=int)
idx_train_isobutane = np.where(traj_idx_isobutane != 2)[0]
n_samples_isobutane = 2500
shuffle(idx_train_isobutane)
idx_train_isobutane = idx_train_isobutane[:n_samples_isobutane]
xyz_isobutane = np.asarray(dataset_isobutane['xyz'])[idx_train_isobutane]
zs_isobutane = np.array(dataset_isobutane['zs'])[idx_train_isobutane]
energies_isobutane = np.asarray(dataset_isobutane['ene'])[idx_train_isobutane] * 2625.50
energies_isobutane -= ref_energy
ene_scaled_isobutane = scaling.transform(zs_isobutane, energies_isobutane)
traj_idx_isobutane = traj_idx_isobutane[idx_train_isobutane]

# Keeping a trajectory for testing Isobutane
traj_idx_isopentane = np.asarray(dataset_isopentane['traj_idx'], dtype=int)
idx_train_isopentane = np.where(traj_idx_isopentane != 1)[0]
n_samples_isopentane = 1500
shuffle(idx_train_isopentane)
idx_train_isopentane = idx_train_isopentane[:n_samples_isopentane]
xyz_isopentane = np.asarray(dataset_isopentane['xyz'])[idx_train_isopentane]
zs_isopentane = np.array(dataset_isopentane['zs'])[idx_train_isopentane]
energies_isopentane = np.asarray(dataset_isopentane['ene'])[idx_train_isopentane] * 2625.50
energies_isopentane -= ref_energy
ene_scaled_isopentane = scaling.transform(zs_isopentane, energies_isopentane)
traj_idx_isopentane = traj_idx_isopentane[idx_train_isopentane]

# Updating the Data object
data.coordinates = np.asarray(list(xyz_meth) + list(xyz_ethane) + list(xyz_isobutane)+list(xyz_isopentane))
data.nuclear_charges = np.asarray(list(zs_meth) + list(zs_ethane) + list(zs_isobutane)+list(zs_isopentane))
data._set_ncompounds(len(data.nuclear_charges))
data.natoms = np.asarray([len(x) for x in data.nuclear_charges])
data.set_energies(np.asarray(list(ene_scaled_meth) + list(ene_scaled_ethane) + list(ene_scaled_isobutane)+list(ene_scaled_isopentane)))

# Create model
estimator = sklearn.pipeline.make_pipeline(
                qmlearn.representations.AtomCenteredSymmetryFunctions(data),
                qmlearn.models.NeuralNetwork(hl3=0)
                )

pickle.dump(estimator, open('model.pickle', 'wb'))
indices = np.arange(n_samples_meth+n_samples_ethane+n_samples_isobutane+n_samples_isopentane)
with open('idx.csv', 'w') as f:
    for i in indices:
        f.write('%s\n' % i)

joint_traj_idx = list(traj_idx_meth) + list(traj_idx_ethane+max(traj_idx_meth)) +\
                 list(traj_idx_isobutane+max(traj_idx_meth)+max(traj_idx_ethane)) + list(traj_idx_isopentane + max(traj_idx_meth)+max(traj_idx_ethane)+max(traj_idx_isobutane))

with open('groups.csv', 'w') as f:
    for i in indices:
        f.write('%s\n' % joint_traj_idx[i])
