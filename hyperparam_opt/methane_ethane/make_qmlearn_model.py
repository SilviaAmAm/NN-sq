import pickle
import numpy as np
import sklearn.pipeline
from qml import qmlearn
import h5py
from random import shuffle

# Create data
dataset_meth = h5py.File("../../data_sets/methane_cn_dft.hdf5")
dataset_ethane = h5py.File("../../data_sets/ethane_cn_dft.hdf5")

ref_energy = -133.1 * 2625.5
scaling = pickle.load(open("../../scaler/larger_scaler.pickle", "rb"))

# Data objects
data = qmlearn.Data()

# Keeping a trajectory for testing methane
traj_idx_meth = np.asarray(dataset_meth['traj_idx'], dtype=int)
idx_train_meth = np.where(traj_idx_meth != 14)[0]
n_samples_meth = 10000
shuffle(idx_train_meth)
idx_train_meth = idx_train_meth[:n_samples_meth]
xyz_meth = np.asarray(dataset_meth['xyz'])[idx_train_meth]
zs_meth = np.array(dataset_meth['zs'])[idx_train_meth]
energies_meth = np.asarray(dataset_meth['ene'])[idx_train_meth] * 2625.50
energies_meth -= ref_energy
ene_scaled_meth = scaling.transform(zs_meth, energies_meth)
traj_idx_meth = traj_idx_meth[idx_train_meth]

# Keeping a trajectory for testing ethaneane
traj_idx_ethane = np.asarray(dataset_ethane['traj_idx'], dtype=int)
idx_train_ethane = np.where(traj_idx_ethane != 7)[0]
n_samples_ethane = 5000
shuffle(idx_train_ethane)
idx_train_ethane = idx_train_ethane[:n_samples_ethane]
xyz_ethane = np.asarray(dataset_ethane['xyz'])[idx_train_ethane]
zs_ethane = np.array(dataset_ethane['zs'])[idx_train_ethane]
energies_ethane = np.asarray(dataset_ethane['ene'])[idx_train_ethane] * 2625.50
energies_ethane -= ref_energy
ene_scaled_ethane = scaling.transform(zs_ethane, energies_ethane)
traj_idx_ethane = traj_idx_ethane[idx_train_ethane]


# Updating the Data object
data.coordinates = np.asarray(list(xyz_meth) + list(xyz_ethane))
data.nuclear_charges = np.asarray(list(zs_meth) + list(zs_ethane))
data._set_ncompounds(len(data.nuclear_charges))
data.natoms = np.asarray([len(x) for x in data.nuclear_charges])
data.set_energies(np.asarray(list(ene_scaled_meth) + list(ene_scaled_ethane)))

# Create model
estimator = sklearn.pipeline.make_pipeline(
                qmlearn.representations.AtomCenteredSymmetryFunctions(data),
                qmlearn.models.NeuralNetwork(hl3=0)
                )

pickle.dump(estimator, open('model.pickle', 'wb'))
indices = np.arange(n_samples_meth+n_samples_ethane)
with open('idx.csv', 'w') as f:
    for i in indices:
        f.write('%s\n' % i)

joint_traj_idx = list(traj_idx_meth) + list(traj_idx_ethane+max(traj_idx_meth))
with open('groups.csv', 'w') as f:
    for i in indices:
        f.write('%s\n' % joint_traj_idx[i])