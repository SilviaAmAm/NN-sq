import pickle
import numpy as np
import sklearn.pipeline
from qml import qmlearn
import h5py
from random import shuffle

# Create data
dataset_meth = h5py.File("../../data_sets/methane_cn_dft.hdf5")
dataset_isopent = h5py.File("../../data_sets/isopentane_cn_dft.hdf5")
dataset_2isohex = h5py.File("../../data_sets/2isohexane_cn_dft_pruned.hdf5")
dataset_3isohex = h5py.File("../../data_sets/3isohexane_cn_dft_pruned.hdf5")

ref_energy = -133.1 * 2625.5
scaling = pickle.load(open("../../scaler/scaler.pickle", "rb"))

# Data objects
data = qmlearn.Data()

# Keeping a trajectory for testing methane
traj_idx_meth = np.asarray(dataset_meth['traj_idx'], dtype=int)
idx_train_meth = np.where(traj_idx_meth != 14)[0]
n_samples_meth = 8000
shuffle(idx_train_meth)
idx_train_meth = idx_train_meth[:n_samples_meth]
xyz_meth = np.asarray(dataset_meth['xyz'])[idx_train_meth]
zs_meth = np.array(dataset_meth['zs'])[idx_train_meth]
energies_meth = np.asarray(dataset_meth['ene'])[idx_train_meth] * 2625.50
energies_meth -= ref_energy
ene_scaled_meth = scaling.transform(zs_meth, energies_meth)
traj_idx_meth = traj_idx_meth[idx_train_meth]

# Keeping a trajectory for testing isopentane
traj_idx_isopent = np.asarray(dataset_isopent['traj_idx'], dtype=int)
idx_train_isopent = np.where(traj_idx_isopent != 1)[0]
n_samples_isopent = 4000
shuffle(idx_train_isopent)
idx_train_isopent = idx_train_isopent[:n_samples_isopent]
xyz_isopent = np.asarray(dataset_isopent['xyz'])[idx_train_isopent]
zs_isopent = np.array(dataset_isopent['zs'])[idx_train_isopent]
energies_isopent = np.asarray(dataset_isopent['ene'])[idx_train_isopent] * 2625.50
energies_isopent -= ref_energy
ene_scaled_isopent = scaling.transform(zs_isopent, energies_isopent)
traj_idx_isopent = traj_idx_isopent[idx_train_isopent]

# Keeping a trajectory for testing 2-isohexane
traj_idx_2isohex = np.asarray(dataset_2isohex['traj_idx'], dtype=int)
idx_train_2isohex = np.where(traj_idx_2isohex != 2)[0]
n_samples_2isohex = 1500
shuffle(idx_train_2isohex)
idx_train_2isohex = idx_train_2isohex[:n_samples_2isohex]
xyz_2isohex = np.asarray(dataset_2isohex['xyz'])[idx_train_2isohex]
zs_2isohex = np.array(dataset_2isohex['zs'])[idx_train_2isohex]
energies_2isohex = np.asarray(dataset_2isohex['ene'])[idx_train_2isohex] * 2625.50
energies_2isohex -= ref_energy
ene_scaled_2isohex = scaling.transform(zs_2isohex, energies_2isohex)
traj_idx_2isohex = traj_idx_2isohex[idx_train_2isohex]

# Keeping a trajectory for testing 3-isohexane
traj_idx_3isohex = np.asarray(dataset_3isohex['traj_idx'], dtype=int)
idx_train_3isohex = np.where(traj_idx_3isohex != 13)[0]
n_samples_3isohex = 1500
shuffle(idx_train_3isohex)
idx_train_3isohex = idx_train_3isohex[:n_samples_3isohex]
xyz_3isohex = np.asarray(dataset_3isohex['xyz'])[idx_train_3isohex]
zs_3isohex = np.array(dataset_3isohex['zs'])[idx_train_3isohex]
energies_3isohex = np.asarray(dataset_3isohex['ene'])[idx_train_3isohex] * 2625.50
energies_3isohex -= ref_energy
ene_scaled_3isohex = scaling.transform(zs_3isohex, energies_3isohex)
traj_idx_3isohex = traj_idx_3isohex[idx_train_3isohex]

# Updating the Data object
data.coordinates = np.asarray(list(xyz_meth) + list(xyz_isopent) + list(xyz_2isohex) + list(xyz_3isohex))
data.nuclear_charges = np.asarray(list(zs_meth) + list(zs_isopent) + list(zs_2isohex) + list(zs_3isohex))
data._set_ncompounds(len(data.nuclear_charges))
data.natoms = np.asarray([len(x) for x in data.nuclear_charges])
data.set_energies(np.asarray(list(ene_scaled_meth) + list(ene_scaled_isopent) + list(ene_scaled_2isohex) + list(ene_scaled_3isohex)))

# Create model
estimator = sklearn.pipeline.make_pipeline(
                qmlearn.representations.AtomCenteredSymmetryFunctions(data),
                qmlearn.models.NeuralNetwork(hl3=0)
                )

pickle.dump(estimator, open('model.pickle', 'wb'))
indices = np.arange(n_samples_meth+n_samples_isopent+n_samples_2isohex+n_samples_3isohex)
with open('idx.csv', 'w') as f:
    for i in indices:
        f.write('%s\n' % i)

joint_traj_idx = list(traj_idx_meth) + list(traj_idx_isopent+max(traj_idx_meth)) +\
                 list(traj_idx_2isohex+max(traj_idx_meth)+max(traj_idx_isopent)) +\
                 list(traj_idx_3isohex+max(traj_idx_meth)+max(traj_idx_isopent)+max(traj_idx_2isohex))
with open('groups.csv', 'w') as f:
    for i in indices:
        f.write('%s\n' % joint_traj_idx[i])