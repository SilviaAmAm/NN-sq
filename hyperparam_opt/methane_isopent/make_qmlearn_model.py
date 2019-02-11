import pickle
import numpy as np
import sklearn.pipeline
from qml import qmlearn
import h5py
from random import shuffle

# Create data
dataset_meth = h5py.File("../../data_sets/methane_cn_dft.hdf5")
dataset_isopent = h5py.File("../../data_sets/isopentane_cn_dft.hdf5")

ref_energy = -133.1 * 2625.5
scaling = pickle.load(open("../../scaler/scaler.pickle", "rb"))

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

# Keeping a trajectory for testing isopentane
traj_idx_isopent = np.asarray(dataset_isopent['traj_idx'], dtype=int)
idx_train_isopent = np.where(traj_idx_isopent != 22)[0]
n_samples_isopent = 7500
shuffle(idx_train_isopent)
idx_train_isopent = idx_train_isopent[:n_samples_isopent]
xyz_isopent = np.asarray(dataset_isopent['xyz'])[idx_train_isopent]
zs_isopent = np.array(dataset_isopent['zs'])[idx_train_isopent]
energies_isopent = np.asarray(dataset_isopent['ene'])[idx_train_isopent] * 2625.50
energies_isopent -= ref_energy
ene_scaled_isopent = scaling.transform(zs_isopent, energies_isopent)


# Updating the Data object
data.coordinates = np.asarray(list(xyz_meth) + list(xyz_isopent))
data.nuclear_charges = np.asarray(list(zs_meth) + list(zs_isopent))
data._set_ncompounds(len(data.nuclear_charges))
data.natoms = np.asarray([len(x) for x in data.nuclear_charges])
data.set_energies(np.asarray(list(ene_scaled_meth) + list(ene_scaled_isopent)))

# Create model
estimator = sklearn.pipeline.make_pipeline(
                qmlearn.representations.AtomCenteredSymmetryFunctions(data),
                qmlearn.models.NeuralNetwork(hl3=0)
                )

pickle.dump(estimator, open('model.pickle', 'wb'))
indices = np.arange(n_samples_meth+n_samples_isopent)
with open('idx.csv', 'w') as f:
    for i in indices:
        f.write('%s\n' % i)
