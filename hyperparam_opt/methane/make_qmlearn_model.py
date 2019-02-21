import pickle
import numpy as np
import sklearn.pipeline
from qml import qmlearn
import h5py
from random import shuffle

# Create data
dataset = h5py.File("../../data_sets/methane_cn_dft.hdf5")

ref_energy = -133.1 * 2625.5
scaling = pickle.load(open("../../scaler/scaler.pickle", "rb"))

data = qmlearn.Data()
traj_idx = np.asarray(dataset['traj_idx'], dtype=int)

# Keeping a trajectory for testing
idx_train = np.where(traj_idx != 14)[0]

# Taking N samples for training
n_samples = 15000
shuffle(idx_train)
idx_train = idx_train[:n_samples]

data.coordinates = np.asarray(dataset['xyz'])[idx_train]
data.nuclear_charges = np.array(dataset['zs'])[idx_train]
data._set_ncompounds(len(data.nuclear_charges))
data.natoms = np.asarray([len(data.nuclear_charges[0])]*len(data.nuclear_charges))
energies = np.asarray(dataset['ene'])[idx_train]*2625.50
energies -= ref_energy
ene_scaled = scaling.transform(data.nuclear_charges, energies)
data.set_energies(ene_scaled)
traj_idx = traj_idx[idx_train]

# Create model
estimator = sklearn.pipeline.make_pipeline(
                qmlearn.representations.AtomCenteredSymmetryFunctions(data),
                qmlearn.models.NeuralNetwork(hl3=0)
                )

pickle.dump(estimator, open('model.pickle', 'wb'))
indices = np.arange(n_samples)
with open('idx.csv', 'w') as f:
    for i in indices:
        f.write('%s\n' % i)

with open('groups.csv', 'w') as f:
    for i in indices:
        f.write('%s\n' % traj_idx[i])