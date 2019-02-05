import pickle
import numpy as np
import sklearn.pipeline
from qml import qmlearn
import h5py

# Create data
dataset = h5py.File("../data_sets/methane_cn_dft.hdf5")

ref_energy = -133.1 * 2625.5
scaling = pickle.load(open("../scaler/scaler.pickle", "rb"))


data = qmlearn.Data()
traj_idx = np.asarray(dataset['traj_idx'], dtype=int)
# Keep trajectory 22 as a test set
mask = (traj_idx == 22)
data.coordinates = np.asarray(dataset['xyz'])[~mask]
data.nuclear_charges = np.array(dataset['zs'])[~mask]
data._set_ncompounds(len(data.nuclear_charges))
data.natoms = np.asarray([len(data.nuclear_charges[0])]*len(data.nuclear_charges))
energies = np.asarray(dataset['ene'])[~mask]*2625.50
energies -= ref_energy
ene_scaled = scaling.transform(data.nuclear_charges, energies)
data.set_energies(ene_scaled)

# Create model
estimator = sklearn.pipeline.make_pipeline(
                qmlearn.representations.AtomCenteredSymmetryFunctions(data),
                qmlearn.models.NeuralNetwork(hl2=0, hl3=0, iterations=50)
                )


pickle.dump(estimator, open('model.pickle', 'wb'))
indices = np.arange(len(data.coordinates))
with open('idx.csv', 'w') as f:
    for i in indices:
        f.write('%s\n' % i)
indices = traj_idx[~mask]
with open('groups.csv', 'w') as f:
    for i in indices:
        f.write('%s\n' % i)