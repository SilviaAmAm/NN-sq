import sklearn.pipeline
from qml import qmlearn
import numpy as np
import h5py
import pickle
from random import shuffle
import time

ref_ene = -133.1 * 2625.5
scaling = pickle.load(open("../../scaler/scaler.pickle", "rb"))

# Data objects
data = qmlearn.Data()

# Getting the dataset
dataset_meth = h5py.File("../../data_sets/methane_cn_dft.hdf5", "r")
dataset_squal = h5py.File("../../data_sets/squalane_cn_dft.hdf5", "r")

# Training data for methane
traj_idx_meth = np.asarray(dataset_meth['traj_idx'], dtype=int)
idx_train_meth = np.where(traj_idx_meth != 14)[0]

n_samples_meth = 15000
shuffle(idx_train_meth)
idx_train_meth = idx_train_meth[:n_samples_meth]
xyz_meth_train = np.asarray(dataset_meth['xyz'])[idx_train_meth]
zs_meth_train = np.array(dataset_meth['zs'])[idx_train_meth]
energies_meth_train = np.asarray(dataset_meth['ene'])[idx_train_meth] * 2625.50
energies_meth_train -= ref_ene
ene_scaled_meth_train = scaling.transform(zs_meth_train, energies_meth_train)

# Keeping a test trajectory
idx_test_meth = np.where(traj_idx_meth == 14)[0]
xyz_meth_test = np.asarray(dataset_meth['xyz'])[idx_test_meth]
zs_meth_test = np.array(dataset_meth['zs'])[idx_test_meth]
energies_meth_test = np.asarray(dataset_meth['ene'])[idx_test_meth] * 2625.50
energies_meth_test -= ref_ene
ene_scaled_meth_test = scaling.transform(zs_meth_test, energies_meth_test)
n_samples_test = len(ene_scaled_meth_test)

# GGetting squalane data
xyz_squal = np.asarray(dataset_squal['xyz'])
zs_squal = np.array(dataset_squal['zs'])
energies_squal = np.asarray(dataset_squal['ene'])* 2625.50
energies_squal -= ref_ene
ene_scaled_squal = scaling.transform(zs_squal, energies_squal)
n_samples_squal = len(ene_scaled_squal)

# Updating the Data object
data.coordinates = np.asarray(list(xyz_meth_train) + list(xyz_meth_test) +list(xyz_squal))
data.nuclear_charges = np.asarray(list(zs_meth_train) + list(zs_meth_test) + list(zs_squal))
data._set_ncompounds(len(data.nuclear_charges))
data.natoms = np.asarray([len(x) for x in data.nuclear_charges])
data.set_energies(np.asarray(list(ene_scaled_meth_train) + list(ene_scaled_meth_test) + list(ene_scaled_squal)))


# ACSF parameters
n_basis = 15
r_cut = 3.399097317902879
tau = 2.6849677326324266

# Create model
estimator = sklearn.pipeline.make_pipeline(
                qmlearn.representations.AtomCenteredSymmetryFunctions(data=data, cutoff=r_cut, precision=tau, nbasis=n_basis),
                qmlearn.models.NeuralNetwork(iterations=100, hl1=115, hl2=75, batch_size=11, learning_rate=0.001205221637691679, l1_reg=1.0004741874144448e-06,
                 l2_reg=1.5864761374564372e-08, size=94)
                )

# Indices of training/test/squalane
training_samples = list(range(n_samples_meth))
test_samples = list(range(n_samples_meth, n_samples_meth+n_samples_test))
squalane_samples = list(range(n_samples_meth+n_samples_test, n_samples_meth+n_samples_test+n_samples_squal))

# Training
st = time.time()
estimator.fit(training_samples)
end = time.time()
print("Training took %s s for 100 iterations" % str(end-st))

# Scoring the model
print("The score on the methane trajectory is:")
score_traj = estimator.score(test_samples)
print(score_traj)

print("The score on the squalane trajectory is:")
score_squal = estimator.score(squalane_samples)
print(score_squal)

ene_pred_traj = estimator.predict(test_samples)
ene_pred_squal = estimator.predict(squalane_samples)

np.savez("predictions.npz", ene_pred_traj, ene_scaled_meth_test, ene_pred_squal, ene_scaled_squal)
