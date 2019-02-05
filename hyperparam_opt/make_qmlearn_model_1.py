import pickle
import numpy as np
import h5py
from qml.aglaia.aglaia import ARMP
from random import shuffle


ref_ene = -133.1 * 2625.5
scaling = pickle.load(open("../scaler/scaler.pickle", "rb"))
n_samples = 15000

data_methane = h5py.File("/home/sa16246/repositories/NN-sq/data_sets/methane_cn_dft.hdf5", "r")
data_squal = h5py.File("/home/sa16246/repositories/NN-sq/data_sets/squalane_cn_dft.hdf5", "r")

# Getting the data
ene_methane = np.array(data_methane.get("ene")) * 2625.50
ene_methane = ene_methane - ref_ene
zs_methane = np.array(data_methane.get("zs"), dtype=np.int32)
xyz_methane = np.array(data_methane.get("xyz"), dtype=np.int32)


ene_squal = np.array(data_squal.get("ene")) * 2625.50
ene_squal = ene_squal - ref_ene
zs_squal = np.array(data_squal.get("zs"), dtype=np.int32)
xyz_squal = np.array(data_squal.get("xyz"), dtype=np.int32)
n_atoms_squal = len(zs_squal[0])

# Padding the inputs
pad_xyz_methane = np.concatenate((xyz_methane, np.zeros((xyz_methane.shape[0], n_atoms_squal - xyz_methane.shape[1], 3))), axis=1)
pad_zs_methane = np.concatenate((zs_methane, np.zeros((zs_methane.shape[0], n_atoms_squal - zs_methane.shape[1]), dtype=np.int32)), axis=1)

# Scaling the data



# Taking 15000 random methane samples
all_idx = list(range(len(ene_methane)))
shuffle(all_idx)
train_idx = all_idx[:n_samples]



# Create the estimator
acsf_params = {"nRs2":14, "nRs3":14, "nTs":14, "rcut":3.29, "acut":3.29, "zeta":100.06564927139748, "eta":39.81824764370754}
estimator = ARMP(representation_name='acsf', representation_params=acsf_params, hidden_layer_sizes=(150,))

estimator.set_properties(concat_ene_scaled[:n_samples])
estimator.generate_representation(pad_xyz_methane, pad_zs_methane, method="fortran")

pickle.dump(estimator, open('model.pickle', 'wb'))

with open('idx.csv', 'w') as f:
    for i in range(n_samples):
        f.write('%s\n' % i)
