from qml.qmlearn.preprocessing import AtomScaler
from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel


# Getting the dataset
data_methane = h5py.File("/home/sa16246/repositories/NN-sq/data_sets/methane_cn_dft.hdft", "r")
data_isopentane = h5py.File("/home/sa16246/repositories/NN-sq/data_sets/isopentane_cn_dft.hdf5", "r")
data_2isohex = h5py.File("/home/sa16246/repositories/NN-sq/data_sets/2isohexane_cn_dft_pruned.hdf5", "r")

ref_ene = data_methane.get("ene")[0] * 2625.50

n_samples = 5000

xyz_hex = np.array(data_2isohex.get("xyz")[:n_samples])
ene_hex = np.array(data_2isohex.get("ene")[:n_samples]) * 2625.50
ene_hex = ene_hex - ref_ene
zs_hex = np.array(data_2isohex.get("zs")[:n_samples], dtype=np.int32)
n_atoms_hex = len(zs_hex[0])

xyz_isopent = np.array(data_isopentane.get("xyz")[:n_samples])
ene_isopent = np.array(data_isopentane.get("ene")[:n_samples]) * 2625.50
ref_ene = data_isopentane.get("ene")[0] * 2625.50
ene_isopent = ene_isopent - ref_ene
zs_isopent = np.array(data_isopentane.get("zs")[:n_samples], dtype=np.int32)

pad_xyz_isopent = np.concatenate((xyz_isopent, np.zeros((xyz_isopent.shape[0], n_atoms_hex - xyz_isopent.shape[1], 3))), axis=1)
pad_zs_isopent = np.concatenate((zs_isopent, np.zeros((zs_isopent.shape[0], n_atoms_hex - xyz_isopent.shape[1]), dtype=np.int32)), axis=1)

xyz_methane = np.array(data_methane.get("xyz")[:n_samples])
ene_methane = np.array(data_methane.get("ene")[:n_samples]) * 2625.50
ene_methane = ene_methane - ref_ene
zs_methane = np.array(data_methane.get("zs")[:n_samples], dtype=np.int32)

pad_xyz_methane = np.concatenate((xyz_methane, np.zeros((xyz_methane.shape[0], n_atoms_hex - xyz_methane.shape[1], 3))), axis=1)
pad_zs_methane = np.concatenate((zs_methane, np.zeros((zs_methane.shape[0], n_atoms_hex - zs_methane.shape[1]), dtype=np.int32)), axis=1)

zs_for_scaler =  list(zs_methane) + list(zs_isopent) + list(zs_hex)

concat_xyz = np.concatenate((pad_xyz_methane, pad_xyz_isopent, xyz_hex))
concat_ene = np.concatenate((ene_methane, ene_isopent, ene_hex))
concat_zs = np.concatenate((pad_zs_methane, pad_zs_isopent, zs_hex))

scaling = AtomScaler()
concat_ene_scaled = scaling.fit_transform(zs_for_scaler, concat_ene)

acsf_params={"nRs2":15, "nRs3":15, "nTs":15, "rcut":5, "acut":5, "zeta":220.127, "eta":30.8065}

# Generate estimator
estimator = ARMP(iterations=500, l1_reg=0.0005, l2_reg=0.005, learning_rate=0.0001, representation_name='acsf',
                 representation_params=acsf_params, tensorboard=True, store_frequency=25, hidden_layer_sizes=(50,30,10), batch_size=250)

estimator.set_properties(concat_ene_scaled)
estimator.generate_representation(concat_xyz, concat_zs, method='fortran')

# Indices for methane + isopentane data
idx = list(range(3*n_samples))
idx_train, idx_test = modsel.train_test_split(idx, random_state=42, test_size=0.1, shuffle=True)

estimator.fit(idx_train)

estimator.save_nn()

print("The score the methane/isopentane/2hexane data set:")
score = estimator.score(idx_test)
print(score)

# print("The score on the squalane data:")
# idx_squal = list(range(2*n_samples, len(concat_ene)))
# score_squal = estimator.score(idx_squal)
# print(score_squal)
#
# pred_squal = estimator.predict(idx_squal)
# np.savez("pred_squal.npz", pred_squal, concat_ene_scaled[idx_squal])


