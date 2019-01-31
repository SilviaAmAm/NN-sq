import h5py
import numpy as np
import pickle
import matplotlib.pyplot as plt


data_methane = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/methane_cn_dft.hdf5", "r")
data_isopentane = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/isopentane_cn_dft.hdf5", "r")
data_2isohex = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/2isohexane_cn_dft_pruned.hdf5", "r")
data_3isohex = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/3isohexane_cn_dft_pruned.hdf5", "r")
data_squal = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/squalane_cn_dft.hdf5", "r")

ref_ene = -133.1 * 2625.50

n_samples = 5000

ene_squal = np.array(data_squal.get("ene")[:n_samples]) * 2625.50
ene_squal = ene_squal - ref_ene
zs_squal = np.array(data_squal.get("zs")[:n_samples], dtype=np.int32)

ene_3hex = np.array(data_3isohex.get("ene")[:n_samples]) * 2625.50
ene_3hex = ene_3hex - ref_ene
zs_3hex = np.array(data_3isohex.get("zs")[:n_samples], dtype=np.int32)

ene_2hex = np.array(data_2isohex.get("ene")[:n_samples]) * 2625.50
ene_2hex = ene_2hex - ref_ene
zs_2hex = np.array(data_2isohex.get("zs")[:n_samples], dtype=np.int32)

ene_isopent = np.array(data_isopentane.get("ene")[:n_samples]) * 2625.50
ene_isopent = ene_isopent - ref_ene
zs_isopent = np.array(data_isopentane.get("zs")[:n_samples], dtype=np.int32)

ene_methane = np.array(data_methane.get("ene")[:n_samples]) * 2625.50
ene_methane = ene_methane - ref_ene
zs_methane = np.array(data_methane.get("zs")[:n_samples], dtype=np.int32)

zs_for_scaler_long = list(zs_methane) + list(zs_isopent) + list(zs_2hex) + list(zs_3hex) + list(zs_squal)
concat_ene = np.concatenate((ene_methane, ene_isopent, ene_2hex, ene_3hex, ene_squal))

scaling = pickle.load(open("./scaler.pickle", "rb"))
concat_ene_scaled = scaling.transform(zs_for_scaler_long, concat_ene)

plt.scatter(list(range(len(concat_ene))), concat_ene)
plt.scatter(list(range(len(concat_ene))), concat_ene_scaled)
# plt.savefig("scaling_all.png")
plt.show()
exit()