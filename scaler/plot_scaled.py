import h5py
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context("talk")
sns.set_style("white")

data_methane = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/methane_cn_dft.hdf5", "r")
data_ethane = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/ethane_cn_dft.hdf5", "r")
data_isobutane = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/isobutane_cn_dft.hdf5", "r")
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

ene_isobutane = np.array(data_isobutane.get("ene")[:n_samples]) * 2625.50
ene_isobutane = ene_isobutane - ref_ene
zs_isobutane = np.array(data_isobutane.get("zs")[:n_samples], dtype=np.int32)

ene_ethane = np.array(data_ethane.get("ene")[:n_samples]) * 2625.50
ene_ethane = ene_ethane - ref_ene
zs_ethane = np.array(data_ethane.get("zs")[:n_samples], dtype=np.int32)

ene_methane = np.array(data_methane.get("ene")[:n_samples]) * 2625.50
ene_methane = ene_methane - ref_ene
zs_methane = np.array(data_methane.get("zs")[:n_samples], dtype=np.int32)

zs_for_scaler_long = list(zs_methane) + list(zs_ethane) + list(zs_isobutane) + list(zs_isopent) + list(zs_2hex) + list(zs_3hex) + list(zs_squal)
concat_ene = np.concatenate((ene_methane, ene_ethane, ene_isobutane, ene_isopent, ene_2hex, ene_3hex, ene_squal))

scaling = pickle.load(open("./scaler.pickle", "rb"))
concat_ene_scaled = scaling.transform(zs_for_scaler_long, concat_ene)

fig, ax = plt.subplots(1, figsize=(8,6))
ax.scatter(list(range(len(concat_ene))), concat_ene, label="Non scaled", s=30)
ax.scatter(list(range(len(concat_ene))), concat_ene_scaled, label="Scaled", s=30)
offset = 1e5
ax.text(0,concat_ene[0]+offset,'Methane')
ax.text(5500,concat_ene[5000]+offset,'Ethane')
ax.text(10500,concat_ene[10000]+offset,'Isobutane')
ax.text(14300,concat_ene[15000]-2.7*offset,'Isopentane')
ax.text(20500,concat_ene[20000]+offset,'2-Isohexane')
ax.text(25000,concat_ene[25000]-2.5*offset,'3-Isohexane')
ax.text(28000,concat_ene[30000]+offset,'Squalane')
ax.legend()
ax.set_xlim((-500, 34000))
ax.set_ylim((-3.25*1e6,0.5*1e6))
# ax.set_ylim((-2e2,2e2))
ax.set_xlabel("Frames")
ax.set_ylabel("Energy (kJ/mol)")
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig("/Volumes/Transcend/repositories/thesis/ffnn_results_fig/scaling.png")
plt.show()
