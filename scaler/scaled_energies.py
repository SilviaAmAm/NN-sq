import h5py
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context("talk")
sns.set_style("white")

data_methane = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/methane_cn_dft.hdf5", "r")
data_isopentane = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/isopentane_cn_dft.hdf5", "r")
data_2isohex = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/2isohexane_cn_dft_pruned.hdf5", "r")
data_3isohex = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/3isohexane_cn_dft_pruned.hdf5", "r")
data_dimer = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/isopentane_dimer_cn_dft.hdf5", "r")
data_squal = h5py.File("/Volumes/Transcend/repositories/NN-sq/data_sets/squalane_cn_dft.hdf5", "r")

ref_ene = -133.1 * 2625.50

n_samples = 5000

ene_squal = np.array(data_squal.get("ene")[:n_samples]) * 2625.50
ene_squal = ene_squal - ref_ene
zs_squal = np.array(data_squal.get("zs")[:n_samples], dtype=np.int32)

ene_dimer = np.array(data_dimer.get("ene")[:n_samples]) * 2625.50
ene_dimer = ene_dimer - ref_ene
zs_dimer = np.array(data_dimer.get("zs")[:n_samples], dtype=np.int32)

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

zs_for_scaler_long = list(zs_methane) + list(zs_isopent) + list(zs_2hex) + list(zs_3hex) + list(zs_dimer) + list(zs_squal)
concat_ene = np.concatenate((ene_methane, ene_isopent, ene_2hex, ene_3hex, ene_dimer, ene_squal))

scaling = pickle.load(open("./scaler.pickle", "rb"))
concat_ene_scaled = scaling.transform(zs_for_scaler_long, concat_ene)

fig, ax = plt.subplots(1, figsize=(8,6))
ax.scatter(list(range(len(concat_ene))), concat_ene, label="Non scaled")
ax.scatter(list(range(len(concat_ene))), concat_ene_scaled, label="Scaled")
offset = 1e5
ax.text(0,concat_ene[0]+offset,'Methane')
ax.text(5000,concat_ene[5000]+offset,'Isopentane')
ax.text(10000,concat_ene[10000]-2*offset,'2-Isohexane')
ax.text(15000,concat_ene[15000]+offset,'3-Isohexane')
ax.text(20000,concat_ene[20000]+offset,'Isopentane dimer')
ax.text(25000,concat_ene[25000]+offset,'Squalane')
ax.legend()
ax.set_xlim((-500, 32000))
# ax.set_ylim((-2e2,2e2))
ax.set_xlabel("Frames")
ax.set_ylabel("Energy (kJ/mol)")
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.savefig("../images/scaling_all_zoom.png")
plt.show()
