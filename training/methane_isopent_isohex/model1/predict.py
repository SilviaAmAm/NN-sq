from qml.qmlearn.preprocessing import AtomScaler
from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel
import pickle
from random import shuffle

def sort(traj_idx, fn, ene, xyz, zs):
    # Sorting the trajectories
    idx_sorted = traj_idx.argsort()

    ene = ene[idx_sorted]
    xyz = xyz[idx_sorted]
    zs = zs[idx_sorted]
    traj_idx = traj_idx[idx_sorted]
    fn = fn[idx_sorted]

    n_traj = np.unique(traj_idx)

    for item in n_traj:
        indices = np.where(traj_idx == item)

        idx_sorted = fn[indices].argsort()

        ene[indices] = ene[indices][idx_sorted]
        traj_idx[indices] = traj_idx[indices][idx_sorted]
        fn[indices] = fn[indices][idx_sorted]
        xyz[indices] = xyz[indices][idx_sorted]
        zs[indices] = zs[indices][idx_sorted]
        
    return traj_idx, fn, ene, xyz, zs
    

# Getting the dataset
data_methane = h5py.File("../../data_sets/methane_cn_dft.hdf5", "r")
data_isopentane = h5py.File("../../data_sets/isopentane_cn_dft.hdf5", "r")
data_squal = h5py.File("../../data_sets/squalane_cn_dft.hdf5", "r")
data_2isohex = h5py.File("../../data_sets/2isohexane_cn_dft_pruned.hdf5")
data_3isohex = h5py.File("../../data_sets/3isohexane_cn_dft_pruned.hdf5")

ref_ene = -133.1 * 2625.50

# Squalane data
xyz_squal = np.array(data_squal.get("xyz"))
ene_squal = np.array(data_squal.get("ene")) * 2625.50
ene_squal = ene_squal - ref_ene
zs_squal = np.array(data_squal.get("zs"), dtype=np.int32)
n_atoms_squal = len(zs_squal[0])
idx_squal = np.asarray(data_squal.get('traj_idx'), dtype=int)
fn_squal = np.asarray(data_squal.get('Filenumber'), dtype=int)

idx_squal, fn_squal, ene_squal, xyz_squal, zs_squal = sort(idx_squal, fn_squal, ene_squal, xyz_squal, zs_squal)

# 3-isohexane
idx_3isohex = np.asarray(data_3isohex.get('traj_idx'), dtype=int)
idx_3isohex_traj = np.where(idx_3isohex == 14)[0]

xyz_3isohex = np.array(data_3isohex.get("xyz"))[idx_3isohex_traj]
zs_3isohex = np.array(data_3isohex.get("zs"), dtype=np.int32)[idx_3isohex_traj]
ene_3isohex = np.array(data_3isohex.get("ene"))[idx_3isohex_traj]*2625.5 - ref_ene

pad_xyz_3isohex = np.concatenate((xyz_3isohex, np.zeros((xyz_3isohex.shape[0], n_atoms_squal - xyz_3isohex.shape[1], 3))), axis=1)
pad_zs_3isohex = np.concatenate((zs_3isohex, np.zeros((zs_3isohex.shape[0], n_atoms_squal - zs_3isohex.shape[1]), dtype=np.int32)), axis=1)


# 2-isohexane
idx_2isohex = np.asarray(data_2isohex.get('traj_idx'), dtype=int)
idx_2isohex_traj = np.where(idx_2isohex == 12)[0]

xyz_2isohex = np.array(data_2isohex.get("xyz"))[idx_2isohex_traj]
zs_2isohex = np.array(data_2isohex.get("zs"), dtype=np.int32)[idx_2isohex_traj]
ene_2isohex = np.array(data_2isohex.get("ene"))[idx_2isohex_traj]*2625.5 - ref_ene

pad_xyz_2isohex = np.concatenate((xyz_2isohex, np.zeros((xyz_2isohex.shape[0], n_atoms_squal - xyz_2isohex.shape[1], 3))), axis=1)
pad_zs_2isohex = np.concatenate((zs_2isohex, np.zeros((zs_2isohex.shape[0], n_atoms_squal - zs_2isohex.shape[1]), dtype=np.int32)), axis=1)


# Isopentane
idx_isopentane = np.asarray(data_isopentane.get('traj_idx'), dtype=int)
idx_isopentane_traj = np.where(idx_isopentane == 22)[0]

xyz_isopentane = np.array(data_isopentane.get("xyz"))[idx_isopentane_traj]
zs_isopentane = np.array(data_isopentane.get("zs"), dtype=np.int32)[idx_isopentane_traj]
ene_isopentane = np.array(data_isopentane.get("ene"))[idx_isopentane_traj]*2625.5 - ref_ene

pad_xyz_isopentane = np.concatenate((xyz_isopentane, np.zeros((xyz_isopentane.shape[0], n_atoms_squal - xyz_isopentane.shape[1], 3))), axis=1)
pad_zs_isopentane = np.concatenate((zs_isopentane, np.zeros((zs_isopentane.shape[0], n_atoms_squal - zs_isopentane.shape[1]), dtype=np.int32)), axis=1)

# Methane
idx_methane = np.asarray(data_methane.get('traj_idx'), dtype=int)

idx_methane_traj = np.where(idx_methane == 14)[0]

xyz_methane = np.array(data_methane.get("xyz"))[idx_methane_traj]
zs_methane = np.array(data_methane.get("zs"), dtype=np.int32)[idx_methane_traj]
ene_methane = np.array(data_methane.get("ene"))[idx_methane_traj]*2625.5 - ref_ene
fn_methane = np.asarray(data_methane.get('Filenumber'), dtype=int)[idx_methane_traj]
idx_methane = idx_methane[idx_methane_traj]

idx_methane, fn_methane, ene_methane, xyz_methane, zs_methane = sort(idx_methane, fn_methane, ene_methane, xyz_methane, zs_methane)

pad_xyz_methane = np.concatenate((xyz_methane, np.zeros((xyz_methane.shape[0], n_atoms_squal - xyz_methane.shape[1], 3))), axis=1)
pad_zs_methane = np.concatenate((zs_methane, np.zeros((zs_methane.shape[0], n_atoms_squal - zs_methane.shape[1]), dtype=np.int32)), axis=1)

concat_xyz = np.concatenate((pad_xyz_methane, pad_xyz_isopentane, pad_xyz_2isohex, pad_xyz_3isohex, xyz_squal))
concat_ene = np.concatenate((ene_methane, ene_isopentane, ene_2isohex, ene_3isohex, ene_squal))
concat_zs = np.concatenate((pad_zs_methane, pad_zs_isopentane, pad_zs_2isohex, pad_zs_3isohex, zs_squal))

zs_for_scaler = list(zs_methane) + list(zs_isopentane) + list(zs_2isohex) + list(zs_3isohex) + list(zs_squal)

scaling = pickle.load(open("../../scaler/scaler.pickle", "rb"))
concat_ene_scaled = scaling.transform(zs_for_scaler, concat_ene)


n_basis = 16
r_min = 0.8
r_cut = 4.4826433285055
tau = 1.703369115912955
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))

acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

# Generate estimator
estimator = ARMP(iterations=927, l1_reg=0.00017541494055395613, l2_reg=1.3474385177740575e-08, learning_rate=0.0007658014586464996, representation_name='acsf',
                 representation_params=acsf_params, tensorboard=True, store_frequency=10, hidden_layer_sizes=(135,75,), batch_size=14)


estimator.load_nn("saved_model_1")

estimator.set_properties(concat_ene_scaled)
estimator.generate_representation(concat_xyz, concat_zs, method='fortran')

pred_idx_methane = list(range(len(ene_methane)))
pred_methane = estimator.predict(pred_idx_methane)
print("Methane trajectory score: %s" % str(estimator.score(pred_idx_methane)))

pred_idx_isopentane = list(range(len(ene_methane), len(ene_methane)+len(ene_isopentane)))
pred_isopentane = estimator.predict(pred_idx_isopentane)
print("Isopentane trajectory score: %s" % str(estimator.score(pred_idx_isopentane)))

pred_idx_2isohex = list(range(len(ene_methane)+len(ene_isopentane), len(ene_methane)+len(ene_isopentane)+len(ene_2isohex)))
pred_2isohex = estimator.predict(pred_idx_2isohex)
print("2-isohexane trajectory score: %s" % str(estimator.score(pred_idx_2isohex)))

pred_idx_3isohex = list(range(len(ene_methane)+len(ene_isopentane)+len(ene_2isohex), len(ene_methane)+len(ene_isopentane)+len(ene_2isohex)+len(ene_3isohex)))
pred_3isohex = estimator.predict(pred_idx_3isohex)
print("3-isohexane trajectory score: %s" % str(estimator.score(pred_idx_3isohex)))

pred_idx_squal = list(range(len(ene_methane)+len(ene_isopentane)+len(ene_2isohex)+len(ene_3isohex), len(ene_methane)+len(ene_isopentane)+len(ene_2isohex)+len(ene_3isohex)+len(ene_squal)))
pred_squal = estimator.predict(pred_idx_squal)
print("Squalane trajectory score: %s" % str(estimator.score(pred_idx_squal)))

np.savez("predictions.npz", pred_methane, concat_ene_scaled[pred_idx_methane], pred_isopentane, concat_ene_scaled[pred_idx_isopentane], pred_2isohex, concat_ene_scaled[pred_idx_2isohex], pred_3isohex, concat_ene_scaled[pred_idx_3isohex], pred_squal, concat_ene_scaled[pred_idx_squal])