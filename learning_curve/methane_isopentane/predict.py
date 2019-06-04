from qml.qmlearn.preprocessing import AtomScaler
from qml.aglaia.aglaia import ARMP
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import model_selection as modsel
import pickle
from random import shuffle
import tensorflow as tf

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

ref_ene = -133.1 * 2625.50

tot_n_samples =  [300, 1000, 3000, 10000, 15000, 20000]

# Squalane data
xyz_squal = np.array(data_squal.get("xyz"))
ene_squal = np.array(data_squal.get("ene")) * 2625.50
ene_squal = ene_squal - ref_ene
zs_squal = np.array(data_squal.get("zs"), dtype=np.int32)
n_atoms_squal = len(zs_squal[0])
idx_squal = np.asarray(data_squal.get('traj_idx'), dtype=int)
fn_squal = np.asarray(data_squal.get('Filenumber'), dtype=int)

idx_squal, fn_squal, ene_squal, xyz_squal, zs_squal = sort(idx_squal, fn_squal, ene_squal, xyz_squal, zs_squal)

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

concat_xyz = np.concatenate((pad_xyz_methane, pad_xyz_isopentane, xyz_squal))
concat_ene = np.concatenate((ene_methane, ene_isopentane, ene_squal))
concat_zs = np.concatenate((pad_zs_methane, pad_zs_isopentane, zs_squal))

zs_for_scaler = list(zs_methane) + list(zs_isopentane) + list(zs_squal)

scaling = pickle.load(open("../../scaler/scaler.pickle", "rb"))
concat_ene_scaled = scaling.transform(zs_for_scaler, concat_ene)

n_basis = 16
r_min = 0.8
r_cut = 3.24756528424191
tau = 1.3399321961520696
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))

acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

methane_scores = []
isopentane_scores = []
squalane_scores = []

for samples in tot_n_samples:
    # Generate estimator
    estimator = ARMP(iterations=1130, l1_reg=9.6983542971344e-06, l2_reg=2.7847233160230936e-06, learning_rate=0.0009050324762982482, representation_name='acsf', representation_params=acsf_params, tensorboard=True, store_frequency=10, hidden_layer_sizes=(371,94,), batch_size=73)

    model_name = "model_" + str(samples)
    estimator.load_nn(model_name)

    estimator.set_properties(concat_ene_scaled)
    estimator.generate_representation(concat_xyz, concat_zs, method='fortran')

    pred_idx_methane = list(range(len(ene_methane)))
    pred_methane = estimator.predict(pred_idx_methane)
    methane_scores.append(estimator.score(pred_idx_methane))

    pred_idx_isopentane = list(range(len(ene_methane), len(ene_methane)+len(ene_isopentane)))
    pred_isopentane = estimator.predict(pred_idx_isopentane)
    isopentane_scores.append(estimator.score(pred_idx_isopentane))


    pred_idx_squal = list(range(len(ene_methane)+len(ene_isopentane), len(ene_methane)+len(ene_isopentane)+len(ene_squal)))
    pred_squal = estimator.predict(pred_idx_squal)
    squalane_scores.append(estimator.score(pred_idx_squal))

    del estimator
    tf.reset_default_graph()

    filename = "predictions_" + str(samples) + ".npz"
    np.savez(filename, pred_methane, concat_ene_scaled[pred_idx_methane], pred_isopentane, concat_ene_scaled[pred_idx_isopentane], pred_squal, concat_ene_scaled[pred_idx_squal])

np.savez("scores.npz", np.asarray(methane_scores), np.asarray(isopentane_scores), np.asarray(squalane_scores))