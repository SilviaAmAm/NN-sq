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
data_ethane = h5py.File("../../data_sets/ethane_cn_dft.hdf5")
data_isobutane = h5py.File("../../data_sets/isobutane_cn_dft.hdf5")
data_isopentane = h5py.File("../../data_sets/isopentane_cn_dft.hdf5", "r")
data_squal = h5py.File("../../data_sets/squalane_cn_dft.hdf5", "r")

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

# Isopentane
idx_isopentane = np.asarray(data_isopentane.get('traj_idx'), dtype=int)
idx_isopentane_traj = np.where(idx_isopentane == 22)[0]

xyz_isopentane = np.array(data_isopentane.get("xyz"))[idx_isopentane_traj]
zs_isopentane = np.array(data_isopentane.get("zs"), dtype=np.int32)[idx_isopentane_traj]
ene_isopentane = np.array(data_isopentane.get("ene"))[idx_isopentane_traj]*2625.5 - ref_ene

pad_xyz_isopentane = np.concatenate((xyz_isopentane, np.zeros((xyz_isopentane.shape[0], n_atoms_squal - xyz_isopentane.shape[1], 3))), axis=1)
pad_zs_isopentane = np.concatenate((zs_isopentane, np.zeros((zs_isopentane.shape[0], n_atoms_squal - zs_isopentane.shape[1]), dtype=np.int32)), axis=1)

# Isobutane
idx_isobutane = np.asarray(data_isobutane.get('traj_idx'), dtype=int)
idx_isobutane_traj = np.where(idx_isobutane == 2)[0]

xyz_isobutane = np.array(data_isobutane.get("xyz"))[idx_isobutane_traj]
zs_isobutane = np.array(data_isobutane.get("zs"), dtype=np.int32)[idx_isobutane_traj]
ene_isobutane = np.array(data_isobutane.get("ene"))[idx_isobutane_traj]*2625.5 - ref_ene

pad_xyz_isobutane = np.concatenate((xyz_isobutane, np.zeros((xyz_isobutane.shape[0], n_atoms_squal - xyz_isobutane.shape[1], 3))), axis=1)
pad_zs_isobutane = np.concatenate((zs_isobutane, np.zeros((zs_isobutane.shape[0], n_atoms_squal - zs_isobutane.shape[1]), dtype=np.int32)), axis=1)

# Ethane
idx_ethane = np.asarray(data_ethane.get('traj_idx'), dtype=int)
idx_ethane_traj = np.where(idx_ethane == 7)[0]

xyz_ethane = np.array(data_ethane.get("xyz"))[idx_ethane_traj]
zs_ethane = np.array(data_ethane.get("zs"), dtype=np.int32)[idx_ethane_traj]
ene_ethane = np.array(data_ethane.get("ene"))[idx_ethane_traj]*2625.5 - ref_ene

pad_xyz_ethane = np.concatenate((xyz_ethane, np.zeros((xyz_ethane.shape[0], n_atoms_squal - xyz_ethane.shape[1], 3))), axis=1)
pad_zs_ethane = np.concatenate((zs_ethane, np.zeros((zs_ethane.shape[0], n_atoms_squal - zs_ethane.shape[1]), dtype=np.int32)), axis=1)

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

concat_xyz = np.concatenate((pad_xyz_methane, pad_xyz_ethane, pad_xyz_isobutane, pad_xyz_isopentane, xyz_squal))
concat_ene = np.concatenate((ene_methane, ene_ethane, ene_isobutane, ene_isopentane, ene_squal))
concat_zs = np.concatenate((pad_zs_methane,  pad_zs_ethane, pad_zs_isobutane, pad_zs_isopentane, zs_squal))

zs_for_scaler = list(zs_methane) + list(zs_ethane) + list(zs_isobutane) + list(zs_isopentane) + list(zs_squal)

scaling = pickle.load(open("../../scaler/scaler.pickle", "rb"))
concat_ene_scaled = scaling.transform(zs_for_scaler, concat_ene)

n_basis = 13
r_min = 0.8
r_cut = 3.440665282995163
tau = 1.8930017259735163
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))

acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

# Generate estimator
estimator = ARMP(iterations=1181, l1_reg=0.00025021816931208597, l2_reg=4.0694790725485916e-05, learning_rate=0.0007055546397595542, representation_name='acsf',
                 representation_params=acsf_params, tensorboard=True, store_frequency=10, hidden_layer_sizes=(94,174,), batch_size=26)

estimator.load_nn("saved_model")

estimator.set_properties(concat_ene_scaled)
estimator.generate_representation(concat_xyz, concat_zs, method='fortran')

pred_idx_methane = list(range(len(ene_methane)))
pred_methane = estimator.predict(pred_idx_methane)
print("Methane trajectory score: %s" % str(estimator.score(pred_idx_methane)))

pred_idx_ethane = list(range(len(ene_methane), len(ene_methane)+len(ene_ethane)))
pred_ethane = estimator.predict(pred_idx_ethane)
print("Ethane trajectory score: %s" % str(estimator.score(pred_idx_ethane)))

pred_idx_isobutane = list(range(len(ene_methane)+len(ene_ethane), len(ene_methane)+len(ene_ethane)+len(ene_isobutane)))
pred_isobutane = estimator.predict(pred_idx_isobutane)
print("Isobutane trajectory score: %s" % str(estimator.score(pred_idx_isobutane)))

pred_idx_isopentane = list(range(len(ene_methane)+len(ene_ethane)+len(ene_isobutane), len(ene_methane)+len(ene_ethane)+len(ene_isobutane)+len(ene_isopentane)))
pred_isopentane = estimator.predict(pred_idx_isopentane)
print("Isopentane trajectory score: %s" % str(estimator.score(pred_idx_isopentane)))


pred_idx_squal = list(range(len(ene_methane)+len(ene_ethane)+len(ene_isobutane)+len(ene_isopentane), len(ene_methane)+len(ene_ethane)+len(ene_isobutane)+len(ene_isopentane)+len(ene_squal)))
pred_squal = estimator.predict(pred_idx_squal)
print("Squalane trajectory score: %s" % str(estimator.score(pred_idx_squal)))

np.savez("sorted_predictions_right_model.npz", pred_methane, concat_ene_scaled[pred_idx_methane], pred_ethane, concat_ene_scaled[pred_idx_ethane], pred_isobutane, concat_ene_scaled[pred_idx_isobutane], pred_isopentane, concat_ene_scaled[pred_idx_isopentane], pred_squal, concat_ene_scaled[pred_idx_squal])