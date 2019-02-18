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

concat_xyz = np.concatenate((pad_xyz_methane, xyz_squal))
concat_ene = np.concatenate((ene_methane, ene_squal))
concat_zs = np.concatenate((pad_zs_methane, zs_squal))

zs_for_scaler = list(zs_methane) + list(zs_squal)

scaling = pickle.load(open("../../scaler/scaler.pickle", "rb"))
concat_ene_scaled = scaling.transform(zs_for_scaler, concat_ene)


# ACSF parameters
n_basis = 19
r_min = 0.8
r_cut = 4.179471213058485
tau = 1.4149304209115
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))

acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

# Generate estimator
estimator = ARMP(iterations=500, l1_reg=9.062776275956941e-06, l2_reg=3.093356834458892e-06, learning_rate=0.0015127649122759221, representation_name='acsf',
                 representation_params=acsf_params, tensorboard=True, store_frequency=10, hidden_layer_sizes=(300,67,), batch_size=73)

estimator.load_nn("saved_model")

estimator.set_properties(concat_ene_scaled)
estimator.generate_representation(concat_xyz, concat_zs, method='fortran')

pred_idx_methane = list(range(len(ene_methane)))
pred_methane = estimator.predict(pred_idx_methane)
print("Methane trajectory score: %s" % str(estimator.score(pred_idx_methane)))

pred_idx_squal = list(range(len(ene_methane), len(ene_methane)+len(ene_squal)))
pred_squal = estimator.predict(pred_idx_squal)
print("Squalane trajectory score: %s" % str(estimator.score(pred_idx_squal)))

np.savez("sorted_predictions.npz", pred_methane, concat_ene_scaled[pred_idx_methane], pred_squal, concat_ene_scaled[pred_idx_squal])
