import h5py
import numpy as np
from qml.qmlearn.preprocessing import AtomScaler
import pickle
import sys
import os

cwd = os.getcwd()
sys.path.append(cwd+"/../utils/")

import util_fn

# Getting all the data files
data_methane = h5py.File("../data_sets/methane_cn_dft.hdf5", "r")
data_ethane = h5py.File("../data_sets/ethane_cn_dft.hdf5", "r")
data_isobutane = h5py.File("../data_sets/isobutane_cn_dft.hdf5", "r")
data_isopentane = h5py.File("../data_sets/isopentane_cn_dft.hdf5", "r")
data_2isohex = h5py.File("../data_sets/2isohexane_cn_dft_pruned.hdf5", "r")
data_3isohex = h5py.File("../data_sets/3isohexane_cn_dft_pruned.hdf5", "r")
# data_dimer = h5py.File("../data_sets/isopentane_dimer_cn_dft.hdf5", "r")
data_squalane = h5py.File("../data_sets/squalane_cn_dft.hdf5", "r")

ref_ene = -133.1 * 2625.50

# Getting the energies and the nuclear charges of all the systems
ene_methane = np.array(data_methane.get("ene")) * 2625.50
ene_methane = ene_methane - ref_ene
zs_methane = np.array(data_methane.get("zs"), dtype=np.int32)
xyz_methane = np.array(data_methane.get("xyz"), dtype=np.int32)
forces_methane = np.array(data_methane.get("forces"), dtype=np.int32)
traj_idx_methane = np.array(data_methane.get("traj_idx"))
fn_methane = np.array(data_methane.get("Filenumber"))

# Sorting the trajectories of methane
traj_idx_methane, ene_methane, zs_methane, fn_methane, xyz_methane, forces_methane = util_fn.sort_traj(traj_idx_methane, ene_methane, zs_methane, fn_methane, xyz_methane, forces_methane)

# Getting the energies and the nuclear charges of all the systems
ene_ethane = np.array(data_ethane.get("ene")) * 2625.50
ene_ethane = ene_ethane - ref_ene
zs_ethane = np.array(data_ethane.get("zs"), dtype=np.int32)
xyz_ethane = np.array(data_ethane.get("xyz"), dtype=np.int32)
forces_ethane = np.array(data_ethane.get("forces"), dtype=np.int32)
traj_idx_ethane = np.array(data_ethane.get("traj_idx"))
fn_ethane = np.array(data_ethane.get("Filenumber"))

# Sorting the trajectories of ethane
traj_idx_ethane, ene_ethane, zs_ethane, fn_ethane, xyz_ethane, forces_ethane = util_fn.sort_traj(traj_idx_ethane, ene_ethane, zs_ethane, fn_ethane, xyz_ethane, forces_ethane)

# Getting the energies and the nuclear charges of all the systems
ene_isobutane = np.array(data_isobutane.get("ene")) * 2625.50
ene_isobutane = ene_isobutane - ref_ene
zs_isobutane = np.array(data_isobutane.get("zs"), dtype=np.int32)
xyz_isobutane = np.array(data_isobutane.get("xyz"), dtype=np.int32)
forces_isobutane = np.array(data_isobutane.get("forces"), dtype=np.int32)
traj_idx_isobutane = np.array(data_isobutane.get("traj_idx"))
fn_isobutane = np.array(data_isobutane.get("Filenumber"))

# Sorting the trajectories of isobutane
traj_idx_isobutane, ene_isobutane, zs_isobutane, fn_isobutane, xyz_isobutane, forces_isobutane = util_fn.sort_traj(traj_idx_isobutane, ene_isobutane, zs_isobutane, fn_isobutane, xyz_isobutane, forces_isobutane)

ene_isopent = np.array(data_isopentane.get("ene")) * 2625.50
ene_isopent = ene_isopent - ref_ene
zs_isopent = np.array(data_isopentane.get("zs"), dtype=np.int32)
xyz_isopent = np.array(data_isopentane.get("xyz"), dtype=np.int32)
forces_isopent = np.array(data_isopentane.get("forces"), dtype=np.int32)
traj_idx_isopent = np.array(data_isopentane.get("traj_idx"))
fn_isopent = np.array(data_isopentane.get("Filenumber"))

# Sorting the trajectories of isopentane
traj_idx_isopent, ene_isopent, zs_isopent, fn_isopent, xyz_isopent, forces_isopent = util_fn.sort_traj(traj_idx_isopent, ene_isopent, zs_isopent, fn_isopent, xyz_isopent, forces_isopent)


ene_2hex = np.array(data_2isohex.get("ene")) * 2625.50
ene_2hex = ene_2hex - ref_ene
zs_2hex = np.array(data_2isohex.get("zs"), dtype=np.int32)
xyz_2hex = np.array(data_2isohex.get("xyz"), dtype=np.int32)
forces_2hex = np.array(data_2isohex.get("forces"), dtype=np.int32)
traj_idx_2hex = np.array(data_2isohex.get("traj_idx"))
fn_2hex = np.array(data_2isohex.get("Filenumber"))

# Sorting the trajectories of 2-isohexane
traj_idx_2hex, ene_2hex, zs_2hex, fn_2hex, xyz_2hex, forces_2hex = util_fn.sort_traj(traj_idx_2hex, ene_2hex, zs_2hex, fn_2hex, xyz_2hex, forces_2hex)

ene_3hex = np.array(data_3isohex.get("ene")) * 2625.50
ene_3hex = ene_3hex - ref_ene
zs_3hex = np.array(data_3isohex.get("zs"), dtype=np.int32)
xyz_3hex = np.array(data_3isohex.get("xyz"), dtype=np.int32)
forces_3hex = np.array(data_3isohex.get("forces"), dtype=np.int32)
traj_idx_3hex = np.array(data_3isohex.get("traj_idx"))
fn_3hex = np.array(data_3isohex.get("Filenumber"))

# Sorting the trajectories of 3-isohexane
traj_idx_3hex, ene_3hex, zs_3hex, fn_3hex, xyz_3hex, forces_3hex = util_fn.sort_traj(traj_idx_3hex, ene_3hex, zs_3hex, fn_3hex, xyz_3hex, forces_3hex)


# ene_dimer = np.array(data_dimer.get("ene")) * 2625.50
# ene_dimer = ene_dimer - ref_ene
# zs_dimer = np.array(data_dimer.get("zs"), dtype=np.int32)
# xyz_dimer = np.array(data_dimer.get("xyz"), dtype=np.int32)
# forces_dimer = np.array(data_dimer.get("forces"), dtype=np.int32)
# traj_idx_dimer = np.array(data_dimer.get("traj_idx"))
# fn_dimer = np.array(data_dimer.get("Filenumber"))
#
# # Sorting the trajectories of isopentane dimer
# traj_idx_dimer, ene_dimer, zs_dimer, fn_dimer, xyz_dimer, forces_dimer = util_fn.sort_traj(traj_idx_dimer, ene_dimer, zs_dimer, fn_dimer, xyz_dimer, forces_dimer)

ene_squal = np.array(data_squalane.get("ene")) * 2625.50
ene_squal = ene_squal - ref_ene
zs_squal = np.array(data_squalane.get("zs"), dtype=np.int32)
xyz_squal = np.array(data_squalane.get("xyz"), dtype=np.int32)
forces_squal = np.array(data_squalane.get("forces"), dtype=np.int32)
traj_idx_squal = np.array(data_squalane.get("traj_idx"))
fn_squal = np.array(data_squalane.get("Filenumber"))

traj_idx_squal, ene_squal, zs_squal, fn_squal, xyz_squal, forces_squal = util_fn.sort_traj(traj_idx_squal, ene_squal, zs_squal, fn_squal, xyz_squal, forces_squal)

# Concatenating all the data
ene_for_scaler = np.concatenate((ene_methane[:100], ene_ethane[:100], ene_isobutane[:100], ene_isopent[:100], ene_2hex[:100], ene_3hex[:100], ene_squal[:100]))
zs_for_scaler = list(zs_methane[:100]) + list(zs_ethane[:100]) + list(zs_isobutane[:100]) + list(zs_isopent[:100]) + list(zs_2hex[:100]) + list(zs_3hex[:100]) + list(zs_squal[:100])

scaling = AtomScaler()
scaling.fit(zs_for_scaler, ene_for_scaler)

pickle.dump(scaling, open("larger_scaler.pickle", "wb"))