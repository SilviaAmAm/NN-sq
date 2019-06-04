from qml.representations import generate_acsf
import numpy as np
import h5py
# import matplotlib.pyplot as plt
import time
import sys
sys.path.append("../utils")
import util_fn

def acsfy(xyz, classes, acsf_params):
    
    elements = np.unique(classes)
    elements_no_zero = np.ma.masked_equal(elements, 0).compressed()
    
    representation = []

    for i in range(xyz.shape[0]):

        g = generate_acsf(coordinates=xyz[i], elements=elements_no_zero, gradients=False, nuclear_charges=classes[i],
                          rcut=acsf_params['rcut'],
                          acut=acsf_params['acut'],
                          nRs2=acsf_params['nRs2'],
                          nRs3=acsf_params['nRs3'],
                          nTs=acsf_params['nTs'],
                          eta2=acsf_params['eta'],
                          eta3=acsf_params['eta'],
                          zeta=acsf_params['zeta'],
                          bin_min=0.8)

        # Hotfix t make sure the representation is single precision
        single_precision_g = g.astype(dtype=np.float32)
        del g

        representation.append(single_precision_g)

    return np.asarray(representation)

def reshape_trim(acsf, classes):
    
    acsf = np.reshape(acsf, (acsf.shape[0]*acsf.shape[1], acsf.shape[-1]))
    classes = np.reshape(classes, (classes.shape[0] * classes.shape[1], ))
    
    C_idx = np.where(classes==6)[0]
    
    return acsf[C_idx]
    
# Getting the dataset
data_methane = h5py.File("../data_sets/methane_cn_dft.hdf5", "r")
data_ethane = h5py.File("../data_sets/ethane_cn_dft.hdf5", "r")
data_isobutane = h5py.File("../data_sets/isobutane_cn_dft.hdf5", "r")
data_isopentane = h5py.File("../data_sets/isopentane_cn_dft.hdf5", "r")
data_2isohex = h5py.File("../data_sets/2isohexane_cn_dft_pruned.hdf5")
data_3isohex = h5py.File("../data_sets/3isohexane_cn_dft_pruned.hdf5")
data_squal = h5py.File("../data_sets/squalane_cn_dft.hdf5", "r")
# data_dimer = h5py.File("../data_sets/isopentane_dimer_cn_dft.hdf5")

# Squalane data
xyz_squal = np.array(data_squal.get("xyz")) 
zs_squal = np.array(data_squal.get("zs"), dtype=np.int32)

# Dimer (secondary abstraction)
# idx_dimer = np.asarray(data_dimer.get('traj_idx'), dtype=int)
# idx_dimer_traj = np.where(idx_dimer == 3)[0]
# 
# xyz_dimer = np.array(data_dimer.get("xyz"))[idx_dimer_traj]
# zs_dimer = np.array(data_dimer.get("zs"), dtype=np.int32)[idx_dimer_traj]

# 3-isohexane (secondary abstraction)
idx_3isohex = np.asarray(data_3isohex.get('traj_idx'), dtype=int)
idx_3isohex_traj = np.where(idx_3isohex == 13)[0]

xyz_3isohex = np.array(data_3isohex.get("xyz"))[idx_3isohex_traj]
zs_3isohex = np.array(data_3isohex.get("zs"), dtype=np.int32)[idx_3isohex_traj]

# 2-isohexane (secondary abstraction)
idx_2isohex = np.asarray(data_2isohex.get('traj_idx'), dtype=int)
idx_2isohex_traj = np.where(idx_2isohex == 2)[0]

xyz_2isohex = np.array(data_2isohex.get("xyz"))[idx_2isohex_traj]
zs_2isohex = np.array(data_2isohex.get("zs"), dtype=np.int32)[idx_2isohex_traj]

# Isopentane (secondary abstraction)
idx_isopentane = np.asarray(data_isopentane.get('traj_idx'), dtype=int)
idx_isopentane_traj = np.where(idx_isopentane == 1)[0]

xyz_isopentane = np.array(data_isopentane.get("xyz"))[idx_isopentane_traj]
zs_isopentane = np.array(data_isopentane.get("zs"), dtype=np.int32)[idx_isopentane_traj]

# Methane
idx_methane = np.asarray(data_methane.get('traj_idx'), dtype=int)
idx_methane_traj = np.where(idx_methane == 14)[0]

xyz_methane = np.array(data_methane.get("xyz"))[idx_methane_traj]
zs_methane = np.array(data_methane.get("zs"), dtype=np.int32)[idx_methane_traj]

# Ethane
idx_ethane = np.asarray(data_ethane.get('traj_idx'), dtype=int)
idx_ethane_traj = np.where(idx_ethane == 0)[0]

xyz_ethane = np.array(data_ethane.get("xyz"))[idx_ethane_traj]
zs_ethane = np.array(data_ethane.get("zs"), dtype=np.int32)[idx_ethane_traj]

# Ethane
idx_isobutane = np.asarray(data_isobutane.get('traj_idx'), dtype=int)
idx_isobutane_traj = np.where(idx_isobutane == 0)[0]

xyz_isobutane = np.array(data_isobutane.get("xyz"))[idx_isobutane_traj]
zs_isobutane = np.array(data_isobutane.get("zs"), dtype=np.int32)[idx_isobutane_traj]

# Generating all the representations
n_basis = 14
r_min = 0.8
r_cut = 3.248470148281216
tau = 1.6110162523935854
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))

acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

acsf_methane = acsfy(xyz_methane, zs_methane, acsf_params)
acsf_ethane = acsfy(xyz_ethane, zs_ethane, acsf_params)
acsf_isobutane = acsfy(xyz_isobutane, zs_isobutane, acsf_params)
acsf_isopentane = acsfy(xyz_isopentane, zs_isopentane, acsf_params)
acsf_2isohex = acsfy(xyz_2isohex, zs_2isohex, acsf_params)
acsf_3isohex = acsfy(xyz_3isohex, zs_3isohex, acsf_params)
# acsf_dimer = acsfy(xyz_dimer, zs_dimer, acsf_params)
acsf_squal = acsfy(xyz_squal, zs_squal, acsf_params)

print(acsf_methane.shape, acsf_isopentane.shape, acsf_2isohex.shape, acsf_squal.shape)

# Removing all the non-carbon atoms
acsf_methane_c = reshape_trim(acsf_methane, zs_methane)
acsf_ethane_c = reshape_trim(acsf_ethane, zs_ethane)
acsf_isobutane_c = reshape_trim(acsf_isobutane, zs_isobutane)
acsf_isopentane_c = reshape_trim(acsf_isopentane, zs_isopentane)
acsf_2isohex_c = reshape_trim(acsf_2isohex, zs_2isohex)
acsf_3isohex_c = reshape_trim(acsf_3isohex, zs_3isohex)
# acsf_dimer_c = reshape_trim(acsf_dimer, zs_dimer)
acsf_squal_c = reshape_trim(acsf_squal, zs_squal)

print(acsf_methane_c.shape, acsf_isopentane_c.shape, acsf_2isohex_c.shape, acsf_squal_c.shape)

# # Comparing the carbon atoms from squalane to methane
diff_euc_methane=[]
diff_man_methane=[]

start = time.time()
for j in range(acsf_squal_c.shape[0]):
    diff_euc=[]
    diff_man=[]

    for i in range(acsf_methane_c.shape[0]):
        diff_euc.append(np.linalg.norm(acsf_methane_c[i] - acsf_squal_c[j]))
        diff_man.append(np.sum(np.abs(acsf_methane_c[i] - acsf_squal_c[j])))

    diff_euc_methane.append(min(diff_euc))
    diff_man_methane.append(min(diff_man))

end = time.time()
print("It took %s s for methane" % str(end-start))

np.savez("diff_methane.npz", diff_euc_methane, diff_man_methane)

# # Comparing the carbon atoms from squalane to ethane
diff_euc_ethane=[]
diff_man_ethane=[]

start = time.time()
for j in range(acsf_squal_c.shape[0]):
    diff_euc=[]
    diff_man=[]

    for i in range(acsf_ethane_c.shape[0]):
        diff_euc.append(np.linalg.norm(acsf_ethane_c[i] - acsf_squal_c[j]))
        diff_man.append(np.sum(np.abs(acsf_ethane_c[i] - acsf_squal_c[j])))

    diff_euc_ethane.append(min(diff_euc))
    diff_man_ethane.append(min(diff_man))

end = time.time()
print("It took %s s for ethane" % str(end-start))

np.savez("diff_ethane.npz", diff_euc_ethane, diff_man_ethane)

# # Comparing the carbon atoms from squalane to isobutane
diff_euc_isobutane=[]
diff_man_isobutane=[]

start = time.time()
for j in range(acsf_squal_c.shape[0]):
    diff_euc=[]
    diff_man=[]

    for i in range(acsf_isobutane_c.shape[0]):
        diff_euc.append(np.linalg.norm(acsf_isobutane_c[i] - acsf_squal_c[j]))
        diff_man.append(np.sum(np.abs(acsf_isobutane_c[i] - acsf_squal_c[j])))

    diff_euc_isobutane.append(min(diff_euc))
    diff_man_isobutane.append(min(diff_man))

end = time.time()
print("It took %s s for isobutane" % str(end-start))

np.savez("diff_isobutane.npz", diff_euc_isobutane, diff_man_isobutane)

# Comparing the carbon atoms from squalane to isopentane
diff_euc_isopentane=[]
diff_man_isopentane=[]

start = time.time()
for j in range(acsf_squal_c.shape[0]):
    diff_euc=[]
    diff_man=[]

    for i in range(acsf_isopentane_c.shape[0]):
        diff_euc.append(np.linalg.norm(acsf_isopentane_c[i] - acsf_squal_c[j]))
        diff_man.append(np.sum(np.abs(acsf_isopentane_c[i] - acsf_squal_c[j])))

    diff_euc_isopentane.append(min(diff_euc))
    diff_man_isopentane.append(min(diff_man))

end = time.time()
print("It took %s s for isopentane" % str(end-start))

np.savez("diff_isopentane_2nd.npz", diff_euc_isopentane, diff_man_isopentane)

# Comparing the carbon atoms from squalane to 2isohex
diff_euc_2isohex=[]
diff_man_2isohex=[]

start = time.time()
for j in range(acsf_squal_c.shape[0]):
    diff_euc=[]
    diff_man=[]

    for i in range(acsf_2isohex_c.shape[0]):
        diff_euc.append(np.linalg.norm(acsf_2isohex_c[i] - acsf_squal_c[j]))
        diff_man.append(np.sum(np.abs(acsf_2isohex_c[i] - acsf_squal_c[j])))

    diff_euc_2isohex.append(min(diff_euc))
    diff_man_2isohex.append(min(diff_man))

end = time.time()
print("It took %s s for 2-isohexane" % str(end-start))

np.savez("diff_2isohex_2nd.npz", diff_euc_2isohex, diff_man_2isohex)

# Comparing the carbon atoms from squalane to 3isohex
diff_euc_3isohex=[]
diff_man_3isohex=[]

start = time.time()
for j in range(acsf_squal_c.shape[0]):
    diff_euc=[]
    diff_man=[]

    for i in range(acsf_3isohex_c.shape[0]):
        diff_euc.append(np.linalg.norm(acsf_3isohex_c[i] - acsf_squal_c[j]))
        diff_man.append(np.sum(np.abs(acsf_3isohex_c[i] - acsf_squal_c[j])))

    diff_euc_3isohex.append(min(diff_euc))
    diff_man_3isohex.append(min(diff_man))

end = time.time()
print("It took %s s for 3-isohexane" % str(end-start))

np.savez("diff_3isohex_2nd.npz", diff_euc_3isohex, diff_man_3isohex)

# # Comparing the carbon atoms from squalane to dimer
# diff_euc_dimer=[]
# diff_man_dimer=[]
#
# start = time.time()
# for j in range(acsf_squal_c.shape[0]):
#     diff_euc=[]
#     diff_man=[]
#
#     for i in range(acsf_dimer_c.shape[0]):
#         diff_euc.append(np.linalg.norm(acsf_dimer_c[i] - acsf_squal_c[j]))
#         diff_man.append(np.sum(np.abs(acsf_dimer_c[i] - acsf_squal_c[j])))
#
#     diff_euc_dimer.append(min(diff_euc))
#     diff_man_dimer.append(min(diff_man))
#
# end = time.time()
# print("It took %s s for the dimer" % str(end-start))
#
# np.savez("diff_dimer_2nd.npz", diff_euc_dimer, diff_man_dimer)

# part_fig_1, part_ax_1 = plt.subplots(figsize=(6,5))
# part_ax_1.plot(bins[1:], hist, label="First carbon")
# part_ax_1.set(xlabel="Euclidean distance", ylabel="Occurrences")
# part_ax_1.legend()
# part_fig_1.savefig("../images/isopentane_bins.png", dpi=200)
# plt.show()
