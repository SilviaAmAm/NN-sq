from qml.representations import generate_acsf
import numpy as np
import h5py
# import matplotlib.pyplot as plt
import time

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
data_isopentane = h5py.File("../data_sets/isopentane_cn_dft.hdf5", "r")
data_squal = h5py.File("../data_sets/squalane_cn_dft.hdf5", "r")
data_2isohex = h5py.File("../data_sets/2isohexane_cn_dft_pruned.hdf5")
data_3isohex = h5py.File("../data_sets/3isohexane_cn_dft_pruned.hdf5")


# Squalane data
idx = [0, -1]
xyz_squal = np.array(data_squal.get("xyz")) 
zs_squal = np.array(data_squal.get("zs"), dtype=np.int32) 

# 3-isohexane
idx_3isohex = np.asarray(data_3isohex.get('traj_idx'), dtype=int)
idx_3isohex_traj = np.where(idx_3isohex == 14)[0]

xyz_3isohex = np.array(data_3isohex.get("xyz"))[idx_3isohex_traj]
zs_3isohex = np.array(data_3isohex.get("zs"), dtype=np.int32)[idx_3isohex_traj]
xyz_3isohex = xyz_3isohex 
zs_3isohex = zs_3isohex 

# 2-isohexane
idx_2isohex = np.asarray(data_2isohex.get('traj_idx'), dtype=int)
idx_2isohex_traj = np.where(idx_2isohex == 12)[0]

xyz_2isohex = np.array(data_2isohex.get("xyz"))[idx_2isohex_traj]
zs_2isohex = np.array(data_2isohex.get("zs"), dtype=np.int32)[idx_2isohex_traj]
xyz_2isohex = xyz_2isohex 
zs_2isohex = zs_2isohex 

# Isopentane
idx_isopentane = np.asarray(data_isopentane.get('traj_idx'), dtype=int)
idx_isopentane_traj = np.where(idx_isopentane == 22)[0]

xyz_isopentane = np.array(data_isopentane.get("xyz"))[idx_isopentane_traj]
zs_isopentane = np.array(data_isopentane.get("zs"), dtype=np.int32)[idx_isopentane_traj]
xyz_isopentane = xyz_isopentane 
zs_isopentane = zs_isopentane 

# Methane
idx_methane = np.asarray(data_methane.get('traj_idx'), dtype=int)
idx_methane_traj = np.where(idx_methane == 14)[0]

xyz_methane = np.array(data_methane.get("xyz"))[idx_methane_traj]
zs_methane = np.array(data_methane.get("zs"), dtype=np.int32)[idx_methane_traj]
xyz_methane = xyz_methane 
zs_methane = zs_methane 

# Generating all the representations
n_basis = 14
r_min = 0.8
r_cut = 3.248470148281216
tau = 1.6110162523935854
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))

acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

acsf_methane = acsfy(xyz_methane, zs_methane, acsf_params)
acsf_isopentane = acsfy(xyz_isopentane, zs_isopentane, acsf_params)
acsf_2isohex = acsfy(xyz_2isohex, zs_2isohex, acsf_params)
acsf_3isohex = acsfy(xyz_3isohex, zs_3isohex, acsf_params)
acsf_squal = acsfy(xyz_squal, zs_squal, acsf_params)

print(acsf_methane.shape, acsf_isopentane.shape, acsf_2isohex.shape, acsf_squal.shape)

# Removing all the non-carbon atoms
acsf_methane_c = reshape_trim(acsf_methane, zs_methane)
acsf_isopentane_c = reshape_trim(acsf_isopentane, zs_isopentane)
acsf_2isohex_c = reshape_trim(acsf_2isohex, zs_2isohex)
acsf_3isohex_c = reshape_trim(acsf_3isohex, zs_3isohex)
acsf_squal_c = reshape_trim(acsf_squal, zs_squal)

print(acsf_methane_c.shape, acsf_isopentane_c.shape, acsf_2isohex_c.shape, acsf_squal_c.shape)

# Comparing the carbon atoms from squalane to methane
# diff_methane=[]
#
# start = time.time()
# for j in range(acsf_squal_c.shape[0]):
#     diff=[]
#
#     for i in range(acsf_methane_c.shape[0]):
#         diff.append(np.linalg.norm(acsf_methane_c[i]-acsf_squal_c[j]))
#
#     diff_methane.append(min(diff))
#
# end = time.time()
# print("It took %s s for methane" % str(end-start))
#
# np.savez("diff_methane.npz", diff_methane)

# Comparing the carbon atoms from squalane to isopentane
diff_isopentane=[]

start = time.time()
for j in range(acsf_squal_c.shape[0]):
    diff=[]

    for i in range(acsf_isopentane_c.shape[0]):
        diff.append(np.linalg.norm(acsf_isopentane_c[i]-acsf_squal_c[j]))

    diff_isopentane.append(min(diff))

end = time.time()
print("It took %s s for isopentane" % str(end-start))

np.savez("diff_isopentane.npz", diff_isopentane)

# Comparing the carbon atoms from squalane to 2isohex
diff_2isohex=[]

start = time.time()
for j in range(acsf_squal_c.shape[0]):
    diff=[]

    for i in range(acsf_2isohex_c.shape[0]):
        diff.append(np.linalg.norm(acsf_2isohex_c[i]-acsf_squal_c[j]))

    diff_2isohex.append(min(diff))

end = time.time()
print("It took %s s for 2-isohexane" % str(end-start))

np.savez("diff_2isohex.npz", diff_2isohex)

# Comparing the carbon atoms from squalane to 3isohex
diff_3isohex=[]

start = time.time()
for j in range(acsf_squal_c.shape[0]):
    diff=[]

    for i in range(acsf_3isohex_c.shape[0]):
        diff.append(np.linalg.norm(acsf_3isohex_c[i]-acsf_squal_c[j]))

    diff_3isohex.append(min(diff))

end = time.time()
print("It took %s s for 3-isohexane" % str(end-start))

np.savez("diff_3isohex.npz", diff_3isohex)

np.savez("differences.npz", diff_methane, diff_isopentane, diff_2isohex, diff_3isohex)

# part_fig_1, part_ax_1 = plt.subplots(figsize=(6,5))
# part_ax_1.plot(bins[1:], hist, label="First carbon")
# part_ax_1.set(xlabel="Euclidean distance", ylabel="Occurrences")
# part_ax_1.legend()
# part_fig_1.savefig("../images/isopentane_bins.png", dpi=200)
# plt.show()