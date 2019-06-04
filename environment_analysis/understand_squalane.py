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

    n_c = len(np.where(classes[0] == 6)[0])
    atom_idx = np.tile(np.where(classes[0] == 6)[0], acsf.shape[0])
    mol_idx = np.repeat(np.asarray(range(acsf.shape[0])), n_c)
    acsf = np.reshape(acsf, (acsf.shape[0]*acsf.shape[1], acsf.shape[-1]))
    classes = np.reshape(classes, (classes.shape[0] * classes.shape[1], ))
    
    C_idx = np.where(classes==6)[0]
    
    return acsf[C_idx], mol_idx, atom_idx

def write_vmd(xyz, zs, idx):

    dict = {1:"H", 6:"C", 7:"N", 2:"P"}

    f = open("short_traj.xyz", "w")

    for mol_idx,atom_indices in idx.items():

        f.write(str(len(zs[mol_idx])))
        f.write("\n\n")

        for j in range(len(zs[mol_idx])):
            if j in atom_indices:
                zs[mol_idx][j] = 2
            f.write(dict[zs[mol_idx][j]])
            f.write("\t")

            for k in range(3):
                f.write(str(xyz[mol_idx][j][k]))
                f.write("\t")
            f.write("\n")
    f.close()
    
# Getting the dataset
data_isopentane = h5py.File("../data_sets/isopentane_cn_dft.hdf5", "r")
data_squal = h5py.File("../data_sets/squalane_cn_dft.hdf5", "r")


# Squalane data
xyz_squal = np.array(data_squal.get("xyz")) 
zs_squal = np.array(data_squal.get("zs"), dtype=np.int32)

# Isopentane (secondary abstraction)
idx_isopentane = np.asarray(data_isopentane.get('traj_idx'), dtype=int)
idx_isopentane_traj = np.where(idx_isopentane == 1)[0]

xyz_isopentane = np.array(data_isopentane.get("xyz"))[idx_isopentane_traj]
zs_isopentane = np.array(data_isopentane.get("zs"), dtype=np.int32)[idx_isopentane_traj]

# Generating all the representations
n_basis = 14
r_min = 0.8
r_cut = 3.248470148281216
tau = 1.6110162523935854
eta = 4 * np.log(tau) * ((n_basis-1)/(r_cut - r_min))**2
zeta = - np.log(tau) / (2*np.log(np.cos(np.pi/(4*n_basis-4))))

acsf_params={"nRs2":n_basis, "nRs3":n_basis, "nTs":n_basis, "rcut":r_cut, "acut":r_cut, "zeta":zeta, "eta":eta}

acsf_isopentane = acsfy(xyz_isopentane, zs_isopentane, acsf_params)
acsf_squal = acsfy(xyz_squal, zs_squal, acsf_params)


# Removing all the non-carbon atoms
acsf_isopentane_c, _, _ = reshape_trim(acsf_isopentane, zs_isopentane)
acsf_squal_c, mol_idx, atom_idx = reshape_trim(acsf_squal, zs_squal)


# Comparing the carbon atoms from squalane to isopentane
bad_represented_c = {}

start = time.time()
for j in range(acsf_squal_c.shape[0]):
    diff_man=[]

    for i in range(acsf_isopentane_c.shape[0]):
        diff_man.append(np.sum(np.abs(acsf_isopentane_c[i] - acsf_squal_c[j])))

    min_d = min(diff_man)

    if min_d >= 6:
        print("Found a bad carbon!")
        if mol_idx[j] in bad_represented_c:
            bad_represented_c[mol_idx[j]].append(atom_idx[j])
        else:
            bad_represented_c[mol_idx[j]] = [atom_idx[j]]

end = time.time()
write_vmd(xyz_squal, zs_squal, bad_represented_c)
print("This took %s s." % str(end-start))

