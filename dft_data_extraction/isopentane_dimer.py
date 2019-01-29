import numpy as np
import glob
import h5py

def get_filename(filename):
    split_1 = filename.split("/")
    split_2 = split_1[-1].split("_")
    return int(split_2[0]), int(split_2[1])

def get_energy(file):

    file.seek(0)

    ene = 0

    for line in file:
        if "!UKS STATE  1.1 Energy" in line:
            line = line.strip()
            ene = float(line[len("!UKS STATE  1.1 Energy"):].strip())
            break

    return ene

def get_xyz(file):

    file.seek(0)

    zs_dict = {"C": 6, "H":1, "N":7}

    zs = []
    xyz = []
    for line in file:
        if "geometry={" in line:
            while True:
                line = next(file)
                if "}" in line:
                    break
                line = line.strip()
                line = line.strip(",")
                lineSplit = line.split(",")
                zs.append(zs_dict[lineSplit[0]])
                xyz_1 = []
                for j in range(1,4):
                    xyz_1.append(float(lineSplit[j]))
                xyz.append(xyz_1)
            break

    return np.asarray(xyz), np.asarray(zs)

def get_forces(file):

    file.seek(0)

    forces = []

    for line in file:
        if "dE/dx" in line:
            line = next(file)
            line = next(file)

            while True:
                line = line.strip()
                line = line.strip(",")
                lineSplit = line.split(" ")
                filtered_line = [item for item in lineSplit if item != '']
                forces_1 = []
                for i in range(1,4):
                    forces_1.append(float(filtered_line[i]))
                forces.append(forces_1)
                line = next(file)
                if len(line) <= 3:
                    break
            break

    return np.asarray(forces)

def extract_data(filename):
    f = open(filename, 'r')

    xyz = []
    zs =[]
    ene = 0
    forces = []
    filename_number = -1
    traj_idx = -1

    # Getting the filename
    filename_number, traj_idx = get_filename(filename)

    # Getting the energy
    ene = get_energy(f)

    # Getting Cartesian coordinates
    xyz, zs = get_xyz(f)

    # Getting forces
    forces = get_forces(f)

    return filename_number, traj_idx, xyz, zs, ene, forces

def check_data(f_n, traj_idx, xyz, zs, ene, forces):

    if ene == 0:
        return False

    if f_n == -1 or traj_idx == -1:
        return False

    if len(xyz) == 0 or len(forces) == 0 or len(zs) == 0:
        return False

    if zs.shape[0] != xyz.shape[0] or zs.shape[0] != forces.shape[0] or forces.shape[0] != xyz.shape[0]:
        return False

    if xyz.shape[-1] != 3 or forces.shape[-1] != 3:
        return False

    return True

# Load all the dft filenames
filenames_molpro = glob.glob("/Volumes/Transcend/mounted_dirs/*.out")

# From each file extract energy, forces, and cartesian coordinates
all_f_n = []
all_traj_idx = []
all_xyz = []
all_ene = []
all_zs = []
all_forces = []

for item in filenames_molpro:
    filename_number, traj_idx, xyz, zs, ene, forces = extract_data(item)

    is_correct = check_data(filename_number, traj_idx, xyz, zs, ene, forces)

    if is_correct:
        all_f_n.append(filename_number)
        all_traj_idx.append(traj_idx)
        all_xyz.append(xyz)
        all_zs.append(zs)
        all_ene.append(ene)
        all_forces.append(forces)
    else:
        print("File %s has errors" % (filename_number))

    if len(all_ene) % 100 == 0:
        print("Done %i files." % len(all_ene))

all_f_n = np.asarray(all_f_n)
all_traj_idx = np.asarray(all_traj_idx)
all_xyz = np.asarray(all_xyz)
all_ene = np.asarray(all_ene)
all_zs = np.asarray(all_zs)
all_forces = np.asarray(all_forces)

print("The shape of the xyz, zs, ene and forces is %s, %s, %s and %s." % (str(all_xyz.shape), str(all_zs.shape), str(all_ene.shape), str(all_forces.shape)) )

# Make a hdf5 dataset with filenames, cartesian coordinates, energy and forces
f = h5py.File("../data_sets/isopentane_dimer_cn_dft.hdf5", "w")

f.create_dataset("Filenumber", all_f_n.shape, data=all_f_n)
f.create_dataset("traj_idx", all_traj_idx.shape, data=all_traj_idx)
f.create_dataset("xyz", all_xyz.shape, data=all_xyz)
f.create_dataset("ene", all_ene.shape, data=all_ene)
f.create_dataset("zs", all_zs.shape, data=all_zs)
f.create_dataset("forces", all_forces.shape, data=all_forces)

f.close()