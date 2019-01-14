import glob
import matplotlib.pyplot as plt
import numpy as np
import h5py
import seaborn as sns
sns.set()

def find_abstracted_h(xyz, h_idx, h_type):

    min_dist = 10
    idx_h_abstracted = -1

    for i, idx in enumerate(h_idx):

        dist_vec = xyz[idx] - xyz[20]
        dist = np.linalg.norm(dist_vec)

        if dist < min_dist:
            min_dist = dist
            idx_h_abstracted = i

    h_abstracted_type = h_type[idx_h_abstracted][1]

    return h_abstracted_type

def check_hnc(xyz, h_idx):

    min_dist = 10
    hnc = False

    for i, idx in enumerate(h_idx):
        dist_vec = xyz[idx] - xyz[21]
        dist = np.linalg.norm(dist_vec)
        if dist <= min_dist:
            min_dist = dist

    if min_dist <= 1.02:
        hnc = True

    return hnc

def check_dimer_dist(xyz):

    dist_vec = xyz[30] - xyz[0]
    dist = np.linalg.norm(dist_vec)

    if dist > 8.0:
        return True
    else:
        return False

path = "/Volumes/Transcend/data_sets/CN_double_isopentane/DoubleIsopentane_pm6"

filenames = glob.glob(path + "/*.xyz")

xyz = []
zs = []
energy = []
forces = []
trajectory_idx = []
snapshot_numbers = []

label_to_zs = {"H":1, "C":6, "N":7}

traj_counter = 0
snap_counter = 0

for file in filenames:
    f = open(file, 'r')
    counter = 0
    for line in f:
        if counter == 0:
            n_atoms = int(line)
            counter += 1
        else:
            index_1 = line.find("PotentialEnergy:")
            index_1 = index_1 + len("PotentialEnergy:")
            index_2 = line.find("kcal/mol")
            ene = float(line[index_1:index_2])

            nc = []
            coord = []
            force = []

            for i in range(n_atoms):
                line = next(f)
                strip_line = line.strip("\n")
                split_line = strip_line.split("\t")
                nc.append(label_to_zs[split_line[0]])
                coord.append([float(split_line[1]), float(split_line[2]), float(split_line[3])])
                force.append([float(split_line[4]), float(split_line[5]), float(split_line[6])])
            snap_counter += 1


            xyz.append(coord)
            zs.append(nc)
            energy.append(ene)
            forces.append(force)
            trajectory_idx.append(traj_counter)
            snapshot_numbers.append(snap_counter)

            counter = 0

    f.close()
    traj_counter += 1

# Check if the dimers are further apart than 5.0 A at the beginning of each trajectory
too_far = []
unique_traj = np.unique(trajectory_idx)
for tr in unique_traj:
    idx = np.where(trajectory_idx == tr)[0]

    xyz_first = np.asarray(xyz)[idx][0]
    too_far.append(check_dimer_dist(xyz_first))

print(too_far)

