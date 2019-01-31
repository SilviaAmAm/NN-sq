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

    dist_vec = xyz[17] - xyz[2]
    dist = np.linalg.norm(dist_vec)

    if dist > 6.0:
        return True
    else:
        return False

path = "/Volumes/Transcend/data_sets/CN_squalane/squalaneCN/"

filenames = [path + "trajectory_2019-01-31_11-56-01-AM.xyz"]

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


xyz = np.asarray(xyz)
zs = np.asarray(zs)
energy = np.asarray(energy)
forces = np.asarray(forces)
trajectory_idx = np.asarray(trajectory_idx)
snapshot_numbers = np.asarray(snapshot_numbers)

print(len(energy))

fig, ax = plt.subplots(1, figsize=(8,6))
ax.scatter(range(len(energy)), energy, c=trajectory_idx)
ax.set_xlabel("Time step (0.0005 ps)")
ax.set_ylabel("Energy (Kcal/mol)")
# plt.savefig("../images/squalane_traj.png", dpi=200)
# plt.show()
exit()


# Saving Pruned PM6 data
f = h5py.File("../data_sets/squalane_cn_pm6.hdf5", "w")

f.create_dataset("xyz", xyz.shape, data=xyz)
f.create_dataset("ene", energy.shape, data=energy)
f.create_dataset("zs", zs.shape, data=zs)
f.create_dataset("forces", forces.shape, data=forces)
f.create_dataset("traj_idx", trajectory_idx.shape, data=trajectory_idx)
f.create_dataset("Filenumber", snapshot_numbers.shape, data=snapshot_numbers)

f.close()
