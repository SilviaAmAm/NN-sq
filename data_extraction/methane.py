import glob
import matplotlib.pyplot as plt
import numpy as np
import h5py
import seaborn as sns
sns.set()

def check_hnc(xyz, h_idx):

    min_dist = 10
    hnc = False

    for i, idx in enumerate(h_idx):
        dist_vec = xyz[idx] - xyz[-1]
        dist = np.linalg.norm(dist_vec)
        if dist <= min_dist:
            min_dist = dist

    if min_dist <= 1.02:
        hnc = True

    return hnc

path = "/Volumes/Transcend/data_sets/CN_methane_pm6/MethaneCN_pm6"

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

# Gathering the indices of the Hydrogens
h_idx = np.where(np.array(zs[0]) == 1)[0]

ene_reactants = []
ene_prods = []
hnc_abstraction = []

# Choosing the half-way point energy
unique_traj = np.unique(trajectory_idx)
for tr in unique_traj:
    idx = np.where(trajectory_idx == tr)[0]
    ene_traj = np.asarray(energy)[idx]
    xyz_last = np.asarray(xyz)[idx][-1]

    is_hnc = check_hnc(xyz_last, h_idx)
    hnc_abstraction.append(is_hnc)
    if is_hnc:
        continue

    ene_reactants.append(ene_traj[:400])
    ene_prods.append(ene_traj[-400:])

tot_ene_react = np.mean(ene_reactants)
tot_ene_prod = np.mean(ene_prods)

mid_point = 0.5*(tot_ene_react + tot_ene_prod)


# Plotting
x = range(len(xyz))
print("There are %i samples." % x[-1])

# fig, ax = plt.subplots(1, figsize=(8,6))
# ax.scatter(x, energy, c=trajectory_idx)
# ax.set_xlabel("Time step (0.0005 ps)")
# ax.set_ylabel("Energy (Kcal/mol)")
# ax.plot([0.0, len(x)], [tot_ene_react, tot_ene_react], 'r--', linewidth=2, c="black")
# ax.plot([0.0, len(x)], [tot_ene_prod, tot_ene_prod], 'r--', linewidth=2, c="black")
# # plt.savefig("raw_shortlist.png", dpi=200)
# plt.show()

# Pruning
xyz_p1 = []
zs_p1 = []
energy_p1 = []
forces_p1 = []
trajectory_idx_p1 = []
snapshot_number_p1 = []

unique_traj = np.unique(trajectory_idx)
for n,tr in enumerate(unique_traj):
    idx = np.where(trajectory_idx == tr)[0]
    ene_traj = np.asarray(energy)[idx]

    if hnc_abstraction[n]:
        continue

    idx_mid = np.where(ene_traj <= mid_point)[0][0]

    xyz_p1.append(np.asarray(xyz)[idx][idx_mid-600:idx_mid+600])
    zs_p1.append(np.asarray(zs)[idx][idx_mid-600:idx_mid+600])
    energy_p1.append(np.asarray(energy)[idx][idx_mid-600:idx_mid+600])
    forces_p1.append(np.asarray(forces)[idx][idx_mid-600:idx_mid+600])
    trajectory_idx_p1.append(np.asarray(trajectory_idx)[idx][idx_mid-600:idx_mid+600])
    snapshot_number_p1.append(np.asarray(snapshot_numbers)[idx][idx_mid - 600:idx_mid + 600])

# joining the arrays
xyz_p1 = np.concatenate(xyz_p1)
zs_p1 = np.concatenate(zs_p1)
energy_p1 = np.concatenate(energy_p1)
forces_p1 = np.concatenate(forces_p1)
trajectory_idx_p1 = np.concatenate(trajectory_idx_p1)
snapshot_number_p1 = np.concatenate(snapshot_number_p1)


# Plotting
x_p1 = range(len(xyz_p1))
print("There are %i samples after the 1st round of pruning." % x_p1[-1])

fig, ax = plt.subplots(1, figsize=(8,6))
ax.scatter(x_p1, energy_p1, c=trajectory_idx_p1)
ax.set_xlabel("Time step (0.0005 ps)")
ax.set_ylabel("Energy (Kcal/mol)")
ax.plot([0.0, len(x_p1)], [tot_ene_react, tot_ene_react], 'r--', linewidth=2, c="black")
ax.plot([0.0, len(x_p1)], [tot_ene_prod, tot_ene_prod], 'r--', linewidth=2, c="black")
plt.savefig("../images/methane_vr_pruned.png", dpi=200)
plt.show()

# Saving Pruned PM6 data
# f = h5py.File("../data_sets/methane_cn_pm6_pruned.hdf5", "w")
#
# f.create_dataset("xyz", xyz_p1.shape, data=xyz_p1)
# f.create_dataset("ene", energy_p1.shape, data=energy_p1)
# f.create_dataset("zs", zs_p1.shape, data=zs_p1)
# f.create_dataset("forces", forces_p1.shape, data=forces_p1)
# f.create_dataset("traj_idx", trajectory_idx_p1.shape, data=trajectory_idx_p1)
# f.create_dataset("Filenumber", snapshot_number_p1.shape, data=snapshot_number_p1)
#
# f.close()
