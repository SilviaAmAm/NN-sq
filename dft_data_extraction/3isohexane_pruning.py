import numpy as np
import h5py

def get_h_c_abstracted(traj_id, xyz, zs):

    h_id = np.zeros(traj_id.shape, dtype=np.int32)
    c_id = np.zeros(traj_id.shape, dtype=np.int32)

    unique_traj_idx = np.unique(traj_id)

    for id in unique_traj_idx:
        # Frames that belong to a particular trajectory
        idx_frames = np.where(traj_id==id)[0]

        # Coordinates of those frames
        xyz_frames = xyz[idx_frames]
        zs_frames = zs[idx_frames]

        # f = open("vmdtraj.xyz", "w")
        # idx_to_char = {1: "H", 6:"C", 7:"N"}
        # for traj_frame in range(len(zs_frames)):
        #     f.write(str(len(zs_frames[traj_frame])))
        #     f.write("\n\n")
        #
        #     for n in range(len(zs_frames[traj_frame])):
        #         f.write(str(idx_to_char[zs_frames[traj_frame][n]]))
        #         f.write("\t")
        #         for i in range(3):
        #             f.write(str(xyz_frames[traj_frame][n][i]))
        #             f.write("\t")
        #         f.write("\n")
        # f.close()
        # exit()

        # Coordinates of the last frame of a trajectory
        xyz_last_frame = xyz_frames[-1]
        zs_last_frame = zs_frames[-1]

        # Coordinates of the first frame of a trajectory
        xyz_first_frame = xyz_frames[0]
        zs_first_frame = zs_frames[0]

        # The H being abstracted will be the one with the shortest distance to the CN carbon
        idx_of_all_h = np.where(zs_last_frame == 1)[0]
        min_dist = 10
        abstracted_h = -1       # Index of the abstracted hydrogen
        for h in idx_of_all_h:
            dist = np.linalg.norm((xyz_last_frame[h]- xyz_last_frame[-2]))    # distance between a H and the CN carbon
            if dist < min_dist:
                min_dist = dist
                abstracted_h = int(h)

        h_id[idx_frames] = abstracted_h

        # The C initially bonded to the abstracted H will be the one with the shortest distance to it in the first frame
        idx_of_all_c = np.where(zs_first_frame == 6)[0]
        min_dist = 10
        bonded_c = -1  # Index of the C bonded to the hydrogen that gets abstracted
        for c in idx_of_all_c:
            dist = np.linalg.norm((xyz_first_frame[c]-xyz_first_frame[abstracted_h]))  # distance between a H and the CN carbon
            if dist < min_dist:
                min_dist = dist
                bonded_c = int(c)

        c_id[idx_frames] = bonded_c

    return h_id, c_id

def get_distances(xyz, h_id, c_id):
    ch_dist_alk = np.zeros(h_id.shape)
    ch_dist_cn = np.zeros(h_id.shape)

    # For each frame, calculate the distance between the h and the c in question
    for i in range(h_id.shape[0]):
        ch_dist_cn[i] = np.linalg.norm((xyz[i][h_id[i]]- xyz[i][-2]))
        ch_dist_alk[i] = np.linalg.norm((xyz[i][h_id[i]]- xyz[i][c_id[i]]))

    return ch_dist_alk, ch_dist_cn

data = h5py.File("../data_sets/3isohexane_cn_dft.hdf5", "r")

all_traj_idx = np.array(data.get("traj_idx"))
all_xyz = np.array(data.get("xyz"))
all_ene = np.array(data.get("ene"))
all_forces = np.array(data.get("forces"))
all_zs = np.array(data.get("zs"))
all_f_n = np.array(data.get("Filenumber"))

# Pruning HNC which I don't understand why it's here since I already pruned it...
# Sorting the trajectories
idx_sorted = all_traj_idx.argsort()

all_traj_idx = all_traj_idx[idx_sorted]
all_f_n = all_f_n[idx_sorted]
all_xyz = all_xyz[idx_sorted]
all_zs = all_zs[idx_sorted]
all_ene = all_ene[idx_sorted]
all_forces = all_forces[idx_sorted]

n_traj = np.unique(all_traj_idx)

for item in n_traj:
    indices = np.where(all_traj_idx == item)

    idx_sorted = all_f_n[indices].argsort()
    all_traj_idx[indices] = all_traj_idx[indices][idx_sorted]
    all_f_n[indices] = all_f_n[indices][idx_sorted]
    all_xyz[indices] = all_xyz[indices][idx_sorted]
    all_zs[indices] = all_zs[indices][idx_sorted]
    all_ene[indices] = all_ene[indices][idx_sorted]
    all_forces[indices] = all_forces[indices][idx_sorted]

# Finding out which H is abstracted
h_id, c_id = get_h_c_abstracted(all_traj_idx, all_xyz, all_zs)
ch_dist_alk, ch_dist_cn = get_distances(all_xyz, h_id, c_id)

idx_hnc = []
for frame in range(all_traj_idx.shape[0]):
    if ch_dist_cn[frame] >= 2.0 and ch_dist_alk[frame] >= 1.6:
        idx_hnc.append(all_traj_idx[frame])

# Find which is the trajectory to remove
traj_to_remove = np.unique(idx_hnc)
idx_traj_to_remove = np.where(all_traj_idx == traj_to_remove)[0]

all_traj_idx = np.delete(all_traj_idx, idx_traj_to_remove, axis=0)
all_f_n = np.delete(all_f_n, idx_traj_to_remove, axis=0)
all_xyz = np.delete(all_xyz, idx_traj_to_remove, axis=0)
all_zs = np.delete(all_zs, idx_traj_to_remove, axis=0)
all_ene = np.delete(all_ene, idx_traj_to_remove, axis=0)
all_forces = np.delete(all_forces, idx_traj_to_remove, axis=0)

print("The shape of the xyz, zs, ene and forces is %s, %s, %s and %s." % (str(all_xyz.shape), str(all_zs.shape), str(all_ene.shape), str(all_forces.shape)) )

# Make a hdf5 dataset with filenames, cartesian coordinates, energy and forces
f = h5py.File("../data_sets/3isohexane_cn_dft_pruned.hdf5", "w")

f.create_dataset("Filenumber", all_f_n.shape, data=all_f_n)
f.create_dataset("traj_idx", all_traj_idx.shape, data=all_traj_idx)
f.create_dataset("xyz", all_xyz.shape, data=all_xyz)
f.create_dataset("ene", all_ene.shape, data=all_ene)
f.create_dataset("zs", all_zs.shape, data=all_zs)
f.create_dataset("forces", all_forces.shape, data=all_forces)

f.close()