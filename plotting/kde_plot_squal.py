import numpy as np
import bisect
from random import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("white")

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

def print_how_many_pst(h_id, traj_idx):
    """
    Counts how many trajectories are primary, secondary and tertiary abstractions.
    """
    # Where the hydrogens are primary, secondary or tertiary
    identity = [None, None, 1, 1, 1, 2, None, 2, None, 3, None, 1, 1, 1, 1, 1, 1, None, None]

    # Figuring out where each traj starts
    traj, idx_unique = np.unique(traj_idx, return_index=True)
    h_id_per_traj = h_id[idx_unique]

    primary = 0
    secondary = 0
    tertiary = 0

    for id in h_id_per_traj:
        if identity[id] == 1:
            primary += 1
        elif identity[id] == 2:
            secondary += 1
        elif identity[id] == 3:
            tertiary += 1

    print("There are %i, %i and %i primary, secondary and tertiary abstractions in %i trajectories" % (primary, secondary, tertiary, len(traj)) )

# Data VR
data = h5py.File("../data_sets/squalane_cn_dft.hdf5", "r")

traj_idx = np.array(data.get("traj_idx"))
xyz = np.array(data.get("xyz"))
zs = np.array(data.get("zs"))
file_number = np.array(data.get("Filenumber"))

# Sorting the trajectories
idx_sorted = file_number.argsort()

traj_idx = traj_idx[idx_sorted]
file_number = file_number[idx_sorted]
xyz = xyz[idx_sorted]
zs = zs[idx_sorted]

# Finding out which H is abstracted
h_id, c_id = get_h_c_abstracted(traj_idx, xyz, zs)
ch_dist_alk_vr, ch_dist_cn_vr = get_distances(xyz, h_id, c_id)

# f = open("vmdtraj.xyz", "w")
# for frame in range(traj_idx.shape[0]):
#     if ch_dist_cn_vr[frame] >= 2.0 and ch_dist_alk_vr[frame] >= 1.6:
#
#         idx_to_char = {1: "H", 6:"C", 7:"N"}
#
#         f.write(str(len(zs[frame])))
#         f.write("\n\n")
#
#         for n in range(len(zs[frame])):
#             f.write(str(idx_to_char[zs[frame][n]]))
#             f.write("\t")
#             for i in range(3):
#                 f.write(str(xyz[frame][n][i]))
#                 f.write("\t")
#             f.write("\n")
# f.close()
# exit()

# print_how_many_pst(h_id, traj_idx)

# plt.scatter(range(len(ch_dist_alk_vr)), ch_dist_alk_vr)
# plt.show()
# exit()

g = sns.jointplot(ch_dist_alk_vr, ch_dist_cn_vr, kind="kde",  height=7, space=0, xlim=(0.5, 5.0), ylim=(0.5, 5.0))
g.set_axis_labels("D2 (Å)", "D1 (Å)")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig("../images/2d_kde_squalane.png", dpi=200)
plt.show()