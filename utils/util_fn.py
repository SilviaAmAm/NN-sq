import numpy as np

def sort_traj(traj_idx, ene, zs, file_n, xyz, forces):
    idx_sorted = traj_idx.argsort()

    ene = ene[idx_sorted]
    zs = zs[idx_sorted]
    traj_idx = traj_idx[idx_sorted]
    file_n = file_n[idx_sorted]
    xyz = xyz[idx_sorted]
    forces = forces[idx_sorted]
    n_traj = np.unique(traj_idx)

    for item in n_traj:
        indices = np.where(traj_idx == item)

        idx_sorted = file_n[indices].argsort()

        zs[indices] = zs[indices][idx_sorted]
        ene[indices] = ene[indices][idx_sorted]
        traj_idx[indices] = traj_idx[indices][idx_sorted]
        file_n[indices] = file_n[indices][idx_sorted]
        forces[indices] = forces[indices][idx_sorted]
        xyz[indices] = xyz[indices][idx_sorted]

    return traj_idx, ene, zs, file_n, xyz, forces

def write_vmd(xyz, zs):

    dict = {1:"H", 6:"C", 7:"N"}

    f = open("traj.xyz", "w")

    for i in range(zs.shape[0]):
        f.write(str(len(zs[i])))
        f.write("\n\n")

        for j in range(len(zs[i])):
            f.write(dict[zs[i][j]])
            f.write("\t")

            for k in range(3):
                f.write(str(xyz[i][j][k]))
                f.write("\t")
            f.write("\n")

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

def check_hnc(xyz, h_idx, n_idx):

    min_dist = 10
    hnc = False

    for i, idx in enumerate(h_idx):
        dist_vec = xyz[idx] - xyz[n_idx]
        dist = np.linalg.norm(dist_vec)
        if dist <= min_dist:
            min_dist = dist

    if min_dist <= 1.02:
        hnc = True

    return hnc