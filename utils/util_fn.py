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