import matplotlib.pyplot as plt
import numpy as np
import h5py
import seaborn as sns
sns.set()
sns.set_style("white")

data = h5py.File("../data_sets/squalane_cn_dft.hdf5", "r")

# Extracting the data (If plotting the DFT surface, uncomment unit conversion!)
xyz = np.array(data.get("xyz"))
zs = np.array(data.get("zs"))
# ene = np.array(data.get("ene"))*2625.5 - ene_ref        # Converting the energy to kJ/mol and removing the reference energy
ene = np.array(data.get("ene"))
traj_idx = np.array(data.get("traj_idx"))
file_number = np.array(data.get("Filenumber"))

# Sorting the trajectories
idx_sorted = traj_idx.argsort()

ene = ene[idx_sorted]
traj_idx = traj_idx[idx_sorted]
file_number = file_number[idx_sorted]

n_traj = np.unique(traj_idx)

for item in n_traj:
    indices = np.where(traj_idx == item)

    idx_sorted = file_number[indices].argsort()

    ene[indices] = ene[indices][idx_sorted]
    traj_idx[indices] = traj_idx[indices][idx_sorted]
    file_number[indices] = file_number[indices][idx_sorted]


# Plotting
x = list(range(len(ene)))
fig, ax = plt.subplots(1, figsize=(8,6))
ax.scatter(x, ene, c=traj_idx)
ax.set_xlabel("Time step (0.0005 ps)")
ax.set_ylabel("Energy (Ha)")
# ax.plot([0.0, len(x)], [tot_ene_react, tot_ene_react], 'r--', linewidth=2, c="black")
# ax.plot([0.0, len(x)], [tot_ene_prod, tot_ene_prod], 'r--', linewidth=2, c="black")
plt.savefig("../images/squalane_dft.png", dpi=200)
plt.show()