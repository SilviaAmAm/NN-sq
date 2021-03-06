import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()
sns.set_style("white")
sns.set_context("poster")

data_methane = np.load("../environment_analysis/diff_methane.npz")
data_ethane = np.load("../environment_analysis/diff_ethane.npz")
data_isobutane = np.load("../environment_analysis/diff_isobutane.npz")
data_isopentane = np.load("../environment_analysis/diff_isopentane_2nd.npz")
data_2isohex = np.load("../environment_analysis/diff_2isohex_2nd.npz")
data_3isohex = np.load("../environment_analysis/diff_3isohex_2nd.npz")
# data_dimer = np.load("../environment_analysis/diff_dimer_2nd.npz")
# data_total = np.load("../environment_analysis/diff_total_2nd.npz")


diff_euc_methane = data_methane["arr_0"]
diff_euc_ethane = data_ethane["arr_0"]
diff_euc_isobutane = data_isobutane["arr_0"]
diff_euc_isopentane = data_isopentane["arr_0"]
diff_euc_2isohex = data_2isohex["arr_0"]
diff_euc_3isohex = data_3isohex["arr_0"]
# diff_euc_dimer = data_dimer["arr_0"]
# diff_euc_total = data_total["arr_0"]

bins_euc = np.arange(0, max(diff_euc_methane), 0.05)

hist_euc_methane, bin_euc_methane = np.histogram(diff_euc_methane, bins=bins_euc, density=False)
hist_euc_ethane, bin_euc_ethane = np.histogram(diff_euc_ethane, bins=bins_euc, density=False)
hist_euc_isobutane, bin_euc_isobutane = np.histogram(diff_euc_isobutane, bins=bins_euc, density=False)
hist_euc_isopentane, bin_euc_isopentane = np.histogram(diff_euc_isopentane, bins=bins_euc, density=False)
hist_euc_2isohex, bin_euc_2isohex = np.histogram(diff_euc_2isohex, bins=bins_euc, density=False)
hist_euc_3isohex, bin_euc_3isohex = np.histogram(diff_euc_3isohex, bins=bins_euc, density=False)
# hist_euc_dimer, bin_euc_dimer = np.histogram(diff_euc_dimer, bins=bins_euc, density=False)
# hist_euc_total, bin_euc_total = np.histogram(diff_euc_total, bins=bins_euc, density=False)

part_fig_1, part_ax_1 = plt.subplots(figsize=(11,10))
# part_ax_1.plot(bins_euc[1:], hist_euc_methane, label="Methane")
# part_ax_1.plot(bins_euc[1:], hist_euc_ethane, label="Ethane")
# part_ax_1.plot(bins_euc[1:], hist_euc_isobutane, label="Isobutane")
part_ax_1.plot(bins_euc[1:], hist_euc_isopentane, label="Isopentane")
part_ax_1.plot(bins_euc[1:], hist_euc_2isohex, label="2-Isohexane")
part_ax_1.plot(bins_euc[1:], hist_euc_3isohex, label="3-Isohexane")
# part_ax_1.plot(bins_euc[1:], hist_euc_dimer, label="Dimer")
# part_ax_1.plot(bins_euc[1:], hist_euc_total, label="Total")
part_ax_1.set(xlabel="Euclidean distance", ylabel="Occurrences")
part_ax_1.legend()
part_fig_1.savefig("/Volumes/Transcend/repositories/thesis/ffnn_results_fig/acsf_euc_distances_big.png", dpi=200)
plt.show()


diff_man_methane = data_methane["arr_1"]
diff_man_ethane = data_ethane["arr_1"]
diff_man_isobutane = data_isobutane["arr_1"]
diff_man_isopentane = data_isopentane["arr_1"]
diff_man_2isohex = data_2isohex["arr_1"]
diff_man_3isohex = data_3isohex["arr_1"]
# diff_man_dimer = data_dimer["arr_1"]
# diff_man_total = data_total["arr_1"]

bins_man = np.arange(0, max(diff_man_methane), 0.5)

hist_man_methane, bin_man_methane = np.histogram(diff_man_methane, bins=bins_man, density=False)
hist_man_ethane, bin_man_ethane = np.histogram(diff_man_ethane, bins=bins_man, density=False)
hist_man_isobutane, bin_man_isobutane = np.histogram(diff_man_isobutane, bins=bins_man, density=False)
hist_man_isopentane, bin_man_isopentane = np.histogram(diff_man_isopentane, bins=bins_man, density=False)
hist_man_2isohex, bin_man_2isohex = np.histogram(diff_man_2isohex, bins=bins_man, density=False)
hist_man_3isohex, bin_man_3isohex = np.histogram(diff_man_3isohex, bins=bins_man, density=False)
# hist_man_dimer, bin_man_dimer = np.histogram(diff_man_dimer, bins=bins_man, density=False)
# hist_man_total, bin_man_total = np.histogram(diff_man_total, bins=bins_man, density=False)

part_fig_2, part_ax_2 = plt.subplots(figsize=(11, 10))
# part_ax_2.plot(bins_man[1:], hist_man_methane, label="Methane")
# part_ax_2.plot(bins_man[1:], hist_man_ethane, label="Ethane")
# part_ax_2.plot(bins_man[1:], hist_man_isobutane, label="Isobutane")
part_ax_2.plot(bins_man[1:], hist_man_isopentane, label="Isopentane")
part_ax_2.plot(bins_man[1:], hist_man_2isohex, label="2-Isohexane")
part_ax_2.plot(bins_man[1:], hist_man_3isohex, label="3-Isohexane")
# part_ax_2.plot(bins_man[1:], hist_man_dimer, label="Dimer")
# part_ax_2.plot(bins_man[1:], hist_man_total, label="Total")
part_ax_2.set(xlabel="Manhattan distance", ylabel="Occurrences")
part_ax_2.legend()
part_fig_2.savefig("/Volumes/Transcend/repositories/thesis/ffnn_results_fig/acsf_man_distances_big.png", dpi=200)
plt.show()