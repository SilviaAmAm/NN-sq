import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# sns.set_context("talk")
sns.set_style("white")

import numpy as np

def add_subplot_axes(ax, rect):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

data = np.load("../training/methane/sorted_predictions.npz")

pred_ene_methane_traj = data["arr_0"]
true_ene_methane_traj = data["arr_1"]

pred_ene_squal = data["arr_2"]
true_ene_squal = data["arr_3"]

# Methane trajectory
# x = list(range(len(pred_ene_methane_traj)))
# fig, ax = plt.subplots(figsize=(10,10))
#
# subpos = [0.15,0.67,0.3,0.3] # Relative x,y and height, width
# ax_font = {'size':'19'}
# subax_font = {'size':'16'}
# ax.scatter(x, true_ene_methane_traj, label="True values", s=50)
# ax.scatter(x, pred_ene_methane_traj, label="Predictions", s=50)
# ax.set_xlabel("Frame Number", labelpad=7, **ax_font)
# ax.set_ylabel("Scaled energy (kJ/mol)", labelpad=5, **ax_font)
# ax.tick_params(axis='x', labelsize=24)
# ax.tick_params(axis='y', labelsize=24)
# ax.legend(prop={'size':15})
# subax1 = add_subplot_axes(ax, subpos)
# subax1.scatter(true_ene_methane_traj, pred_ene_methane_traj, s=10, c=sns.color_palette()[2])
# subax1.set_xlabel("True energy (kJ/mol)", labelpad=3, **subax_font)
# subax1.set_ylabel("Predicted energy (kJ/mol)", labelpad=2, **subax_font)
# subax1.tick_params(axis='x', labelsize=20)
# subax1.tick_params(axis='y', labelsize=20)
# plt.savefig("../images/methane_pred_1.png", dpi=200)
# plt.show()

# Squalane trajectory
x = list(range(len(pred_ene_squal)))
fig, ax = plt.subplots(figsize=(10,10))

subpos = [0.15,0.35,0.3,0.3] # Relative x,y and height, width
ax_font = {'size':'20'}
subax_font = {'size':'16'}
ax.scatter(x, true_ene_squal, label="True values", s=50)
ax.scatter(x, pred_ene_squal, label="Predictions", s=50)
ax.set_xlabel("Frame Number", labelpad=7, **ax_font)
ax.set_ylabel("Scaled energy (kJ/mol)", labelpad=5, **ax_font)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.legend(prop={'size':15})
subax1 = add_subplot_axes(ax, subpos)
subax1.scatter(true_ene_squal, pred_ene_squal, s=10, c=sns.color_palette()[2])
subax1.set_xlabel("True energy (kJ/mol)", labelpad=3, **subax_font)
subax1.set_ylabel("Predicted energy (kJ/mol)", labelpad=2, **subax_font)
subax1.tick_params(axis='x', labelsize=17)
subax1.tick_params(axis='y', labelsize=17)
plt.savefig("../images/squal_pred_1.png", dpi=200)
# plt.show()