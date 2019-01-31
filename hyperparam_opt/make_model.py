import pickle
import numpy as np
import h5py
from qml.aglaia.aglaia import ARMP


data_methane = h5py.File("/home/sa16246/repositories/NN-sq/data_sets/methane_cn_dft.hdf5", "r")
data_isopentane = h5py.File("/home/sa16246/repositories/NN-sq/data_sets/isopentane_cn_dft.hdf5", "r")
data_2isohex = h5py.File("/home/sa16246/repositories/NN-sq/data_sets/2isohexane_cn_dft_pruned.hdf5", "r")

n_samples = 500

xyz = np.array(data.get("xyz")[:n_samples])
ene = np.array(data.get("ene")[:n_samples])*4.184
ene = ene - ene[0]
zs = np.array(data["zs"][:n_samples], dtype = int)
forces = np.array(data.get("forces")[:n_samples])*4.184

acsf_params = representation_params={"nRs2": 10, "nRs3": 10, "nTs": 5}
estimator = ARMP_G(iterations=500, representation='acsf', representation_params=acsf_params, batch_size=250, hidden_layer_sizes=(50,30,10))

estimator.set_xyz(xyz)
estimator.set_classes(zs)
estimator.set_properties(ene)
estimator.set_gradients(forces)

estimator.generate_representation(method='fortran')

pickle.dump(estimator, open('model.pickle', 'wb'))

with open('idx.csv', 'w') as f:
    for i in range(n_samples):
        f.write('%s\n' % i)
