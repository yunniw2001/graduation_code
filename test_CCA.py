from mcca.cca import CCA
import logging
import numpy as np

# set log level
logging.root.setLevel(level=logging.INFO)

# create data in advance
a = np.random.rand(50, 50)
b = np.random.rand(50, 60)

# create instance of CCA
cca = CCA()
# calculate CCA
cca.fit(a, b)
# transform
cca.transform(a, b)
# transform by PCCA
cca.ptransform(a, b)
# save
root_path = '/home/ubuntu/graduation_project/'
cca.save_params(root_path+"save/cca.h5")
# load
cca.load_params(root_path+"save/cca.h5")
# plot
cca.plot_result()