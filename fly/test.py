"""
for testing the functionality of KDTree and dictionary
"""

import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage.filters import maximum_filter
from skimage.feature import peak_local_max

def test_fct(a):
    a += 2
    print a

a =np.zeros((7,7), dtype=np.float32)
a[3,3] = 7.
a[4,4] = 4.
out = np.zeros((7,7))
coord = peak_local_max(a, min_distance=2)
print coord
#maximum_filter(a, size=3,output=out)
#plt.imshow(out)
#plt.show()
"""
f = h5py.File('maxima.h5','r')
#shape = f['data'].shape
data = f['data'][100:120,100:120,2]
x,y = np.nonzero(data)
print x,y
f.close()
"""

"""
Y = np.array([[2,0,0],[3,0,0]])
tree_y = KDTree(Y, leaf_size=2)
#X = np.random.random((10,3))
#tree = KDTree(X, leaf_size=2)
#print X[0].shape
#print np.array([0., 0., 0.]).shape
dist,ind = tree_y.query(np.array([[0., 0., 0.],[4. , 0. ,0]]), k=1)
print ind
my_dict = {"house": "villa", "car":"porsche", "favourite number":4.}
your_dict = {"house": "tent"}
nested_dict = {"my_dict": my_dict, "your_dict": your_dict}
print nested_dict["my_dict"]["favourite number"]
"""
