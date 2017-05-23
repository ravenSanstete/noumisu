# this implements the so-called projection net, which is used to approximate manifold in an implicit way

import numpy as np
import tensorflow as tf
import s_2_data as feeder


A = np.random.randn(3,3); # construct a random matrix and then compute xA(x.T)






# build a small MLP and then train it with a brand-new constraints
