# generate the s_2 data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *



def toy_func(pts):
    theta = np.arctan(np.sqrt((pts[:,0]**2+pts[:,1]**2)/pts[:,2]**2));
    phi = np.arctan(pts[:,1]/pts[:,0]);
    return np.cos(2*phi)*(np.sin(theta)**2)*np.cos(theta)*(3*(np.cos(theta)**2)-1);



# generate uniform distribution on the S_2 with a given number
# use the independent gaussian and a standard retraction
def gen_s_2(num=1000):
    _num=num;
    unif = np.random.randn(_num, 3);
    norms = np.sqrt(np.expand_dims(np.sum(unif**2, axis=1), axis=1));
    return unif/norms;

def gen_rand_2(num=1000):
    _num = num;
    return np.random.randn(_num, 3);


# ratio x bsize = num_of_pt_on_mfd
def gen_batch(bsize, op):
    pts = gen_s_2(bsize);
    return pts, spectrum_grad(op, pts);


def spectrum(op, prober):
    return np.diag(np.matmul(np.matmul(prober, op), prober.T));



# compute the E-gradient
def spectrum_grad(op, prober):
    return np.matmul(op, x.T);


def heat_map(bsize):
    dat = 5*gen_rand_2(bsize);
    height = toy_func(dat);
    colors = cm.hsv(height/max(height));
    colmap = cm.ScalarMappable(cmap=cm.hsv);
    colmap.set_array(height);
    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d');
    ax.scatter(dat[:,0],dat[:,1],dat[:,2], s=1, color= colors);
    cb = fig.colorbar(colmap);
    plt.show();



if(__name__=='__main__'):
    heat_map(1000);
    # to draw in order to show the uniform distribution on the sphere
