# generate a function from the quadratic family.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *

sampled_function = None;



def _generate_(dim):
    W = np.random.randn(dim, dim);
    y = np.random.randn(dim,1);
    return W, y;


# return sampled function f: R^n -> R
# return euclidean gradient of the function f: R^n -> R^n
def generate(dim):
    W, y = _generate_(dim);
    f = (lambda x: np.diag(np.matmul((np.matmul(W,x)-y).T, (np.matmul(W,x)-y))));
    grad_f = (lambda x: 2*(np.matmul(np.matmul(W.T, W), x)-np.matmul(y.T, W)));
    return f, grad_f;


if(__name__=='__main__'):
    f, grad_f = generate(2);


    # first draw loss surface
    pts = np.random.uniform(-10,10, size=[2,1000]);
    loss_surf = f(pts);
    colors = cm.hsv(loss_surf/max(loss_surf));
    colmap = cm.ScalarMappable(cmap=cm.hsv);
    colmap.set_array(loss_surf);


    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d');
    ax.scatter(pts[0,:],pts[1,:],loss_surf, c=colors, s=0.1);
    cb = fig.colorbar(colmap);



    start_pt = np.random.randn(2,1)+10;

    x_t = start_pt;
    last_f_val = 1000000000000;
    cter=0.000001;
    alpha = 0.05;
    iter=0;

    # construct a list of prober points and their gradients



    while(last_f_val-f(x_t)[0]>=cter):
        iter+=1;
        print("Iteration %d Loss %f" % (iter, f(x_t)[0]));
        direction = grad_f(x_t);
        if(np.mod(iter,1)==0):
            ax.quiver(x_t[0], x_t[1], 0.0, direction[0], direction[1], 0.0, length=0.5, normalize=True);
        last_f_val = f(x_t)[0];
        x_t = x_t - alpha* direction;

    ax.set_xlabel('X')
    ax.set_ylabel('Y');
    ax.set_zlabel('Loss');
    plt.show();
