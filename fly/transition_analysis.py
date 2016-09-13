import numpy as np
import matplotlib.pyplot as plt
import h5py


if __name__ =='__main__':
    #parameters:
    arrow_length = 0.15
    with_arrows = True
    with_random_adding = True
    max_adding = 0.4
    
    #read in data
    f = h5py.File('transitions.h5', 'r')
    trans_arr = f['data'][:,:]
    f.close()
    
    #data
    x = trans_arr[:,0]
    y = trans_arr[:,1]
    alpha = trans_arr[:,2]
    
    #add random values to data to see amount of datapoints with same coord
    n_coord = len(x)
    if with_random_adding==True:
        x = x + (2.*max_adding*np.random.rand(n_coord)-max_adding/2.)
        y = y + (2.*max_adding*np.random.rand(n_coord)-max_adding/2.)    

    print n_coord    

    #show data
    fig = plt.figure()
    fig.set_facecolor('w')
    ax = fig.add_subplot(1,1,1)
    ax.set_axis_bgcolor('w')
    ax.scatter(x,y, marker='.' ,color='k')
    if with_arrows == True:
        for idx in np.arange(trans_arr.shape[0]):
            arrow_y_start = y[idx]
            arrow_x_start = x[idx]
            arrow_delta_x = arrow_length*np.sin(alpha[idx])
            arrow_delta_y = arrow_length*np.cos(alpha[idx])
            ax.arrow(arrow_x_start, arrow_y_start, arrow_delta_x, arrow_delta_y , head_width=0.06, head_length=0.1, fc='k', ec='k', alpha=0.7, width=0.015)
    ax.set_title('fly transition behaviour')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

