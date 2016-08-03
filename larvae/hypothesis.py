import numpy as np
import h5py
from skimage.transform import rotate
import os.path

#for show_3D_surface
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt



#rotational paramater
x_center = 510/2
y_center = 514/2
n_rot = 8

#read in parameter
path_scoremaps = '../../Deformabel-Part-Model/ccv/samples/larvae/pred/scoremap/'
start_idx = 200
n_scoremaps = 1


class ScoreStack:
    #checks if data is available in correct format, initialize class attributes and create stack in correct shape, if readin is True stack is read in else empty
    def __init__(self, img, stack_height, x_center=0, y_center=0,  path_scoremaps='scoremaps/', dataset_name='data', readin='True', transform='True'):
        self.img = img
        self.path_scoremaps = path_scoremaps
        self.dataset_name = dataset_name
        self.path_scoremaps = path_scoremaps 
        self.stackname = 'stack_' + str(img) #stack format: stack_1200
        self.stack_height = stack_height
        self.x_center = x_center
        self.y_center = y_center
        self.have_data = False
        #self.stack_shape = (0, 0, self.stack_height)
        self.score_stack = np.array((0, 0, stack_height), dtype=np.float32)
        if readin:
            self.read_in()
        if transform:
            assert(self.have_data)
            self.transform()

    #helper fct for read_in
    def exist_files(self):
        file_exist = True
        for i_rot in np.arange(self.stack_height):
            filepath = self.path_scoremaps+self.create_score_filename(i_rot)
            if os.path.isfile(filepath)==False:
                file_exist = False
                break
        return file_exist
    
    #helper fct for read_in 
    #set stack_shape according to first scoremap shape and checks if shape of all scoremaps belonging to one img have same shape
    def set_check_shape(self):
        for i_rot in np.arange(self.stack_height):
            filepath = self.path_scoremaps + self.create_score_filename(i_rot)
            score_file = h5py.File(filepath, 'r')
            score_shape = score_file[self.dataset_name].shape
            score_file.close()
            if i_rot==0:
                self.score_stack.resize(score_shape[0], score_shape[1], self.stack_height)
            else:
                if not(self.score_stack.shape[0] == score_shape[0] and self.score_stack.shape[1] == score_shape[1]):
                    print('wrong shape: expexted {}, got: {}'.format(self.score_stack.shape[0], score_shape[0]))
                    return False
        return True
         
    def read_in(self):
        assert self.exist_files()
        shape_consistent = self.set_check_shape()
        assert shape_consistent
        for i_rot in np.arange(self.stack_height):
            filepath = self.path_scoremaps + self.create_score_filename(i_rot)
            score_file = h5py.File(filepath, 'r')
            self.score_stack[:,:,i_rot] = score_file[self.dataset_name][:,:,0]
            score_file.close()
        self.have_data = True
            
    def write_hdf5(self, filepath='', dataset_name='data'):
        assert(self.have_data)
        filename = filepath + self.stackname + '.h5';
        stack_file = h5py.File(filename, 'w')
        stack_file.create_dataset(dataset_name, data=self.score_stack)
        stack_file.close()
    
    #show surface plot of probmap in selected region and for selected angle/stacknumber i_rot
    def show(self, x_start, y_start, x_end, y_end , i_rot, zlim=(-1,1)):
        assert(self.have_data)
        assert(x_start < x_end)
        assert(y_start < y_end)
        assert(i_rot < self.stack_height)
        region = self.score_stack[y_start:y_end, x_start:x_end , i_rot]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(x_end-x_start)
        Y = np.arange(y_end-y_start)
        X, Y = np.meshgrid(X, Y)
        Z = region[Y, X]
        #R = np.sqrt(X**2 + Y**2)
        #Z = np.sin(R)
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_zlim(zlim[0], zlim[1])
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()  
          
    
    def create_score_filename(self, i_rot): #scoremap filename format: scoremap_1234_4.h5
        return 'scoremap_' + str(self.img) + '_' + str(i_rot) + '.h5'        
    
    #rotate images in correct order
    def transform(self):
        angles = np.linspace(0, 360, self.stack_height, endpoint=False)
        for i_rot, angle in enumerate(angles):
            if i_rot==0:
                pass
            else:
                norm_factor = max(self.score_stack[:,:,i_rot].min(), self.score_stack[:,:,i_rot].max(), key=abs)
                self.score_stack[:,:,i_rot] *= 1/norm_factor 
                self.score_stack[:,:,i_rot] = rotate(self.score_stack[:,:,i_rot], 360.-angle, resize = False, center = (self.y_center, self.x_center))
                self.score_stack[:,:,i_rot] *= norm_factor 

if __name__ == '__main__':
    stack_354 = ScoreStack(354, n_rot, x_center=x_center, y_center=y_center, path_scoremaps=path_scoremaps)
    y_start = 162
    x_start = 317
    y_end = 212
    x_end = 360
    i_rot = 7
    #stack_354.show(x_start, y_start, x_end, y_end , i_rot, zlim=(-1.,16))
    stack_354.write_hdf5() 





