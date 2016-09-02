import numpy as np
import h5py
from skimage.transform import rotate
from skimage.draw import circle
import os.path
from scipy.ndimage.filters import maximum_filter
from skimage.feature import peak_local_max

#for show_3D_surface
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt




class ScoreStack:
    #checks if data is available in correct format, initialize class attributes and create stack in correct shape, if readin is True stack is read in else empty
    def __init__(self, img, stack_height, x_center=0, y_center=0,  path_scoremaps='scoremaps/', dataset_name='data', readin='True', transform='True'):
        self.img = img #index of scoremap 
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
            
    def write_hdf5(self, data, filepath='', filename='default', dataset_name='data'):
        filepath_name = filepath + filename + '.h5'
        stack_file = h5py.File(filepath_name, 'w')
        stack_file.create_dataset(dataset_name, data=data)
        stack_file.close()

    def write_stack(self):    
        assert(self.have_data)
        self.write_hdf5(data=self.score_stack, filename=self.stackname)

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

    #computes the maxima in all scoremaps
    #radius: radius in px for circle around current px for looking for maxima, for footprint
    #return maxima: list of arrays, for each i_rot one array
    def extract_maxima(self, radius=3, threshold_abs=0.):
        maxima = []

        #xx, yy = circle(radius, radius, radius)
        #footprint = np.zeros((radius*2+1, radius*2+1), dtype=np.bool_)
        #footprint[xx, yy] = 1
        for i_rot in np.arange(self.stack_height):
            #mode='wrap' means periodic boundary conditions
            coord = peak_local_max(self.score_stack[:,:,i_rot], min_distance=radius, threshold_abs=threshold_abs)
            maxima.append(coord)
        return maxima

    #generate hypotheses and represent them as an an array [samples, attributes] x, y, alpha, score
    def extract_hypotheses(self, radius=3, threshold_abs=0):
        maxima = self.extract_maxima(radius=radius, threshold_abs=threshold_abs)
        n_hypotheses = 0
        for i_rot in np.arange(self.stack_height):
            n_hypotheses += len(maxima[i_rot])
        hypos = np.zeros((n_hypotheses, 4), dtype=np.float32)
        i_sample = 0
        #iterate over samples
        for i_rot in np.arange(self.stack_height):
            for i in np.arange(maxima[i_rot].shape[0]):
                coord = maxima[i_rot][i,:]      #coord [x,y]
                maximum_value = self.score_stack[coord[0], coord[1], i_rot] 
                hypos[i_sample,0] = coord[0]
                hypos[i_sample,1] = coord[1]
                hypos[i_sample,2] = np.float32(i_rot)*360./np.float32(self.stack_height)
                hypos[i_sample,3] = maximum_value
                i_sample += 1
        return hypos

if __name__ == '__main__':
    #rotational paramater
    x_center = 510/2
    y_center = 514/2
    n_rot = 8

    #read in parameter
    path_scoremaps = '../../Deformabel-Part-Model/ccv/samples/fly/pred/scoremap/'
    start_idx = 2900
    n_scoremaps = 200


    stack_2989 = ScoreStack(2989, n_rot, x_center=x_center, y_center=y_center, path_scoremaps=path_scoremaps)
    stack_2989.write_stack()
    x_start = 513/2
    y_start = 922/2
    x_end = 533/2
    y_end = 935/2
    i_rot = 3
    hypos = stack_2989.extract_hypotheses(threshold_abs=0.35)
    print hypos[10,:]
    #for i in np.arange(stack_2942.stack_height)
    #    print len(maxima[i])
    #stack_2942.show(x_start, y_start, x_end, y_end , i_rot, zlim=(-1.,1.1))
    #stack_2942.write_stack() 





