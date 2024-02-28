# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 14:22:23 2021

@author: bob van sluijs
"""
from PIL import Image, ImageEnhance

import os
import cv2 as cv2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

from skimage import io
from skimage.color import label2rgb
from skimage.feature import peak_local_max
from skimage.morphology import extrema
from skimage.measure import label, regionprops

import scipy
from scipy.signal import argrelextrema
from scipy import ndimage
from scipy import fftpack
from scipy.spatial import distance
from scipy.spatial.ckdtree import cKDTree
import copy

import cProfile
import pandas as pd
import numpy as np
import numpy
import itertools as it
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from NucleusSegmentation import RenderVisCoordinates
import warnings
import pickle


from ImageAnalysisOperations import *
    
class Feature:
    def __init__(self,
                 number
                 ):
        
        """The feature tracks the location of the signal in an image stack
        the number corresponds to the plane:
            we update the signal value (int or float)
            we update the coordinates (x,y,z) integers"""
        
        self.number      = number
        self.coordinates = []
        self.signal      = []
        self.signalplane = []
        
    def update_feature(self,
                       z,coordinate,signal
                       ):
        
        self.coordinates.append((z,coordinate))
        self.signal.append((z,signal))
        
    def signalmaxima(self,):
        """this can also be crossreferenced in the sequence coding"""
        z,s = zip(*self.signal)
        n,c = zip(*self.coordinates)
        s   = [0] + list(s) + [0]
        """find local maxima to indicate planeshift"""
        maxima = argrelextrema(numpy.array(s), numpy.greater)[0]
        """indices of the plane"""
        indices = [s.index(s[i]) -1 for i in maxima]
        """signalplane"""
        self.signalplane = [(z[i],c[i]) for i in indices]
        
class Spot:
    def __init__(self,
                 coordinates,spots
                 ):
        
        self.coordinates = coordinates
        self.GPS         = GPS

class GPS:
    def __init__(self,
                 coordinate,signalvalue,neighbors
                 ):
        
        """the GPS is the coordinate of the pixel in the image
            it takes as input the coordinate:
                    x,y,z integers in tuple format
                    the signalvalue (pixel intensity)
                    the coordinates of its neighboring pixels i.e. if the coordinate
                    is (5,5,5) then the neigbors are (4,5,5),(4,4,4)...etc"""
        self.GPS          = coordinate
        self.signalvalue  = signalvalue
        self.neighbors    = neighbors
        
        """Noise, check if the signal is surrounded by a better signal"""
        self.signal     = True
        self.psf        = False 
        self.gaussfit   = False
        
        """average signal strength"""
        self.average = sum(list(neighbors.values()))/float(len(neighbors))
        self.signals = list(self.neighbors.values())
        
    def UpdateDimensions(self,
                         position,rnd,channel
                         ):
        
        self.position = position
        self.round    = rnd
        self.channel  = channel
        
    def Spotdetection(self,
                      SVM
                      ):
        
        """The Support vector machine is added as an input:
            SVM is the trained model to detect point spread functions that reflect signals
            1) we assume the signal is false positiveL
        	2) normalize and reshape the pointspread function so a 7 x 7 matrix goes to a flat 49
            3) give flattened psf as input"""
        self.bool = [0]
        if self.psf.reshape(1,-1).shape == (1,49):
            self.bool = SVM.clf.predict(self.psf.reshape(1, -1)/numpy.amax(self.psf))
                       
    def GuassFit(self,
                 deactivate_fit = False
                 ):
        from astropy.modeling import models, fitting
        """Fit the data using astropy.modeling from the https://github.com/astropy/astropy
        this function is currently not used in the dataset"""
        p_init = models.Gaussian2D()
        fit_p = fitting.LevMarLSQFitter()
        
        x,y,z = [],[],[]
        """the gausslice of the set"""
        for row in range(len(self.psf)):
            for column in range(len(self.psf[row])):
                x.append(column)
                y.append(row)
                z.append(self.psf[row,column])
                
        """the numpy array of the system"""
        x,y,z = numpy.array(x).reshape(7,7), numpy.array(y).reshape(7,7), numpy.array(z).reshape(7,7)    
        z = z/numpy.max(z)
        
        if deactivate_fit:
            with warnings.catch_warnings():
                """Ignore model linearity warning from the fitter"""
                warnings.simplefilter('ignore')
                p = fit_p(p_init, x,y,z)
                """hte guassian fit of the point spread function"""            
                self.gaussfit = p(x, y)


class Slice:
    def __init__(self,
                 image,r,z,
                 tol = 2,distance  = 3,mean = 2,cutoff = 3,
                 show  = False,statistical_test = False,
                 path   = '', nbs    = {}
                 ):
        
        """the slice is a single image for the ARTseqFISH paper this is a 512x512 matrix:
                -this is the class where the image matrix is analysed subjected to global and local image filters
                -you can manually set some hyper parameters for the filter
                - depending on the signal to noise you can play around wth the hyper parameters
                
                INPUTS:
                        the matrix with values 2D (512x512 default), the imaging round (INT), the z-stack coordinate (INT)
                """
        from scipy.ndimage import gaussian_filter
        """the name and the file"""
        self.z      = z
        self.round  = r
        self.matrix = numpy.array(image)
        
        """get the row and column with pixel values sorted, the mean to follow"""
        self.pixels = {}
        for row in range(len(self.matrix)):
            for column in range(len(self.matrix[:,0])):
                self.pixels[(row,column)] = self.matrix[row,column]
        
        """get the mean and median of the pixelset"""
        self.median = numpy.median(numpy.array(list(self.pixels.values())))
        self.mean   = numpy.mean(numpy.array(list(self.pixels.values())))
               
        """map out the neighbors of the pixels"""
        if nbs == {}:
            self.neighbors = {}
            for i in self.pixels.keys():
                self.neighbors[i] = [n for n in neighbors(i)]
        else:
            self.neighbors = nbs

        """remove the median/mean from the matrix"""
        self.filtered_matrix = self.matrix - self.mean
        self.filtered_matrix[self.filtered_matrix < 0] = 0 
            
        """all signal values"""
        signal = [i for i in self.filtered_matrix.flatten() if i > 0]
        
        """initially filter out the peaks"""
        first  = gaussian_filter(self.filtered_matrix, sigma = mean)     
        array  = numpy.array(self.filtered_matrix,dtype = float) - first
        array[array < 0] = 0
        
        """guassian filter"""
        self.gaussian_filter = copy.copy(array)

        
        """sort the vector for interpolation"""
        sorted_vector = list(sorted([i for i in array.flatten()]))
        n2, bins2, patches = plt.hist(sorted_vector,bins = 100)
        plt.close()
        
        """take summed percentage"""
        cumsum = numpy.cumsum(n2)
        percentage = numpy.cumsum(n2)/numpy.sum(n2)
        signal_threshold = sorted_vector[int(cumsum[1])]
        array[array < signal_threshold] = 0
        a = copy.copy(array)
        
        """imagestore"""
        self.noise_isolation = a

        """after thresholding the image we remove the noise by removing
        pixels with without neighboring pixel signals"""
        zero = []
        """neirest neighbor search"""
        for coordinate,n in self.neighbors.items():
            """get x,y"""
            x,y = coordinate
            """set zero count"""
            if array[x,y] > 0:
                zc = 0
                for j in n:
                    try:
                        xn,yn = j
                        if array[xn,yn] == 0:
                            zc += 1
                    except:
                        pass
                    
                """if the pixel is neighbored by 4 zeros i.e. half its set as noise"""
                if zc > cutoff:
                    zero.append(coordinate)   
                    
        """set this sucker to zero"""
        for x,y in zero:
            array[x,y] = 0
            
            
        self.filtered = array        
        """get the local maxima in the filtered array, these are spots"""
        self.coordinates = peak_local_max(array, min_distance=distance,exclude_border=False) 
        """the signalset i.e. position where positive signals were found"""
        self.signalset   = {}
        for x,y in self.coordinates:
            self.signalset[(x,y,z)] = self.matrix[x,y]
        self.GPS = [tuple(i) + (z,) for i in self.coordinates]
        
        """the coordinates in the matrix"""
        self.GPS = {}
        for i in self.coordinates:
            x,y = i
            n = {(xj,yj,)+(z,):self.matrix[xj,yj] for xj,yj in self.neighbors[(x,y)] if xj < len(self.matrix) and yj < len(self.matrix[-1])  and xj >= 0 and yj >= 0}
            n.update({(x,y,z):self.signalset[(x,y,z)]})
            self.GPS[(x,y,z)] = GPS(tuple(i) + (z,),self.signalset[(x,y,z)],n)
            try:
                self.GPS[(x,y,z)].psf = self.matrix[x-3:x+4,y-3:y+4]
                self.GPS[(x,y,z)].GuassFit()
            except:
                pass
                    
        """signals in the set"""
        self.n_signal,self.average = [],[]
        for i in self.GPS.values():
            for n in i.signals:
                self.n_signal.append(n)
            self.average.append(i.average)
            
        """you can do a statistical test to check signal to noise"""
        if statistical_test == True:
            self.data,self.signal = [],[]
            for i in self.matrix:
                for j in i:
                    self.data.append(j)
                    if j > self.mean:
                        self.signal.append(j)
            self.intensities = {}
            for p,signal in self.pixels.items():
                try:
                    self.intensities[signal].append(p)
                except:
                    self.intensities[signal] = []
                    self.intensities[signal].append(p)
                    
            """the scipy stat test to check if a signal is noise or not"""
            self.SNR,self.pvalue = scipy.stats.ttest_ind(self.signalset,self.data)

class Stack:
    def __init__(self,
                 stack,
                 folder = '', nbs = {},
                 tol = 2,
                 show = False, AIStore = False
                 ):
        
        """the stack containing all the slices (single image at specific z-stack) for the ARTseqFISH paper
                - this is the class where the information of all the slices is combined
                - depending on the signal to noise you can play around wth the hyper parameters
                
                - NOTE here the images are once again placed in a 3D matrix! to resemble the original and to find all the connected spots
                      now we can analyse if there is duplicate signals detected across the z-stack due to nyquist samling detecting the same spot
                      we can eliminate duplicates in this manner, these connections are found through a binary graph.
                
                
                INPUTS:
                        The image stack, you can set AIstore to true, this will store the PSFs detected, which you can then sort to build a training dataset
                """
                
        self.signal      = []
        """dict of the same with intensity value"""
        self.signalset   = {}
        """the GDS signal in the system"""
        self.GPS         = {}
      
        for z,slc in stack.items():
            """get the signal values for the depth"""
            self.signalset.update(slc.signalset)
            self.GPS.update(slc.GPS)
        
        """if we want to do proper spot detection is advantagious to select
        signals on the basis of their pointspread, in instance we only select proteins 
        and mRNA signals which have a gaussian pointspread function"""
        if AIStore == True:
            datafolder = "C:\\Users\\huckg\\OneDrive\\Desktop\\Spotdetection\\SignalFilter\\All\\"
            for coordinate,signal in self.GPS.items():
                """store the pointspread functions"""
                psf  = signal.psf
                if type(psf) == numpy.ndarray and len(psf) > 0:
                    try:
                        psf  = psf/numpy.amax(psf)
                        data = psf.flatten()
                        
                        """save the array in the ALL folder"""
                        numpy.save(datafolder + '\\{} data.npy'.format(len(os.listdir(datafolder))), data)
                        plt.title(str(numpy.amax(psf)))
                        plt.imshow(psf,cmap = 'jet')
                        plt.savefig(datafolder + '\\' + '{} figure.png'.format(len(os.listdir(datafolder)) - 1))
                        plt.close()
                        
                    except ValueError:
                        print(psf)
    
        """get rid of the noise in the spots"""  
        signal = sorted([v.signalvalue for k,v in self.GPS.items()])
        n2, bins2, patches = plt.hist(signal,bins = 50)
        plt.close()

        threshold = signal[int(n2[0])]
        if signal[int(sum(n2)-1)]/threshold > 10:
            self.GPS = {c:v for c, v in self.GPS.items() if v.signalvalue > threshold}
        self.coordinates = list(self.GPS.keys())
        
        self.stack = stack
        """the temporary storage of local maxima"""
        cts = {i:True for i in self.coordinates}
        
        "1) build k-d tree"
        kdt = cKDTree(numpy.array(self.coordinates))
        edges = kdt.query_pairs(1)
        
        "2) create graph"
        G = nx.from_edgelist(edges)

        "3) Find Connections"
        ccs = nx.connected_components(G)
        node_component = {v:k for k,vs in enumerate(ccs) for v in vs}
        df = pd.DataFrame(self.coordinates, columns=['x','y','z'])
        df['c'] = pd.Series(node_component)

        "4) extract features"
        feature_sets = {}
        for k,v in node_component.items():
            try:
                feature_sets[v].append((df['x'][k],df['y'][k],df['z'][k]))
            except:
                feature_sets[v] = []
                feature_sets[v].append((df['x'][k],df['y'][k],df['z'][k]))
      
        self.spots = {}
        for i in self.coordinates:
            self.spots[i] = Spot(i,self.GPS[i])
        for i,c in feature_sets.items():
            self.spots[tuple(c)] = Spot(c,[self.GPS[n] for n in c])
        
        """get the overlapping features of the GPS signal i.e. what is the feature count versus the total count"""
        self.GPS_number = len(self.GPS)
        
        """the spotnumber"""
        self.spotnumber = []
        for i,c in feature_sets.items():
            for ii in c:
                cts[ii] = False
            self.spotnumber.append(c)
        for c,boolean in cts.items():
            if cts[c] == True:
                self.spotnumber.append(c)
                
        """Print the relevant information about the counts"""
        print("the largest connected stacks")
        print(df['c'].value_counts().nlargest(5))
        print("The GPS is: {}".format(len(self.GPS)))       
        print("The Spotnumber is: {}".format(len(self.spotnumber)))
        
        average = [i.average for i in self.GPS.values()]
        signal  = [n for i in self.GPS.values() for n in i.signals]
        
        """get the binned histogram and counts"""
        y, x, _ = plt.hist(signal,bins = 10)
        plt.close()
    
        if show == True:
            df['c'] = pd.Series(node_component)
            df.loc[df['c'].isna(), 'c'] = df.loc[df['c'].isna(), 'c'].isna().cumsum() + df['c'].max()
            fig = plt.figure(figsize=(25,25))
            ax = fig.add_subplot(111, projection='3d')
            cmhot = plt.get_cmap("hot")
            ax.scatter(df['x'], df['y'], df['z'], c=df['c'], s=50)
            plt.show()
                
    def plot(self,
             folder = "",name = '',transcriptional_centers = False
             ):
        
        """this function plots the individually analysed images per z-stack
                the main class that tracks everything is IMAGE: you can give this function as an input there."""
        pass
        f = folder
        if folder == "":
            f = os.path.join(os.path.join(os.path.expanduser('~'))) + "\\Desktop\\Visualize Spot Counting\\"
            
        """create folder to store slice"""
        try:
            os.mkdir(f)
        except:
            pass
        
        for z,image in self.stack.items():
            import seaborn as sns
            sns.set_context('notebook',font_scale = 1.35)
            plt.figure(figsize=(10,10))
            plt.subplot(2,2,1)
            plt.title("Original Image")
            upperlimit = 1*numpy.amax(image.matrix.flatten())
            plt.imshow(image.matrix,cmap = 'jet')
            plt.clim(0,upperlimit)
            plt.subplot(2,2,2)
            plt.title("Sharpened Image")
            plt.imshow(image.filtered,cmap = 'jet')
            plt.clim(0,upperlimit)
            plt.subplot(2,2,3)
            plt.title("Local Maxima: "+str(len(image.coordinates)))
            plt.scatter(image.coordinates[:, 1], image.coordinates[:, 0], color = 'w',s = 5)
            plt.imshow(image.matrix,cmap = 'jet')
            plt.clim(0,upperlimit)
            plt.savefig(f+name+ str(z) + '.png',dpi = 700)
            plt.subplot(2,2,4)
            plt.title("Zoom: "+str(len(image.coordinates)))
            plt.scatter(image.coordinates[:, 1], image.coordinates[:, 0], color = 'w',s = 50,edgecolor = 'k')
            plt.imshow(image.matrix,cmap = 'jet')
            
            plt.clim(0,upperlimit)
            plt.xlim([250,350])
            plt.ylim([250,350])
            plt.savefig(f+name+ str(z) + '.png',dpi = 700)
            plt.show() 
            
               
class Image:
    def __init__(self,arrayset,
                 folder = "",location = "\\Spotdetection\\SignalFilter\\", 
                 plot = False,
                 tol = 2, distance = 1,mean = 3, cutoff = 2,
                 name = ''
                 ):
        
        """The image class is where all the information is collected:
            INPUT list of arrays [arr1,arr2...] representing z-stacks (or timepoints)
            Location = the location of the test and training data for the SVM to classify spots as true or false positives
            Plot = True will show a 3 panel plot with the original, the filtered and the detected local maxima
            parameters = TOL 2, distance 1, mean 3, cutoff 2: 
                the tolerance, allowed distance between connected spots in z-stack
                mean - global filter
                cutoff - number of ppixels that need to be larger > 0 for it to be considered a PSF to begin with (saves eval time for the SVM) """
        
        """go through a single image and get all the imformation out of it"""
        if type(arrayset) == str:
            arrayset = OpenImage(arrayset)

        """the compounded image sets"""
        compounded = numpy.add.reduce(arrayset)
        """the stack and the single images within the stack"""
        self.stack,nbs = {},{}
        """the arrayset in the slice"""
        for i in range(len(arrayset)):
            array = Slice(arrayset[i],None,i,nbs = nbs,tol = tol,distance = distance,mean = mean,cutoff = cutoff)
            self.stack[i] = copy.copy(array)
            """this is a timeintensive step so best to only do it once for a single matrix then copy it into others"""
            nbs  = array.neighbors
            try:
                if i%int(len(arrayset)/20.) == 0:
                    print('\rDetecting Spots [%d%%]'%int((i/float(len(arrayset)))*(100)), end="") 
            except:
                pass
                
        self.nbs = nbs
        """the arrayset in the """
        self.arrayset = arrayset
        """the compounded image"""
        self.compounded = Slice(compounded,None,i,nbs)
        """the stacked images analysed"""
        self.stacked = Stack(self.stack,nbs = nbs,show = False)
        """signal coordinates"""
        self.GPS        = self.stacked.GPS
        self.spots      = self.stacked.spots
        self.spotnumber = self.stacked.spotnumber

        """train detector of SVM machine"""
        f = os.path.join(os.path.join(os.path.expanduser('~'))) + '\\' 'desktop\\' + location

        """the path in the set"""
        from MLclassification import MLtrainer
        SVM = MLtrainer(folder = location + 'Training data\\')
        SVM.PSFData()
        SVM.SVM_Classifier()  
        
        """check if the spots have gaussian distributions"""
        self.AIcoordinates = []
        for c,obj in self.GPS.items():
            obj.Spotdetection(SVM)
            if obj.bool[0] == True:
                self.AIcoordinates.append(c)
        self.AIcheck = len(self.AIcoordinates)
        
        """collect the count data and present it"""
        self.count = {'Hybridization Number':len(self.GPS),
                      "Target Number":self.spotnumber, 
                      "Hybridizations AI number":self.AIcheck,
                      "Fraction": self.AIcheck/float(len(self.GPS)),
                      'Hybridization GPS':list(self.GPS.keys()),
                      'Target GPS':list(self.spots.keys()),
                      'AI Hybridization GPS': self.AIcoordinates}
        
            
        """the fraction of spots in the spotcount that are approved by the AI"""
        print('AI spotcount = {}'.format(str(self.AIcheck)))
        print('Fraction: {}'.format(str(self.AIcheck/float(len(self.GPS)))))
        
        """the spots in the compounded image"""
        print( "the compounded stack count: {}".format(len(self.compounded.coordinates)))
        self.rendering = [RenderVisCoordinates(i,'spots',(200,200,200)) for i in self.GPS]

    def plot(self,folder = '',name = '1'):
        self.stacked.plot(folder = folder,name = name)
        
    def plot3D_coordinates(self,):
        """@article{Zhou2018,
        author  = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
        title   = {{Open3D}: {A} Modern Library for {3D} Data Processing},
        journal = {arXiv:1801.09847},
        year    = {2018},}"""
        
        import open3d as o3d
        import open3d
        import numpy
        import random
        
        z,x,y  = zip(*self.GPS)
        xyz    = numpy.array([x,y,z]).T

        R = [random.random() for i in range(len(xyz))]
        G = [random.random() for i in range(len(xyz))]
        B = [random.random() for i in range(len(xyz))]
        
        colors = numpy.array([R,G,B],dtype = numpy.float64).T
        """load the cloud and render it in real terms"""
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        vis = open3d.visualization.VisualizerWithKeyCallback()
        
        """create the window and render"""
        vis.create_window()
        vis.get_render_option().background_color = numpy.asarray([0, 0, 0])
        vis.get_render_option()
        vis.get_render_option().point_size = 2
        vis.add_geometry(point_cloud)
        vis.capture_screen_image('C:\\Users\\hanse\\desktop\\pointcloud.png')
        vis.register_key_callback(32, lambda vis: exit())
        vis.run()
        
