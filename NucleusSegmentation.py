        # -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 14:41:13 2021

@author: bob van sluijs
"""

from scipy import ndimage
import numpy as np
import cv2
import matplotlib.pylab as plt 
from scipy.ndimage import gaussian_filter
import scipy as sp
import scipy.ndimage
from skimage.feature import peak_local_max
from scipy import ndimage
from PIL import Image, ImageOps
import math
import copy
import skimage
import os
import numpy
import skimage
import numpy as np
import networkx as nx
from scipy.spatial.ckdtree import cKDTree
from mpl_toolkits.mplot3d import Axes3D
from ImageAnalysisOperations import *

class RenderVisCoordinates:
    def __init__(self,coordinate,types,color = []):  
        self.coordinate  = coordinate
        self.color       = color 
        self.type        = types
        x,y,z = self.coordinate
        if self.type == 'cell':
            space = numpy.linspace(1,25,30)
            G = color[1] * space[z]
            space = numpy.linspace(0.25,1,30)
            B = self.color[-1] * space[z]
            space = numpy.linspace(1,2.5,30)
            R = self.color[0] * space[z]
            self.color = (R,G,B)
            
class ImageRendering:
    def __init__(self,
                 images
                 ):
        
        self.coordinates = []
        
class Render3D:
    def __init__(self,
                 information = []
                 ):
        
        self.information = information
        
    def update(self,information):
        self.information += information
        
    def visualize(self,):
        """@article{Zhou2018,
        author  = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
        title   = {{Open3D}: {A} Modern Library for {3D} Data Processing},
        journal = {arXiv:1801.09847},
        year    = {2018},}"""
        points = numpy.array([i.coordinate for i in self.information])
        colors  = numpy.array([i.color for i in self.information],dtype = numpy.float64)/255.
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        vis = open3d.visualization.VisualizerWithKeyCallback()
        
        """create the window and render"""
        vis.create_window()
        vis.get_render_option().background_color = numpy.asarray([0, 0, 0])
        vis.get_render_option().point_size = 5
        vis.add_geometry(point_cloud)
        vis.register_key_callback(32, lambda vis: exit())
        vis.run()
        vis.destroy_window() 


class ColorCoordinates:
    def __init__(self,color):
        self.color = color
        """coordinates"""
        self.coordinates = []
        
        """structure for plottin later"""
        self.z_projetion = {}
        self.boxes       = {}
    
    def update(self,coordinates,box):
        self.coordinates.append(coordinates)
        x,y,z = coordinates
        try:
            self.boxes[z].append(box)
            self.z_projetion[z].append((x,y))
        except:
            self.boxes[z] = []
            self.z_projetion[z] = []
            self.boxes[z].append(box)  
            self.z_projetion[z].append((x,y))
   
     
class AIsegment:
    def __init__(self,
                 warped,prediction
                 ):
        
        self.warped     = warped
        self.prediction = prediction

class Nucleus:
    def __init__(self,
                 coordinates,box,centroid,coverage,lbc,
                 warped = [],parameters = [],
                 signal = False
                 ):
        
        """the coordinates"""
        self.coordinates     = coordinates
        
        """individual coordinates"""
        self.x,self.y,self.z = coordinates
        
        """the segment stats"""
        self.box         = box
        self.centroid    = centroid
        self.coverage    = coverage
        self.warped      = warped
        self.label       = lbc
        self.signal      = signal
        
        """the list of boxes"""
        self.overlap = {}
        
        """the parameters of the segmentation"""
        self.parameters = parameters
        
        """AI is true by default"""
        self.AI = True

    def AIScreen(self,AI):
        self.prediction = AI.SVM_predict(self.warped)
        if 1 in self.prediction:
            self.AI = True
        else:
            self.AI = False

    def IntersectedOverlap(self,obj,p = 0.35):
        overlap = calculate_iou(obj.box,self.box)
        if overlap > p:
            self.overlap[obj] = overlap      
    
    def PruneOverlap(self,):
        """the single set with coordinates"""
        self.single = []
        """check z coordinates if overlap sequence is present 
        then its a single object in all likelihood also take 
        average of bounding box"""
        self.boxes       = []
        self.labelset    = []
        self.centroids   = []
        
        cc = []
        """the overlap between segments"""
        if len(self.overlap) > 1:
            for obj,nucleus in self.overlap.items():
                z,x,y   = obj.coordinates
                overlap = nucleus
                
                """add the overlapping box and the label segment"""
                self.boxes.append(obj.box)
                self.labelset.append(obj.label)
                self.centroids.append(obj.centroid)
                cc.append((copy.copy(z),copy.copy(overlap)))
                
            """sort the overlap"""
            overlap = list(sorted(cc))
            zstack,overlap = zip(*cc)
            difference = numpy.diff(zstack)
            
            """the single set, because we have a 3D we do not need to limit 
            segment segment assignmnent depth that is done for us in the 
            vertical segmentations"""
            for n in range(len(difference)):
                self.single.append(cc[n+1])
                    
        """get the centrioid set and apply the method"""
        self.centroidset = []
        for k,v in self.overlap.items():
            self.centroidset.append(k.centroid)
        self.centroidset = list(sorted(self.centroidset))

        """get the averaged out box of the segmentation that can show us
        where the final can be found"""
        if len(self.boxes) > 1:
            summed_bound_box = numpy.sum(self.boxes)
            
 
class Superstructure:
    def __init__(self,labelset,centroids,boxes):
        average = []
        """remove outliers in the dataset"""
        x,y,z = zip(*centroids)
        average.append(numpy.average(numpy.array(x)))
        average.append(numpy.average(numpy.array(y)))
        average.append(numpy.average(numpy.array(z)))
        
        """get centroid"""
        self.x = numpy.average(numpy.array(x))
        self.y = numpy.average(numpy.array(y))
        self.z = numpy.average(numpy.array(z))
        
        """get distances and remove some outliers"""
        distances = [euclidian_distance(tuple(average),i) for i in centroids]
        #remove the percentage that deveates!
        sortlist = numpy.argsort(numpy.array(distances))
        
        self.labelset,self.centroids,self.boxes = [],[],[]
        for i in range(0,int(0.9*len(sortlist)),1):
            for j in range(len(labelset[sortlist[i]])):
                self.labelset.append(labelset[sortlist[i]][j])
            self.centroids.append(centroids[sortlist[i]])
            self.boxes.append(boxes[sortlist[i]])

        """this is where we store the labels"""
        self.segments = []
            
    def CheckPoint(self,point):
        # print(self.boxes,point)
        x,y,z = point
        """the box boolean which """
        boxbool = []
        for box in self.boxes:
            
            boxbool.append(CheckPoint(box,(x,y)))
        # print(boxbool,len(self.boxes))
        IN = False
        """the boolean in the list"""   
        s = sum(boxbool)/float(len(boxbool))
        """check if over half is in""" 
        if s > 0.5:
            IN = True
        return IN
    
    def BuildNuclei(self,):
        from matplotlib.tri import Triangulation
        from scipy.spatial import ConvexHull     
        coordinates,centroids = [],[]
        for i in range(len(self.labelset)):
            coordinates.append(self.labelset[i])
        for i in range(len(self.segments)):
            for n in range(len(self.segments[i].label)):
                coordinates.append(self.segments[i].label[n])

        if type(coordinates[-1]) == list:
            c = [j for i in coordinates for j in i]
        self.coordinates = list(set(coordinates))
        X,Y,Z = zip(*self.coordinates)
        
        """the new set of x and y"""
        xl,xh = min(X),max(X) 
        yl,yh = min(Y),max(Y)
        zl,zh = min(Z),max(Z)
        
        """the generated in the """
        g = []
        for x in range(xl,xh):
            for y in range(yl,yh):
                    for z in range(zl,zh):
                        generated.append((x,y,z))
                        
        """append the two and do the cv hull calculation"""
        self.cvx = ConvexHull(numpy.array([X,Y,Z]).T)              
        self.tri = Triangulation(X,Y, triangles=self.cvx.simplices)
        """the coordinates in the nucleus"""
        for i in g:
            if i not in self.coordinates:
                if pointcheck(numpy.array([X,Y,Z]).T,i):
                    self.coordinates.append(i)
                    
    def Segmentation3D(self,struct,rotation = ''):
        """loop through the segmentations and find the overlap with the horizontal structure"""
        if rotation == 'x':
            centroid = struct.centroid
            """fit bool to check if centroid is in one of the boxes"""
            fitbool = []
            """check if centroid in bounding boxes"""
            for i in self.boxes:
                fitbool.append(CheckPoint(i,centroid))
            boolean = False
            if sum(fitbool)/float(len(fitbool)) > 0.25:
                boolean = True
                
            """if in bounding boxes append the labelsset"""
            if boolean == True:
                self.labelset_x[len(self.labelset_x)] = struct
            
        if rotation == 'y':
            centroid = struct.centroid
            """fit bool to check if centroid is in one of the boxes"""
            fitbool = []
            """check if centroid in bounding boxes"""
            for i in self.boxes:
                fitbool.append(CheckPoint(i,centroid))
            boolean = False
            if sum(fitbool)/float(len(fitbool)) > 0.18:
                boolean = True
                
            """if in bounding boxes append the labelsset"""
            if boolean == True:
                self.labelset_y[len(self.labelset_y)] = struct

class SegmentedNucleus:
    def __init__(self,image, 
                 
                  #is there a z component
                  z = False,
                 
                  #parameters of the segmentation
                  method           = 1,
                  secondary_filter = True,
                  differential = (3,3), min_distance = 15, minthresh = 375, boxsize  = (250,2500),
                 
                  #overwrites the parameters of the segmentation, can give 10, so its segmented 10 times
                  parameters = {},
                    
                  #screeening options after segmentation
                  AIenhanced = True,show_bbox = False, fig = False,SVM = None,
                  
                  #segments can be stored as images for training?
                  store = False,
                  
                  #store the images
                  storefolder = '',

                  ):
        
        
        """segment the nucleus
        INPUT
            Iamge, is the 
        
            segmentation parameters: 
                differential = (3,3), min_distance = 15, minthresh = 375, boxsize  = (250,2500),
                -the gradient of smoothing algorithm
                -minimal distance between cell
                -minimal threshold between cels
                -boxsize the min and max size of the box that can be bounded to a nucleus
            AI enhanced means an SVM model test if a segment label is actually a nucleus (default True)
        """
        
        store = False
        
        print("Starting Segmentation of DAPI Nuclei")
        self.parameters     = parameters
        
        """store data"""
        self.classification = []
        self.scatter        = []
        self.centroids      = []
        self.nucleisegment  = {}
        
        """segementataion of the nuclei, start by normalizing the image
        and create a uint8 object from the arrays!"""
        gray     = numpy.array(image/numpy.amax(image)*255,dtype = numpy.uint8)
        gray     = skimage.morphology.area_opening(gray)
        
        """store a copy of the image to do some image subtraction and addition with the 
        filtered images and with the original one to extract contours and the like"""
        initial  = copy.copy(gray)
        

        """3 different thresholding methodologies and other approaches
        this part of the code IF ELSE etc checks which parameters and filters 
        applied to the image, if you use different settings on the same image
        you wil get all the nucleii out in the end (i.e. different models)"""
        if method == 1:
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=differential)
            equalized = clahe.apply(gray)
            thresh = cv2.threshold(equalized, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 
            
        if method == 2:
            blur     = cv2.GaussianBlur(gray,(5,5),0)
            contrast = gray - blur            
            gray += contrast   
            thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]   
        if method == 3:
            blur     = cv2.GaussianBlur(gray,differential,0)
            contrast = gray - blur
            gray    -= contrast       
            thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  
                
        """initial thresholds, fill holes, find connected components"""
        init_thresh = skimage.morphology.remove_small_objects(thresh,min_size=125)
        thresh = ndimage.binary_fill_holes(init_thresh).astype(int)
        diff   = numpy.array((thresh - numpy.array(init_thresh,dtype = bool))*255,dtype = numpy.uint8)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(diff, connectivity=4) 
        sizes = stats[1:, -1]
        image = numpy.zeros((diff.shape))
        if len(sizes):
            for i in range(0, nb_components-1):
                if sizes[i] <=  minthresh:
                    image[output == i + 1] = 255 
        image  +=  init_thresh

        if secondary_filter and method != 3:
            r,c = numpy.nonzero(image)
            second_image = numpy.zeros(image.shape)
            second_image[r,c] = initial[r,c]
            gray   = numpy.array(second_image/numpy.amax(second_image)*255,dtype = numpy.uint8)
            gray   = skimage.morphology.area_opening(gray)        
            blur   = cv2.GaussianBlur(gray,differential,0)
            contrast = gray - blur
            gray -= contrast       
            thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  
            init_thresh = skimage.morphology.remove_small_objects(thresh,min_size=25)
            thresh = ndimage.binary_fill_holes(init_thresh).astype(int)
            diff   = numpy.array((thresh - numpy.array(init_thresh,dtype = bool))*255,dtype = numpy.uint8)
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(diff, connectivity=4) 

            image = numpy.zeros((diff.shape))
            sizes = stats[1:, -1]
            if len(sizes):
                for i in range(0, nb_components-1):
                    if sizes[i] <=  minthresh:
                        image[output == i + 1] = 255 
                        
        if secondary_filter:
            image  +=  init_thresh
            
        """perform distance tranform and do 'classic' watershed"""
        D = ndimage.distance_transform_edt(image.astype('int32'))
        localMax = peak_local_max(D, indices=False, min_distance=min_distance ,labels=image.astype('int32'))
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=image) 

        """used to compute the total surface area covered by nucleii"""
        self.averaged = copy.copy(init_thresh)

        """get the label in the and the box"""
        for label in np.unique(labels):
            if label == 0:
                continue
            
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            
            """get coordinates"""
            lbc = [tuple(i) for i in numpy.transpose(numpy.nonzero(mask))]
            
            """set minmax"""
            maximum,minimum = 0,10**10 
            signalvalues = {}
            
            """extract data"""
            for r,c in lbc:
                signalvalues[(r,c)] = initial[r,c]
                if initial[r,c] < minimum:
                    minimum = initial[r,c]
                if initial[r,c] > maximum:
                    maximum = initial[r,c]
            data          = numpy.zeros((512,512))
            data_original = numpy.zeros((512,512))
            for k,v in signalvalues.items():
                r,c = k
                data[r,c] = (v-minimum)/maximum
                data_original[r,c] = v

            rows_to_delete    = []
            columns_to_delete = []
            for i in range(len(data)):
                if sum(data[i]) == 0:
                    rows_to_delete.append(i)
                if sum(data[:,i]) == 0:
                    columns_to_delete.append(i)
            data = numpy.delete(data, rows_to_delete, 0)
            data = numpy.delete(data, columns_to_delete, 1)
            data_original = numpy.delete(data_original, rows_to_delete, 0)
            data_original = numpy.delete(data_original, columns_to_delete, 1)
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[labels == label] = 255
            lbc = [tuple(i) + (z,) for i in numpy.transpose(numpy.nonzero(mask))]
            cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box  = cv2.boxPoints(rect)
            box  = np.int0(box)  
            
            low,high = boxsize
            """getting the right segmentation out of the figure"""
            if cv2.contourArea(c) > low and cv2.contourArea(c) < high: 
                x,y         = boxcentroid(box)
                coordinates = (x,y,z)
                box         = box
                centroid    = (x,y,z)
                coverage    = cv2.contourArea(c)

                """store warped segment"""
                width = int(rect[1][0])
                height = int(rect[1][1])
                src_pts = box.astype("float32")
                dst_pts = np.array([[0, height-1],
                                    [0, 0],
                                    [width-1, 0],
                                    [width-1, height-1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))
                
                """keep warped image"""
                warped = cv2.warpPerspective(copy.deepcopy(gray), M, (width, height))
                
                """the unwarped 90 degree box cut-out"""
                cutout = gray[int(box[1][1]):int(box[3][1]),int(box[0][0]):int(box[2][0])]

                """if we use AI to assess the segmentation"""
                if AIenhanced:
                    """only add if passed through filter"""
                    index = len(self.nucleisegment)
                    """append relevant factors"""
                    self.scatter.append((x,y,z,box))
                    self.nucleisegment[index] = Nucleus(coordinates,box,centroid,coverage,lbc,parameters = parameters)
                    self.nucleisegment[index].warped = copy.copy(warped)
                    self.nucleisegment[index].AIScreen(SVM)
                    self.nucleisegment[index].cutout = copy.copy(cutout)

                else:
                    index = len(self.nucleisegment)
                    """append relevant factors"""
                    self.scatter.append((x,y,z,box))
                    self.nucleisegment[index] = Nucleus(coordinates,box,centroid,coverage,lbc,parameters = parameters)
                    self.nucleisegment[index].warped = copy.copy(warped)
                    self.nucleisegment[index].cutout = copy.copy(cutout)

                """get the sliced image with nucleus (or not)"""
                if store == True:
                    fig = plt.figure(frameon=False)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    self.nucleisegment[index].warped = warped
                    length = len(os.listdir(storefolder))
                    plt.imshow(warped,aspect='auto')
                    plt.savefig(storefolder + '\\' + str(length) + '.png')
                    plt.close()

                    fig = plt.figure(frameon=False)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    plt.imshow(cutout,aspect='auto')
                    plt.savefig(storefolder + '\\' + str(length) + 'cutout.png')
                    plt.close()
                    
                    fig = plt.figure(frameon=False)
                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)
                    plt.imshow(data,aspect='auto')
                    plt.savefig(storefolder + '\\' + str(length) + 'original.png')
                    plt.close()

                """delete the warping"""
                del warped
                
        if show_bbox == True:
            plt.figure(figsize = (15,5))
            plt.subplot(1,3,1)
            plt.imshow(initial)
            plt.subplot(1,3,2)
            plt.imshow(thresh)
            plt.subplot(1,3,3)  
            plt.imshow(labels)            
            plt.show()

            
class Y_segment:
    def __init__(self,
                 centroid,segmentlabels
                 ):
        
        self.z,self.y,self.x = centroid
        """because the vertical segmentation is rotated from the persective of the horizontal segmentation we need to assign segments"""
        self.label     = []
        self.centroid  = (self.x,self.y,self.z)
        z,y,x = zip(*segmentlabels)  
       
        """the x-labels from segmentation"""
        for i in range((len(x))):
            self.label.append((x[i],y[i],z[i]))   
            
class X_segment:
    def __init__(self,
                 centroid,segmentlabels
                 ):
        
        self.z,self.x,self.y = centroid
        """because the vertical segmentation is rotated from the persective of the horizontal segmentation we need to assign segments"""
        self.label     = []
        self.centroid  = (self.x,self.y,self.z)
        z,x,y = zip(*segmentlabels)     
      
        """the x-labels from segmentation"""
        for i in range((len(x))):
            self.label.append((x[i],y[i],z[i]))   
      


def PruneSegmentedOverlap(segmentations,rotation = 'x'):
    """this function takes the segmentation classes and clusters the bounding boxes
    together with a very wide margin of error so it might contain 3 stacked cells
    the goal is to find centroids and later align the segmentations with in the other directionsx"""
    boxcord,centroids,coordinates = [],[],{}
    
    """the nuclei segments and the overlap of the boxes"""
    
    for i in range(len(segmentations)):
        for j in range(len(segmentations)):
            if i != j:
                segmentations[i].IntersectedOverlap(segmentations[j])
            else:
                pass
        """the segment and the overlap that is priuned"""
        segment = segmentations[i] 
        segment.PruneOverlap()
    """return the boxcoordinates, the centroids and the coordinates"""
    popbin,structures = [],{}
    
    for i in segmentations.values():
        for n in i.overlap.keys():
            popbin.append(n)
        if i in popbin:
            pass
        else:
            if i.labelset != []:
                struct = Superstructure(i.labelset,i.centroids,i.boxes)
                structures[len(structures)] = Superstructure(i.labelset,i.centroids,i.boxes)
    return structures,segmentations


def Segment(
        #the z-stack coordinate, the image, do we store the images, the SVM model
        z,image,store,SVM,
        
        #the parameters that can be applied to the segmentaiton
        parameters = {},
        
        #show bbox
        show_bbox = False
        ):
    """This Function is called by the Segmentation Class again and again
    INPUTS:
        INT z
        ARR image (default 512x512, other size probably work)
        SVM model
        STRING: storepath where you want to display items      
        """
    
    """the segmentation parameters determine the outcome of an either succsesfull or unsuccessfull segmentation,
    we can take advantage of that by 'fitting' the parameters to indicidual nuclei until we find them with the AI tool """
    method             = parameters['method'] 
    secondary_filter   = parameters['secondary_filter']
    boxsize            = parameters['boxsize'] 
    minthresh          = parameters['minthresh'] 
    min_distance       = parameters['min_distance']   
    differential       = parameters['differential'] 


    """segment the nuclei in the image"""    
    segment = SegmentedNucleus( image,
                                z = z,
                                method = method,
                                secondary_filter = secondary_filter,
                                boxsize = boxsize,
                                differential = differential,
                                min_distance = min_distance,
                                minthresh = minthresh,

                               
                                show_bbox = show_bbox,
                                AIenhanced = True, 
                                store = True,
                                SVM = SVM,
                                storefolder = store,
                                parameters = parameters) 

    data = []
    
    """check the segments"""
    for j in segment.nucleisegment.values():
        if j.AI == True:
            data.append(j)
    return data

class Segmentation:
    def __init__(self,
                 path,
                 array = [],parameters = [],
                 limit = 0,
                 demo_1 = False,demo_2 = False,
                 storepath = '',
                 magnification = 60,
                 store = True, AIenhanced = True, fit = True, SVM = True,segmentsides = False,show_bbox = False,
                 ):
        
        """We check if we have an image or a path and we check if we do AI enhanced segmentation"""
        if array == []:
            images = OpenImage(path)    
        else:
            images = array
        """raw data"""
        self.images = images
        
        """the segment sets in both directions"""
        self.segment_topview   = ''
        self.segment_sideview  = ''
        
        import os
        folder   = 'Segmentation\\'
        topview  = folder + 'Nucleiisegments topview\\'
        sideview = folder + 'Nucleiisegments sideview\\'
        

        """store the segmentations"""
        if store == True:
            split  = 'All\\' + path.split('.')[-2].split('\\')[-1]
            self.segment_sideview = 'C:\\Users\\hanse\\Desktop\\' + sideview + split
            try:
                os.mkdir(self.segment_sideview)
                print("segmentation folder created (sideview)")
            except:
                print("Segmentation folder exists or failed to find directory")
            split  = 'All\\' + path.split('.')[-2].split('\\')[-1]
            self.segment_topview = 'C:\\Users\\hanse\\Desktop\\' + topview + split + '\\'
            try:
                os.mkdir(self.segment_topview)
                print("segmentation folder created (horizontal)")
            except:
                print("Segmentation folder exists or failed to find directory")   
             
        SVM_topview   = False
        SVM_sideview  = False
        """the SVM that detects a succesfull segmnentation"""
        if AIenhanced == True:
            """import the classification algorithm"""
            from MLclassification import MLtrainer

            if segmentsides == True:
                """sideview trainer"""
                SVM_sideview = MLtrainer(folder = sideview + 'Training data\\')  
                SVM_sideview.SegmentationData()
                SVM_sideview.SVM_Classifier() 
                
            """topview trainer"""
            SVM_topview = MLtrainer(folder = topview + 'Training data\\')   
            SVM_topview.SegmentationData()
            SVM_topview.SVM_Classifier()  
            
            """store predictions"""
            self.classification = []

        """the limit of the image"""
        limit  = skimage.filters.threshold_otsu(numpy.array(images))
        """the distribution"""
        self.distribution = []
        """the coordinates"""
        self.signal  = []
        """new image set"""
        self.nuclei = numpy.zeros((len(images),len(images[0]),len(images[0][:,0])))
        for i in range(len(images)):
            for row in range(len(images[i])):
                for column in range(len(images[i][:,0])):
                    self.distribution.append(copy.copy(images[i][row][column]))
                    
        """the limit of the signal is roughly the median, that way we assume half the image is covered in nuclei"""
        limit = numpy.median(numpy.array(self.distribution)) 
        for i in range(len(images)):
            for row in range(len(images[i])):
                for column in range(len(images[i][:,0])):
                    if images[i][row][column] > limit:
                        self.signal.append((row,column,i))
                        self.nuclei[i,row,column] = images[i][row][column]
                    else:
                        self.nuclei[i,row,column] = 0  

        """give the image and split it into new matrix x and y"""
        self.x_splits = {}
        self.y_splits = {}
        z,x,y = self.nuclei.shape
        for i in range(0,x,4):
            
            """x direction"""
            self.x_splits[i] = self.nuclei[:,i,:]
            xp,yp = self.nuclei[:,i,:].shape
            padding = numpy.zeros((xp+20,yp))
            padding[10:-10,:] = self.x_splits[i] 
            self.x_splits[i] = copy.copy(padding)
            
            """y direction"""
            self.y_splits[i] = self.nuclei[:,:,i]
            xp,yp = self.nuclei[:,:,i].shape
            padding = numpy.zeros((xp+20,yp))   
            padding[10:-10,:] = self.y_splits[i] 
            self.y_splits[i] = copy.copy(padding)

        
        self.averaged = {}
        
        """stacksizes that we compound"""
        stacksize = 3
        for i in range(len(self.nuclei) - stacksize):
            slate = numpy.zeros((self.nuclei[i].shape))
            for j in range(stacksize):
                slate += self.nuclei[i+j]
            slate = slate
            self.averaged[i + 1] = copy.copy(slate)
                        
        """number of centroids"""
        self.centroids = []
        
        """number of cells"""
        self.topview_segments,self.sideview_segments_x,self.sideview_segments_y = {},{},{}
        
        """boxes to store at image pos"""
        self.boxes = {i:[] for i in range(len(self.nuclei))} 
        
        """low to high bounding box sizes"""
        low,high,factor = 500,3000,1
        if magnification == 100:
            low,high,factor = 950,4000 * 100/60.,1.66

        if fit == True:
            if parameters == []:
                """manually defined segmentation parameters"""
                parameters = [
                             {'boxsize':(175,3000),
                                  'minthresh':50,
                                  'min_distance':8,
                                  'differential':(9,9),
                                  'method':1,
                                  'secondary_filter':False}
            
                            #  {'boxsize':(175,3000),
                            #                 'minthresh':50,
                            #                 'min_distance':8,
                            #                 'differential':(17,17),
                            #                 'method':1,
                            #                 'secondary_filter':True},
                            # {'boxsize':(175,3000),
                            #                 'minthresh':50,
                            #                 'min_distance':8,
                            #                 'differential':(9,9),
                            #                 'method':3,
                            #                 'secondary_filter':False},
                            # {'boxsize':(175,3000),
                            #                 'minthresh':250,
                            #                 'min_distance':8,
                            #                 'differential':(9,9),
                            #                 'method':3,
                            #                 'secondary_filter':False},
                            # {'boxsize':(175,3000),
                            #                 'minthresh':500,
                            #                 'min_distance':8,
                            #                 'differential':(9,9),
                            #                 'method':3,
                            #                 'secondary_filter':False},
                            # {'boxsize':(125,3000),
                            #                 'minthresh':50,
                            #                 'min_distance':8,
                            #                 'differential':(19,19),
                            #                 'method':2,
                            #                 'secondary_filter':True},
                            # {'boxsize':(125,3000),
                            #                 'minthresh':500,
                            #                 'min_distance':8,
                            #                 'differential':(19,19),
                            #                 'method':2,
                            #                 'secondary_filter':True}
                            ]
                            
                
        """scatter plot points of the bounding boxes"""
        self.scatter = []
        
        """"segment the cells horizontally """
        for p in parameters:
            for z in range(len(self.nuclei)):
                for n in Segment(z,self.nuclei[z],self.segment_topview,SVM_topview,parameters = p,show_bbox=show_bbox):
                    self.topview_segments[len(self.topview_segments)] = n
         
        if segmentsides == True:
            """segment the cells vertically"""
            for p in parameters:
                for i in range(0,max(list(self.x_splits.keys())),4):
                    for n in Segment(i,self.x_splits[i],self.segment_sideview,SVM_sideview,parameters = p,show_bbox=show_bbox):
                        self.sideview_segments_x[len(self.sideview_segments_x)] = n
                    for n in Segment(i,self.y_splits[i],self.segment_sideview,SVM_sideview,parameters = p,show_bbox=show_bbox):
                        self.sideview_segments_y[len(self.sideview_segments_x)] = n
                    
        """reconstruct the cells from the collected segment slices at each z-stack"""
        from CellReconstruction import CellSetFeatures
        from CellReconstruction import CellFeature
        
        """larger collection of cells in the image"""
        self.CSF      = CellSetFeatures(images = self.nuclei)
        
        """the features with the labels for the segments"""
        self.features     = {i:[] for i in range(len(self.nuclei))}

        """collect the features from the segments, unpakcing old class and repacking new class
        is a dumb way to do it but it cleans up memory going forward and its easier to see what gets 
        taken into the alignment and decoding part of the pipeline"""
        self.fcount = {}
        for i in self.topview_segments.values():
            if str(i.parameters) not in self.fcount:
                self.fcount[str(i.parameters)] = 0
            self.fcount[str(i.parameters)] += 1
            """append the cell feature to the dataset"""
            self.features[z].append(CellFeature(i.warped,i.label,i.coordinates,i.cutout,i.centroid,i.parameters))

        """reconstruct the nucleus and do the analysis"""
        self.CSF.update_individual_features(self.features)

        import os as os
        
        if demo_1:
            p = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop\\')
            """the images"""
            self.images = images
            import os
            if storepath == '':
                folder = p + 'Demo_1\\'
            else:
                folder = storepath + '\\Demo_1\\'
            try:
                os.mkdir(folder)
            except:
                pass
            for i in range(len(self.nuclei)):
                for x,y in self.centroids:
                    plt.scatter(x,y,s = 100,c = 'DarkRed')
                plt.imshow(self.images[i],cmap = 'jet')
                plt.savefig(folder + str(i) + '.png',dpi = 450)
            import imageio
            """store a gif of the slice"""
            filenames,imagenames = get_filenames(folder +'\\')
            gnames,filenames = zip(*filenames)
            images = []
            for n in range(len(gnames)):
                for m in filenames:
                    if '\\' + str(n) + '.png' in m:
                        images.append(imageio.imread(m))
            imageio.mimsave(folder+ 'movie.gif', images,fps=5)
            plt.close()
            
    def SegmentAssignment(self,):
        import pandas as pd
        """the temporary storage of local maxima"""
        cts = {i:True for i in self.signal}
        "1) build k-d tree"
        kdt = cKDTree(numpy.array(self.signal))
        edges = kdt.query_pairs(1)
        
        "2) create graph"
        G = nx.from_edgelist(edges)

        "3) Find Connections"
        ccs = nx.connected_components(G)
        node_component = {v:k for k,vs in enumerate(ccs) for v in vs}
        df = pd.DataFrame(self.signal, columns=['x','y','z'])
        df['c'] = pd.Series(node_component)
#        tempstore = []
        "4) extract features"
        feature_sets = {}
        for k,v in node_component.items():
            try:
                feature_sets[v].append((df['x'][k],df['y'][k],df['z'][k]))
            except:
                feature_sets[v] = []
                feature_sets[v].append((df['x'][k],df['y'][k],df['z'][k]))  
                
        cells = []
        for f,crd in feature_sets.items():
            if len(crd) > 250:
                cells.append(crd)

        colors = []
        """create some random colors for the cells"""
        for i in range(len(cells)):
            colors.append(random_color())
            
        """the coordinate system for the cells"""
        self.coordinate_system = []
        for n in range(len(cells)):
            for crd in cells[n]:
                self.coordinate_system.append(RenderVisCoordinates(crd,'cell',(50,10,255)))
                
