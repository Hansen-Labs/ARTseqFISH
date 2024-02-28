# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 12:12:41 2021

@author: huckg
"""
from PIL import Image, ImageEnhance
from PIL import Image as Im

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from scipy import ndimage
from scipy import fftpack
from scipy.spatial import distance
from scipy.spatial.ckdtree import cKDTree
from scipy.signal import argrelextrema

from skimage import io
from skimage.morphology import extrema
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.feature import peak_local_max

import numpy
import os
import cv2 as cv2
import copy
import scipy
import cProfile
import pandas as pd
import numpy as np
import networkx as nx
import itertools as it

from mpl_toolkits.mplot3d import Axes3D
from NucleusSegmentation import RenderVisCoordinates
from NucleusSegmentation import Segmentation
from ImageAnalysisOperations import *

def AlignFeatures(benchmark,comparison):
    """the paths for the benchmark + comparison"""  
    structures = []
    for z in sorted(benchmark.keys()):
        try:
            for b in benchmark[z]:
                for c in comparison[z]:
                    print(b,c)
                    structures.append(StructureComparison(b,c,z = z))
        except:
            print("No structures present on z-" + str(z))

    """sort sets with low scores"""
    b = numpy.argsort(numpy.array([i.lsq for i in structures]))
    c = numpy.argsort(numpy.array([i.lsq_cutout for i in structures]))
        
    shiftset = []
    for i in range(5):        
        """structures"""
        shiftset.append(structures[b[i]].shift)
    x,y = zip(*shiftset)
    
    """roundset of the set"""
    round_x = [round(number/10)*10 for number in x]
    round_y = [round(number/10)*10 for number in y]
    
    """bins"""
    bins = [10*i for i in range(52)]
    for i in range(52):
        bins.append(-10*i)
        
    """neighbors in the set"""
    neighbors = {i:(i-10,i+10) for i in bins}
    """collections in bins"""
    collection = {}
    """rouding the set"""
    for i in range(len(round_x)):
        r = round_x[i]
        low,high = neighbors[r]
        
        if int(r) in collection.keys():
            collection[r].append(i)
        else:
            """the low and high sets"""
            collection[r] =  []    
            collection[r].append(i)  
            
        if int(low) in collection.keys():
            collection[low].append(i) 
        else:
            collection[low] = []   
            collection[low].append(i)
            
        if int(high) in collection.keys():
            collection[high].append(i)
        else:
            collection[high] = []    
            collection[high].append(i)

        
    setrange,index = 0,0
    """find the highest count"""
    for k,v in collection.items():
        if len(v) > setrange:
            setrange = len(v)
            index    = copy.copy(k)
            
    
    """the shif averaged out"""
    shift = [shiftset[i] for i in collection[index]]
    x,y   = zip(*shift)
    x_ave,y_ave = int(sum(list(x))/float(len(x))), int(sum(list(y))/float(len(y)))
    # print(x_ave,y_ave)
    
    """shift the other set"""
    return x_ave,y_ave,shift


def OverlapImages(shift,benchmark,comparison):
    xs,ys  = shift 
    expset = max([xs,ys])
    """padding"""
    p = max([abs(xs),abs(ys)])
    """the benchmark and comparison images"""
    benchmark = OpenImage(benchmark)    
    comparison = OpenImage(comparison)
    for i in range(0,len(benchmark),1):
        b = benchmark[i]
        
        """the benchmark"""
        b_binary = Binary(b)
        
        """the comparison"""
        c = comparison[i]
        c_binary = Binary(c)

        
        stack = numpy.zeros((b_binary.shape))
        for row in range(len(b_binary)):
            for column in range(len(b_binary)):
                
                try:
                    if row + xs >= 0:
                        if column+ys >= 0:
                            stack[row,column] = b_binary[row,column] + c_binary[row+xs,column+ys]
                    else:
                        stack[row,column] = b_binary[row,column]
                except:
                    stack[row,column]  = b_binary[row,column]
        plt.imshow(stack,cmap = 'hot')
        plt.colorbar()
        plt.show()   

def Binary(b):
    import skimage
    """the limit in the race"""
    limit = numpy.median(numpy.array(b)) 
    b[b < limit] = 0 
    
    """Create a rought binary image of the cells"""
    gray = numpy.array(b/numpy.amax(b)*255,dtype = numpy.uint8)
    gray = skimage.morphology.area_opening(gray)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(31,31))
    equalized = clahe.apply(gray)
    thresh = cv2.threshold(equalized, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 
    init_thresh = skimage.morphology.remove_small_objects(thresh,min_size=36)
    thresh = ndimage.binary_fill_holes(init_thresh).astype(int)
    diff = numpy.array((thresh - numpy.array(init_thresh,dtype = bool))*255,dtype = numpy.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(diff, connectivity=4) 
    sizes = stats[1:, -1]
    image = numpy.zeros((diff.shape))
    
    """Set the values of what we consider part of the nucleus"""
    for i in range(0, nb_components-1):
        if sizes[i] <=  50:
            image[output == i + 1] = 255       
    
    image  +=  init_thresh
    return numpy.array(image,dtype = bool)

    
def StoreImageOverlap(
                      shiftset,
                      imagepaths = {},calibration = {},
                      path = '',
                      ):
     """store the images by round in a dictionary"""
     images = {r:OpenImage(p) for r,p in imagepaths.items()}
     """the images"""
     for i, shift in shiftset.items():
        benchmark,comparison = i
        columnshift,rowshift,zshift = shift
        #this part is ugly, it requires an arbitrary factor...
        z = int(len(images[benchmark])*0.85)
        """the benchmark"""
        b = Binary(images[benchmark][z])
        """the comparison"""
        c = Binary(images[comparison][z])

        """the binary set of the nuclei, to be plotted"""
        stack = numpy.zeros((b.shape))
        for row in range(len(c)):
            for column in range(len(c[row])):
                if row + rowshift >= 0 and row + rowshift < 512:
                    if column + columnshift >= 0 and column + columnshift < 512:
                        stack[row+rowshift,column+columnshift] = c[row,column] 
                        
        """shift the stack"""      
        stack += b
                    
        plt.imshow(stack,cmap = 'hot')
        plt.title('R{1} -> R{0}'.format(str(benchmark),str(comparison)) + str(shift))
        try:
            plt.savefig(path + 'R{1} & R{0}'.format(str(benchmark),str(comparison),dpi = 600))
        except:
            pass
        plt.close()

        """create the comparison"""
        s = z + zshift
        if s > len(images[comparison]):
            s = len(images) - 1
        c = Binary(images[comparison][s])
        
        """the binary set of the nuclei, to be plotted"""
        stack = numpy.zeros((b.shape))
        for row in range(len(c)):
            for column in range(len(c[row])):
                if row + rowshift >= 0 and row + rowshift < 512:
                    if column + columnshift >= 0 and column + columnshift < 512:
                        stack[row+rowshift,column+columnshift] = c[row,column] 
                        
        """shift the stack"""      
        stack += b
                    
        plt.imshow(stack,cmap = 'hot')
        plt.title('Z R{1} -> R{0}'.format(str(benchmark),str(comparison))  + str(shift))
        try:
            plt.savefig(path + 'Z R{1} & R{0}'.format(str(benchmark),str(comparison),dpi = 600))
        except:
            pass
        plt.close() 

def StoreImageCorrectionOverlap(
                                shiftset,
                                imagepaths = {},calibration = {},
                                path = ''
                                ):   
    
     """store the images by round in a dictionary"""
     images = {r:OpenImage(p) for r,p in imagepaths.items()}
     """the images"""
     for i, shift in shiftset.items():
        benchmark,comparison = i
        columnshift,rowshift,zshift = shift
        z = int(len(images[benchmark])*0.75)
        """the benchmark"""
        b = Binary(images[benchmark][z])
        """the comparison"""
        c = Binary(images[comparison][z+zshift])

        """the binary set of the nuclei, to be plotted"""
        stack = numpy.zeros((b.shape))
        for row in range(len(c)):
            for column in range(len(c[row])):
                if row + rowshift >= 0 and row + rowshift < 512:
                    if column + columnshift >= 0 and column + columnshift < 512:
                        stack[row+rowshift,column+columnshift] = c[row,column] 
                        
        """shift stack"""      
        stack += b
        cmap  = 'hot'
        plt.imshow(stack,cmap = cmap)
        try:
            plt.savefig(path + 'Manual Correction R{1} & R{0}.png'.format(str(benchmark),str(comparison),dpi = 800))
        except:
            print('uh oh it is not plotting')
        plt.close()
         
def intersect_matrices(
                       mat1, mat2
                       ):
    if not (mat1.shape == mat2.shape):
        return False
    mat_intersect = np.where((mat1 == mat2), mat1, 0)
    score = numpy.sum(mat_intersect)
    del mat_intersect
    return score 

def FeatureFinder(
                  array = [],
                  path  = ''
                 ):
    
    import skimage
    from NucleusSegmentation import Segmentation
    """We check if we have an image or a path and we check if we do AI enhanced segmentation"""
    if array == []:
            images = OpenImage(path)    
    else:
        images = array

    """segment the image and check the path"""
    segment = Segmentation(path,array = images,AIenhanced = True,SVM = False)
    """Use the nuclei to compare the overlap between images """    
    segments = {}
    for i in segment.horizontal_segments.values():
        x,y,z =  i.coordinates
        """get the coordinates and sort by z"""
        try: 
            segments[z].append(ImageFeatures(z,i.warped,i.label,i.coordinates,i.cutout,images[z]))
        except:
            segments[z] = []
            segments[z].append(ImageFeatures(z,i.warped,i.label,i.coordinates,i.cutout,images[z]))
    return segments

def SortLocalRefinement(
                       dataset
                       ):
    #sort by round then calibration then score, sum the total alignment
    rounds = {}
    for i in dataset:
        try:
            rounds[i['Round']].append((i['Score'],i['Calibration']))
        except:
            rounds[i['Round']] = []
            rounds[i['Round']].append((i['Score'],i['Calibration']))
        
    summed = {}
    for r,iset in rounds.items():
        scoredict = {}
        for s,c in iset:
            try:
                scoredict[c] += s
            except:
                scoredict[c] = 0
                scoredict[c] += s
                
        """summed dictionary of the set"""    
        summed[r] = scoredict
        
    calibration = {}
    for r,scores in summed.items():
        calibration[r] = sorted([(v,k) for k,v in scores.items()])[-1]
    return calibration
            

def LocalFeatureRefinement(
                            rounds,
                            shift,
                            imsize = (512,512),
                            section = 4.0
                           ):
    """this function takes the dataset and performs a local shifting procedure
    a few pixels up, down and sideways when the alignment is done we obtain a
    final concentration
    INPUTS:
        the rounds containing the information of the overlap between cells per round DICT
        the shift a DICT with all the coordinates of overlapping cells"""
    xsize,ysize = imsize
    
    """the rounds and the coordinated set"""
    rID   = list(rounds.keys())
    cID   = list(rounds[rID[-1]].channels.keys())
    x,y,z = zip(*rounds[rID[-1]].channels[cID[-1]].coordinates)
    
    """the coordinated middle ground of the image"""
    mx = max(x)
    my = max(y)
    mz = max(z)
    
    """extract the coordinates for all 3 channels"""
    coordinates = {((min(rID),r),c):[] for r in rID for c in cID}
    for r,rnd in rounds.items():
        for c,obj in rnd.channels.items():
              datapoints = [obj.coordinates[i] for i in range(len(obj.coordinates)) if obj.bool[i] == True]  
              coordinates[((min(rID),r),c)] = copy.copy(datapoints)

    """base matrix for the 3 channels"""
    basematrix = {}
           
    """the x,y,z direction, we overlay all the color channels"""
    reference = numpy.zeros((xsize,ysize,mz*2))
    for channel in cID:
        for x,y,z in coordinates[((min(rID),min(rID)),channel)]:
            if x > int(xsize/section) and x < xsize-int(xsize/section) and y > int(ysize/section) and y < ysize-int(xsize/section):   
                reference[x,y,z] = 1
                n = neighbors((x,y))
                for i in [-1,0,1]:
                    zn = z + i
                    for c in n:
                        xn,yn = c
                        reference[xn,yn,zn] = 1
                        
    calibration = []
    """hardcoded approach to create a local shift"""
    #parameterize of find better solution for this
    for x in range(-6,6,1):
        for y in range(-6,6,1):
            calibration.append((x,y,0))
           
    """store the matrices"""
    matrices = {}      
            
    progress = 0
    for r in rID:
        if r != min(rID):
            matrices[r] = {}
            for calshift_x,calshift_y,calshift_z in calibration:  
                """track the progress of adding pixels"""
                if progress%int((len(calibration)*len(rID))/100) == 0:
                    print('\rAligning Sample at [%d%%]'%int(progress/float(len(calibration)*(len(rID)-1))*100), end="") 
                    
                """the shift in the images"""
                xs,ys,zs  = shift[(min(rID),r)]
                
                """the coordinates in the set of reactions"""
                xs += calshift_x
                ys += calshift_y
                zs += calshift_z

                """comparison matrix"""
                comparison = numpy.zeros((xsize,ysize,mz*2))
                for channel in cID:
                    for x,y,z in coordinates[((min(rID),r),channel)]:
                        """the shift in the data""" 
                        x += ys
                        y += xs
                        z -= zs
                        
                        """we do not need all the spots in the dataset, it would take too long, its a quick alignment"""
                        if x > int(xsize/section) and x < xsize-int(xsize/section) and y > int(ysize/section) and y < ysize-int(xsize/section) and z > mz*0.3 and z < mz-0.3*z:    
                            comparison[x,y,z] = 1
                            n = neighbors((x,y))
                            
                            """base coordinates plus neighbors"""
                            for i in [-1,0,1]:
                                zn = z + i
                                for c in n:
                                    xn,yn = c
                                    comparison[xn,yn,zn] = 1
                                    
                

                """add to matrices to the list"""
                matrices[r].update({(xs,ys,zs):numpy.sum(numpy.logical_and(reference == 1, comparison == 1))})
                                 
                """update print statement"""
                progress += 1
        
    """get the shift with best overlap"""
    fittest = {}

    for r,fit in matrices.items():
        fittest[(min(rID),r)] = max(fit, key=fit.get)
        
    """add the reference"""
    fittest[(min(rID),min(rID))] = (0,0,0)
    
    return fittest,matrices

def OverlapScore(
                shift,
                benchmark,
                comparison
                ):
    """INPUT:
        shift: the function takes the shift coordinates x,y,z as INT in TUPLE
        benchmark: the image we are using as reference ARR (512x512 default)
        comparison: the image we are trying to align to the reference ARR (512x512 default)"""
        
    """the shift coordinates x,y and z denoted as columns and rows"""
    columnshift,rowshift,zshift = shift
    
    """calculate the overlap between the different nuclei"""
    columnshift = int(columnshift)
    rowshift    = int(rowshift)
    zshift      = int(zshift)

    """get 40-66% Z-stack"""
    low,high = (int(len(benchmark)*0.65)-5,int(len(benchmark)*0.65)+5)
    
    sumset = []
    for z in range(low,high,1):
        if z in benchmark.keys() and z+zshift in comparison.keys():
            """the benchmark"""
            b = numpy.array(benchmark[z]  ,dtype = bool)
            """the comparison"""
            c = numpy.array(comparison[z+zshift], dtype = bool)

            """the binary set of the nuclei, to be plotted"""
            stack = numpy.zeros((b.shape))
            rl,cl = b.shape
            for row in range(len(c)):
                for column in range(len(c[row])):
                    if row + rowshift >= 0 and row + rowshift < 512:
                        if column + columnshift >= 0 and column + columnshift < 512:
                            stack[row+rowshift,column+columnshift] = c[row,column] 
            
            """the sum of the overlap between benchmark and set"""
            stack[rl-75:rl,0:rl] = 0
            stack[0:rl,rl-75:rl] = 0
            stack[0:75,0:rl] = 0 
            stack[0:rl,0:75] = 0             

            """the sum of the overlap between benchmark and set"""
            b[rl-75:rl,0:rl] = 0
            b[0:rl,rl-75:rl] = 0
            b[0:75,0:rl] = 0 
            b[0:rl,0:75] = 0    
            
            sumset.append(numpy.sum(numpy.logical_and(stack == 1, b == 1)) - numpy.sum(numpy.logical_and(stack == 1, b == 0)) - numpy.sum(numpy.logical_and(stack == 0, b == 1))) 
        else:
            sumset.append(-512*512)
            
    """return a score"""
    return sum(sumset)/len(sumset)   
                        
                       
def AlignFeatures(
                  benchmark,
                  comparison,
                  b_image,
                  c_image
                  ):
    
    """the keys in the set"""
    keys   = [int(i) for i in benchmark.keys() if i in list(comparison.keys())]
    """take the z-slices locally, don't try to find alignments within the entire image that takes too long"""
    keys   = [z for z in keys if z < 0.99*max(keys) and z > 0.7*max(keys)]
    """create selection of image parts i.e. nuclei or spots to compare"""
    select = [keys[i] for i in numpy.round(numpy.linspace(0, len(keys) - 1, 3)).astype(int)]
    
    """the popbin in the set"""
    popbin,structures = [],[]
    """the algorithm that checks the alignment between species"""
    for z in sorted(select):
        for b in benchmark[z]:
            for s in range(-6,6,1):
                if z + s >= 0 and z+s in keys:
                    for c in comparison[z+s]:
                        if sorted([str(b),str(c)]) not in popbin:
                            popbin.append(sorted([str(b),str(c)]))
                            structures.append(StructureComparison(b,c))

    """sort sets with low scores"""
    b = numpy.argsort(numpy.array([i.lsq for i in structures]))

    shiftset = []
    for i in range(10):    
        try:
            """structures in the set that is shifted"""
            shiftset.append(structures[b[i]].shift)
            print(structures[b[i]].shift)
        except IndexError:
            pass
        
    """the coordinates that shift the image shiftset"""
    x,y,z = zip(*shiftset)
    
    """roundset of the set"""
    round_x = [round(number/10)*10 for number in x]
    round_y = [round(number/10)*10 for number in y]
    
    """bins"""
    bins = [10*i for i in range(52)]
    for i in range(52):
        bins.append(-10*i)
        
    """neighbors in the set"""
    neighbors = {i:(i-10,i+10) for i in bins}
    
    """collections in bins"""
    collection = {}
    """rouding the set"""
    for i in range(len(round_x)):
        r = round_x[i]
        low,high = neighbors[r]
        
        if int(r) in collection.keys():
            collection[r].append(i)
        else:
            """the low and high sets"""
            collection[r] =  []    
            collection[r].append(i)  
            
        if int(low) in collection.keys():
            collection[low].append(i) 
        else:
            collection[low] = []   
            collection[low].append(i)
            
        if int(high) in collection.keys():
            collection[high].append(i)
            collection[high] = []    
            collection[high].append(i)
        
    setrange,index = 0,0
    """find the highest count"""
    for k,v in collection.items():
        if len(v) > setrange:
            setrange = len(v)
            index    = copy.copy(k)

    """the shift averaged out"""
    shift   = [shiftset[i] for i in collection[index]]
    x,y,z   = zip(*shift)
    
    """append it to the shiftset"""
    shiftset.append((int(sum(list(x))/float(len(x))), int(sum(list(y))/float(len(y))),int(sum(z)/float(len(z)))))

    scores = {}
    """failsafe, if the method found the wrong alignment"""
    for sft in shiftset:
        for shiftaddition in range(-2,3,1):
            x,y,z = sft
            s = (int(x),int(y),int(z + shiftaddition))
            if abs(int(x)) + abs(int(y)) > 125:
                OS = 0
            elif s not in scores.keys():
                OS = OverlapScore(s,b_image,c_image)
                scores[s] = OS

    """fitshifts in the dataset"""
    shifts, score     = zip(*scores.items())
    x_ave,y_ave,z_ave = shifts[list(score).index(max(list(score)))]
    
    
    """average shift between two rounds of FISH"""
    return x_ave,y_ave,z_ave,shift

class SpotFitMetric:
    def __init__(self,
                 rnd,channel,metric,baseshift,calibration,correction,benchmark,reference
                 ):
        """quantify the degree to which the spots between rounds overlap, though we can assume
        new targets are found each round for some targets a new probe will be bound at the exact same
        coordinates in the image. Thus , if we use this logic we can van an overlap between images
        which accounts for this. Its can be seen as random overlap always present
        i.e. base overlap + overlap present due to proper alignment"""
        
        """round and channel"""
        self.r = rnd
        self.c = channel
        
        """the shift and matrices"""
        self.shift       = baseshift
        self.metric      = metric
        self.calibration = calibration
        self.correction  = correction
        
        """the calculated overlap"""
        self.score = intersect_matrices(benchmark,reference)
        
class StructureComparison:
    def __init__(self,
                 b,c
                 ):
        
        """compare features between structures to see
                if we have a hit...
                b = object b
                c = object c
               they are segmentaion objects"""
        
        from skimage.transform import resize
        self.b  = numpy.array(b.coordinates)
        self.c  = numpy.array(c.coordinates)

        """warp and shape of comparison"""
        wb = b.warped
        wbs = numpy.array(wb).shape
        width_b,height_b = wbs
        
        """warp and shape of benchmark"""
        wc = c.warped
        wcs = numpy.array(wc).shape        
        width_c,height_c = wcs
        
        """the height and width in the """
        self.dw = abs(width_b - width_c)/float(width_b)
        self.dh = abs(height_b - height_c)/float(height_b) 
        
        """the lsq"""
        self.lsq = numpy.sum((resize(wb,(30,30)) - resize(wc,(30,30)))**2)

        """get the shift"""
        self.shift      = tuple(self.b-self.c)
        self.z_shift    = self.shift[2]
        
        """get the distance and z_distance for the set"""
        self.distance   = math.sqrt((self.b[0] - self.c[0]) **2 + (self.b[1] - self.c[1])**2)
        self.z_distance = math.sqrt((self.b[0] - self.c[0]) **2 + (self.b[1] - self.c[1])**2 + (self.b[2] - self.c[2])**2)

      
class ImageFeatures:
    def __init__(self,z,warped,label,coordinates,cutout,image):
        """the z-stack of the image"""
        self.z = z 
        self.warped = warped
        self.label = label
        self.coordinates = coordinates
        self.cutout = cutout
        
        self.signal = {}
        """get the signal"""
        for x,y,z in self.label:
            self.signal[(x,y)] = image[x,y]

class ImageFeature:
    def __init__(self,z,warped,coordinates,image):
        """the z-stack of the image"""
        self.z = z 
        self.warped = warped
        self.coordinates = coordinates
        self.image = image
        
        self.signal = {}
        """get the signal"""
        for x,y,z in coordinates:
            self.signal[(x,y)] = image[x,y]
            
class StructureComparison:
    def __init__(self,b,c,z = 28):
        from skimage.transform import resize
        self.benchmark  = b
        self.comparison = c

        """warp and shape of comparison"""
        wb = b.warped
        wbs = numpy.array(wb).shape
        width_b,height_b = wbs
        
        """warp and shape of benchmark"""
        wc = c.warped
        wcs = numpy.array(wc).shape        
        width_c,height_c = wcs
        
        """the height and width in the """
        self.dw = abs(width_b - width_c)/float(width_b)
        self.dh = abs(height_b - height_c)/float(height_b) 
        
        """the lsq"""
        self.lsq = numpy.sum((resize(wb,(30,30)) - resize(wc,(30,30)))**2)
        try:
            self.lsq_cutout = numpy.sum((resize(b.cutout,(30,30)) - resize(c.cutout,(30,30)))**2)
        except:
            self.lsq_cutout = False
        
        """get the shift"""
        self.shift    = (self.benchmark.coordinates[0] - self.comparison.coordinates[0],self.benchmark.coordinates[1] - self.comparison.coordinates[1])
        self.distance = math.sqrt((self.benchmark.coordinates[0] - self.comparison.coordinates[0]) **2 + (self.benchmark.coordinates[1] - self.comparison.coordinates[1])**2)
        
        




