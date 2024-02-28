# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 14:38:35 2021

@author: bob van sluijs
"""

from PIL import Image, ImageEnhance
import numpy
from scipy.spatial import distance
import os
from scipy import fftpack
from matplotlib.colors import LogNorm
from scipy import ndimage
from skimage.feature import peak_local_max
import itertools as it
from skimage.morphology import extrema
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from shapely.geometry import Polygon
from scipy.signal import argrelextrema
import matplotlib.pylab as plt
import copy
import scipy
import numpy as np
from scipy.ndimage import gaussian_filter
import scipy as sp
import scipy.ndimage
from skimage.segmentation  import watershed
import math
# import open3d as o3d

def OpenImage(path,stack = 250): 
    """open the images"""
    image  = Image.open(path)
    """store the arrays"""
    arrays = []
    """get the z stacks out of the tiff file"""
    for n in range(0,stack,1):
        try:
            image.seek(n)
        except:
            break           
        """arrays with different z stacks"""
        arrays.append(copy.copy(numpy.array(image)))
    return arrays

def OpenSlicedImage(path,stack = 250):
    """open the images"""
    image  = Image.open(path)
    """store the arrays"""
    arrays = []
    """get the z stacks out of the tiff file"""
    for n in range(0,stack,1):
        try:
            image.seek(n)
        except:
            break           
        """arrays with different z stacks"""
        arrays.append(copy.copy(numpy.array(image)))
    """the arrays in the dataset"""
    arrays = numpy.array([arrays[int(len(arrays)/1.5)-1],arrays[int(len(arrays)/1.5)],arrays[int(len(arrays)/1.5) + 1]])
    return arrays

def neighbors(index):
    N = len(index)
    import itertools as it
    for relative_index in it.product((-1, 0, 1), repeat=N):
        if not all(i == 0 for i in relative_index):
            yield tuple(i + i_rel for i, i_rel in zip(index, relative_index))

def findpaths(folder):
    paths = [folder + '\\' + i for i in  os.listdir(folder)]
    return paths


def resize_all(src, pklname, include, width=25, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata. The dictionary is written to a pickle file 
    named '{pklname}_{width}x{height}px.pkl'.
     
    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
        
    Modified by Bob van Sluijs from stackoverflow 55029388
    """
    import os
    import joblib
    from skimage.io import imread
    from skimage.transform import resize
    import numpy
    

    height = height if height is not None else width
    data = dict()
    data['description'] = 'resized ({0}x{1})animal images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['images'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"
    print(width,height)
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        current_path = os.path.join(src, subdir)
        im = imread(current_path,as_gray=True)
        im = resize(im, (width, height)) #[:,:,::-1]
        data['label'].append(subdir[:-4])
        data['filename'].append(current_path)
        data['images'].append(im/numpy.max(im))
    return data


def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1
  
    
def flood_fill(test_array,h_max=255):
    input_array = np.copy(test_array) 
    el = sp.ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = sp.ndimage.binary_erosion(~np.isnan(input_array), structure=el)
    output_array = np.copy(input_array)
    output_array[inside_mask]=h_max
    output_old_array = np.copy(input_array)
    output_old_array.fill(0)   
    el = sp.ndimage.generate_binary_structure(2,1).astype(np.int)
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(input_array,sp.ndimage.grey_erosion(output_array, size=(3,3), footprint=el))
    return output_array

def boxcentroid(box): 
    x = numpy.average(box[:,0])
    y = numpy.average(box[:,1])    
    return (x,y)

def distpoints(x_1 , x_2, y_1 , y_2):
    return math.sqrt((x_1 - x_2)**2 + (y_1 + y_2)**2)
      
def check_for_overlap(box_1, box_2):
    return True
    x_1 = sum(list(box_1[:,0])) /4.
    y_1 = sum(list(box_1[:,1])) /4. 
    x_2 = sum(list(box_2[:,0])) /4.
    y_2 = sum(list(box_2[:,1])) /4. 
    
    cdist   = distpoints(x_1 , x_2, y_1 , y_2)
    if cdist < 150:
        return True
    else:
        return True

def calculate_iou(box_1, box_2):
    """check if boxes have overlap to begin with"""
    boolean = check_for_overlap(box_1,box_2)
#    print(boolean)
    """use average polygon sets"""
    iou = 0
    if boolean == True:
        poly_1 = Polygon(box_1)
        poly_2 = Polygon(box_2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def random_color():
     import random
     rgbl=[random.choice(range(255)),random.choice(range(255)),random.choice(range(255))]
     random.shuffle(rgbl)
     return tuple(rgbl)
 
def euclidian_distance(average,c):
    """unpack coordinates"""
    x,y,z = c
    x_average,y_average,z_average = average
    return math.sqrt(((x-x_average)**2+(y-y_average)**2))

def distance3D(a,b):
    xj,yj,zj = a
    xi,yi,zi = b
    """give two 3D coordinates and obtain distance"""
    return math.sqrt(((xi,xj)**2+(yi,yj)**2) + (zi,zj)**2)


def CheckPoint(box,coordinate):
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    point = Point(coordinate[0],coordinate[1])
    polygon = Polygon(tuple([(i[0],i[1]) for i in box]))
    boolean = polygon.contains(point)
    if boolean == True:
        print()
    return boolean

def pointcheck(hull,points, p):
    from matplotlib.tri import Triangulation
    from scipy.spatial import ConvexHull  
    """take the coordinate vectors and coordinate to test and unpack"""
    X,Y,Z = points
    x,y,z = p
    """append the new coordinates"""
    X.append(x)
    Y.append(y)
    Z.append(z)
    """update the convex hull for this coordinate set, and see if more vertices are added
    if its outside the current structure a new vertix is added if inside than it should be equal"""
    new_hull = ConvexHull(numpy.array([X,Y,Z]).T)
    if list(hull.vertices) == list(new_hull.vertices):
        return True
    else:
        return False

def averagesets(clist):
    distances = {}
    
    """remove 20% of the outliers"""
    x,y,z = zip(*clist)
    x_average = 0
    y_average = 0
    for i in range(len(x)):
        x_average += x[i]
        y_average += y[i]
    x_average = x_average/float(len(x))
    y_average = y_average/float(len(y))
    for x,y,z in clist:
        distances[math.sqrt(((x-x_average)**2+(y-y_average)**2))] = (x,y)
    length = int(0.2*len(distances))
    distlistsort = list(reversed(sorted(list(distances.keys()))))
    
    temp  = []
    for i in range(length):
        temp.append(distlistsort[i])
    pruned = []
    for d,c in distances.items():
        if d in temp:
            pass
        else:
            pruned.append(c)
            
    x,y = zip(*pruned)
    x_average = 0
    y_average = 0
    for i in range(len(x)):
        x_average += x[i]
        y_average += y[i]
    x_average = x_average/float(len(x))
    y_average = y_average/float(len(y))
    return pruned,(x_average,y_average)

def prunecentroids(centroids):
    distance = {}
    for x,y in centroids:
        for xi,yi in centroids:
            if (xi,yi) == (x,y):
                pass
            else:
                dist = math.sqrt(((x-xi)**2+(y-yi)**2))
                try:
                    t = distance[((xi,yi),(x,y))]
                except:
                    distance[((x,y),(xi,yi))] = copy.copy(dist)
                    
    maxdist = []
    for c,dist in distance.items():
        if dist < 10:
            maxdist.append(copy.copy(c))
        else:
            pass
        
    keep,remove = [],[]
    for c1,c2 in maxdist:
        keep.append(c1)
        remove.append(c2)
    clist = [i for i in centroids if i not in remove]
    return clist

def get_filenames(folder):
    import os
    filenames = os.listdir(folder)
    paths = [] 
    for i in filenames:
        paths.append((i,folder+i))
    return paths,filenames    

def demogif(folder,images,coordinates = []):
    import os
    try:
        os.mkdir(folder)
    except:
        pass
    for i in range(len(images)):
        plt.figure(figsize=(10,10))
        plt.imshow(images[i],cmap = 'jet')
        plt.savefig(folder + str(i) + '.png',dpi = 400)

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
    
def plot_3D(array,matrix,alpha = 0.5):
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca(projection='3d')
    array = numpy.array(array,dtype = bool)
    ax.voxels(array,alpha = alpha)
    z,x,y = matrix.shape
    ax.set_xlim(0,25)
    ax.set_ylim(0,512)
    ax.set_zlim(0,512)
    ax.view_init(0, 0)
#    ax._axis3don = False
    plt.show()   
    return   

def CodeDistanceScheme(codebook):
    codebook = list(codebook.keys())
    """find everything that is not a number and not a round"""
    letters = []
    for code in codebook:
        for i in code:
            if i != 'R':
                if i.isnumber == False:
                    letters.append(i)
                    
    """get the letterset"""    
    letterset = list(set(letters))
    
    removed  = {}
    """get the individual letterset"""
    for code in codebook:
        removed[i] = []
        for letter in letterset:
            indices = [pos for pos, char in enumerate(s) if char == letter]
            """copy code of indices"""
            for index in indices:
                p = copy.copy(code)
                del p[index]
                removed[code].append(copy.copy(p))
                
    inverted = {}
    for code,bastardised in removed.items():
        for c in bastardised:
            try:
                inverted[c].append(code)
            except:
                inverted[c] = []
                inverted[c].append(code)

    """find the set one removed codes from each other"""
    k_connected = {}
    
    """the code"""
    for code,removed in removed.items():
        k_connected[code] = []
        for i in removed:
            try:
                c = codebook[i]
                k_connected[c].append(copy.copy(code))
            except:
                pass
            
    return removed,inverted,k_connected
                
def ImportCodeScheme(path,codemap = {}):
    """function takes the identifiers listed in your codemap"""
    colors  = {"G":'Atto',"B":"Cy5","R":'Tm'}
    
    """Path of the codebook"""
    import pandas as pd
    df = pd.read_excel(path)
    flourophores = {v:k for k,v in colors.items()}
    codedict = df.to_dict()
    spots = df["Class"]
    rounds = len(codedict) 
    r = ["R{}".format(i) for i in range(1,rounds,1)]
    
    """function that creates the code """
    codebook = {}

    for number,spotID in spots.items():
        code = ''
        for rnd in r:
            if codedict[rnd][number] != '-':
                sortstring = ''
                for i in sorted(codedict[rnd][number]):
                    sortstring += i
                code += rnd + sortstring
                code += '_'
                
        if code[-1] == '_':
            code = code[0:-1]
        codebook[spotID] = code
        
    return codebook

def ColorMarkerCombinations():
    import itertools
    """create a large set of plot markers to plot dots in the image"""
    plotitems = [':','.',',','o','v','^','<','>','1','2','3','s','p','*','h','H','+','x','D','d','|','_']
    colors    = ['b','g','r','c','m','y','k','w']
    items     = []
    for i in colors:
        for j in plotitems:
            items.append(i+j)
    return items,colors
    
def neighbors(index):
    N = len(index)
    for relative_index in it.product((-1, 0, 1), repeat=N):
        if not all(i == 0 for i in relative_index):
            yield tuple(i + i_rel for i, i_rel in zip(index, relative_index))
    
#read the image
def get_filenames(folder):
    filenames = os.listdir(folder)
    paths = [] 
    for i in filenames:
        paths.append((i,folder+i))
    return paths,filenames
   
def plot_spectrum(im_fft):
    from matplotlib.colors import LogNorm
    plt.imshow(numpy.abs(im_fft), norm=LogNorm(vmin=5))
    plt.colorbar()

def vis_pc(xyz, color_axis=-1, rgb=None):
    # TODO move to the other module and do import in the module
    import open3d
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(xyz)

    if color_axis >= 0:
        if color_axis == 3:
            axis_vis = numpy.arange(0, xyz.shape[0], dtype=np.float32)
        else:
            axis_vis = xyz[:, color_axis]
        min_ = numpy.min(axis_vis)
        max_ = numpy.max(axis_vis)

        colors = cm.gist_rainbow((axis_vis - min_) / (max_ - min_))[:, 0:3]
        pcd.colors = open3d.Vector3dVector(colors)
    if rgb is not None:
        pcd.colors = open3d.Vector3dVector(rgb) 
    open3d.draw_geometries([pcd])
 
def RemovedNeighbor(n):
    x,y = n
    """initial neighbors"""
    cns = neighbors((x,y))
    """neirest neighbors"""
    ngb = [i for i in cns]
    ngb.append((x,y))
    second = []
    for x,y in ngb:
        n = neighbors((x,y))
        for l in n:
            second.append(l)
    ngb += second
    return list(set(ngb))

def SequenceViolinPlot(spots):
    import pandas as pd
    import seaborn as sns
    """take spot sequence"""
    sequencelist = [i.list for i in spots.values()]
    """take the list and dataframe it"""
    array   = numpy.array(sequencelist)
    """the s-slit variables"""
    columns = [f'Round_{num}' for num in range(1,len(sequencelist[0])+1,1)]
    """create frame"""
    df = pd.DataFrame(array, columns=columns)
    """the violinplot"""
    plt.figure(figsize=(10,10))
    plt.xlabel('N',size = 16)
    plt.ylabel("Pixel Value",size = 16)
    plt.title("Pixel Intensity Density (including 7 z-stacks)",size = 20)
    sns.violinplot(data=df)
    plt.show()
    
def OpenImage(path,stack = 250):
    """open the images"""
    image  = Image.open(path)
    """store the arrays"""
    arrays = []
    """get the z stacks out of the tiff file"""
    for n in range(0,stack,1):
        try:
            image.seek(n)
        except:
            break           
        """arrays with different z stacks"""
        arrays.append(copy.copy(numpy.array(image)))
    return arrays
        
def create_shifted_matrix(matrix,coordinates,coordinate):
    benchmark = numpy.zeros((len(matrix),len( matrix)))
    """the ymin and xmin coordinate"""
    xmin,ymin = coordinate
    """the coordinates for the shift"""
    shiftC = []
    for x,y in coordinates:
        xn = x + xmin
        yn = y + ymin
        shiftC.append((copy.copy(xn),copy.copy(yn)))
    shiftC = [(x,y) for x,y in shiftC if x < len(matrix) and x > -1 and y > -1 and y < len(matrix)] 
    for x,y in shiftC:
        benchmark[x,y] = 1
        for i in neighbors((x,y)):
            xi,yi=  i
            if xi < len(matrix) and xi > -1 and yi > -1 and yi < len(matrix):
                benchmark[xi,yi] = 1
    return benchmark

def running_mean(x, N):
    cumsum = numpy.cumsum(numpy.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_overlapping_signals(signals,imsize = (512,512)): 
    '''plot the signal overlap to check if the shift works'''
    z_stack,scatter = {},{}
    for i in signals:
        try:
            z_stack[i.z][i.c] += 1
            scatter[i.z].append((i.round,i.channel,i.c))
        except:
            """sort by scatter"""
            scatter[i.z] = []
            scatter[i.z].append((i.round,i.channel,i.c))
            """sort by array"""
            z_stack[i.z] = numpy.zeros(imsize)
            z_stack[i.z][i.c[0],i.c[1]] += 1            
        
    color = ['yellow','b','r','g','purple','r']
    for z,s in scatter.items():
        plt.figure(dpi = 600)
        r,channel,c = zip(*s)
        for i in range(len(r)):
            if r[i] == 1:
                plt.scatter(c[i][0],c[i][1],c = color[int(channel[i])],s = 1,alpha = 0.33)
        plt.show()
        
   
def plot_signal_overlap(signals,alignment_path,imsize = (512,512)):
    '''plot the signal overlap to check if the shift works'''
    z_stack,scatter = {},{}
    
    for i in signals:
        try:
            z_stack[i.z][i.c] += 1
            scatter[i.z].append((i.round,i.channel,i.c))
        except:
            """sort by scatter"""
            scatter[i.z] = []
            scatter[i.z].append((i.round,i.channel,i.c))
            """sort by array"""
            z_stack[i.z] = numpy.zeros(imsize)
            z_stack[i.z][i.c[0],i.c[1]] += 1            
        
    try:
        for reference in range(2,6):
            print(reference,alignment_path)
            for z,s in scatter.items():
                if int(len(z_stack)*0.65) == z:
                    r,channel,c = zip(*s)
                    for i in range(len(r)):
                        if r[i] == 1:
                            plt.scatter(c[i][0],c[i][1],c = 'r',s = 1,alpha = 0.5)
                        if r[i] == reference:
                            plt.scatter(c[i][0],c[i][1],c = 'b',s = 1,alpha = 0.5)
            plt.savefig(alignment_path + '\\' + str(1) + '&' + str(reference) + '.png',dpi = 750)
            plt.close()
    except:
        print("shift images already exist in the dataset")


def cross_image(im1, im2):
   # get rid of the color channels by performing a grayscale transform
   # the type cast into 'float' is to avoid overflows
   im1_gray = np.sum(im1.astype('float'), axis=2)
   im2_gray = np.sum(im2.astype('float'), axis=2)

   # get rid of the averages, otherwise the results are not good
   im1_gray -= np.mean(im1_gray)
   im2_gray -= np.mean(im2_gray)

   # calculate the correlation image; note the flipping of onw of the images
   return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same')

def neighbors(index):
    N = len(index)
    for relative_index in it.product((-1, 0, 1), repeat=N):
        if not all(i == 0 for i in relative_index):
            yield tuple(i + i_rel for i, i_rel in zip(index, relative_index))
            
            
def fill_contours(arr):
    return numpy.maximum.accumulate(arr,1) & \
           numpy.maximum.accumulate(arr[:,::-1],1)[:,::-1]
      
            
def ParsePath(path,channel):
    """ get the image """
    if '\\' in path:
        split = path.split('\\')
    else:
        split = path.split('$\$')
        
    information = {identifiers[i]:[] for i in range(len(identifiers))}
    """ get the information from the class """
    for ids in range(len(data)):
        for item in range(len(data[ids])):
            information[identifiers[ids]].append(data[ids][item])
    path = path.replace(split[-1],'')


def OverlayGaussian(coordinates,gaussian):
    """the coordinate zet"""
    x,y,z = zip(*coordinates)
    x_max = numpy.amax(x)
    y_max = numpy.amax(y)    
    z_max = numpy.amax(z)    
    
    """create the matrix"""
    slate = numpy.zeros((512,512,z_max+2))
    
    xs,ys       = gaussian[-1].shape
    xhalf,yhalf = int((xs-1))/2,int((ys-1)/2) 
    for i in range(len(coordinates)):   
        x,y,z = coordinates[i]
        slate[int(x-xhalf):int(x+xhalf)+1,int(y-yhalf):int(y+yhalf)+1,z] = gaussian[i]
    return slate

def circle(arr,x,y,radius = 40):
    from skimage import draw
    """the perimeter"""
    rr, cc = draw.circle_perimeter(int(x), int(y), radius=int(radius), shape=arr.shape)
    """bool activation of the cell"""
    arr[rr, cc] = 1
    """we have the perimiter now we still need to fill the perimiter"""
    arr = numpy.array(arr,dtype = 'int')
    """filled array"""
    arr = fill_contours(arr)    
    return arr