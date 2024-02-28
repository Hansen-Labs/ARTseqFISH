# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 17:15:27 2022

@author: Bob van Sluijs
"""
from ImageAnalysisOperations import *
import skimage
from scipy.spatial.ckdtree import cKDTree
from matplotlib.tri import Triangulation
from scipy.spatial import ConvexHull 
import networkx as nx
import pandas as pd
import cv2

def LDtoDL(LD):
    nd={}
    for d in LD:
        for k,v in d.items():
            try:
                nd[k].append(v)
            except KeyError:
                nd[k]=[v] 
    return nd

def binarize_image(b):
    import skimage
    import numpy
    """the limit in the race"""
    gray     = numpy.array(b/numpy.amax(b)*255,dtype = numpy.uint8)
    gray     = skimage.morphology.area_opening(gray)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(9,9))
    equalized = clahe.apply(gray)
    thresh = cv2.threshold(equalized, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] 

    """initial thresholds, fill holes, find connected components"""
    init_thresh = skimage.morphology.remove_small_objects(thresh,min_size=125)
    thresh = ndimage.binary_fill_holes(init_thresh).astype(int)
    diff   = numpy.array((thresh - numpy.array(init_thresh,dtype = bool))*255,dtype = numpy.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(diff, connectivity=4) 
    sizes = stats[1:, -1]
    image = numpy.zeros((diff.shape))
    if len(sizes):
        for i in range(0, nb_components-1):
            if sizes[i] <=  250:
                image[output == i + 1] = 255 
    
    image  +=  init_thresh
    return image

def build_coordinate_graph(c,distance = 1):
    "1) build k-d tree"
    kdt = cKDTree(numpy.array(c))
    edges = kdt.query_pairs(distance)
    
    "2) create graph"
    G = nx.from_edgelist(edges)

    "3) Find Connections"
    ccs = nx.connected_components(G)
    node_component = {v:k for k,vs in enumerate(ccs) for v in vs}
    df = pd.DataFrame(c, columns=['x','y','z'])
    df['c'] = pd.Series(node_component)        
    
    "4) unpack and extract features, sort according to integer ID"
    sets = {}
    for node,label in node_component.items():
        if label in sets:
            sets[label].append((df['x'][node],df['y'][node],df['z'][node]))
        else:
            sets[label] = [(df['x'][node],df['y'][node],df['z'][node])]
            
    flattened = [i for j in sets.values() for i in j]
    for i in c:
        if i not in c:
            sets[len(sets)] = [i]
            
    return sets
            
def calculate_distance(a,b):
    dist = math.sqrt(numpy.sum((numpy.array(a)-numpy.array(b))**2))
    return dist
    

def ReconstituteCell(data):
    cells = CellSetFeatures()
    cells.initiate_cellfeatures(data)
    cells.initiate_DAPI(data)
    return cells


def pairwise_distance(n,m):
    """pairwise distance between each element in two sets of coordinates
    IN: list [] with tuple coordinates
    OUT: sets of coordinate pairs closes to one another"""
    distance = {}
    for i in n:
        """first x,y coordinate"""
        x,y = i
        """create a seperate dictionary"""
        distance[i] = {}
        for j in m:
            xs,ys = j
            distance[i][j] = math.sqrt((x-xs)**2 + (y-ys)**2)  
            
    pairwise_set = {}
    """cluster them in a set"""
    for i,j in distance.items():
        pairwise[i] = []
        for crd,v in j.items():
            if v == min(list(j.values())):
                pairwise[i].append(j)
    
    """average out the set"""
    for c,neighboring in pairwise.items():
        x,y = zip(*neighboring)
        pairwise[c] = (numpy.average(numpy.array(x)),numpy.average(numpy.array(y)))

    return pairwise
  
def interpolate_contours(c,l):
    """we interpolate the contours for missing z stack and approximate what
    the cell contours should be
    IN: c = dict(with contour coordinates)
        l = tuple with z-range that needs to be filled up (z(n),z(n+m)) where n-m represents the gap in segmented knowledge
    OUT: a dict with an array per z-stack"""
    from skimage import feature
    from skimage.segmentation import flood, flood_fill
        
    """add the image coordinates to the z-stack"""

    cells = {}
    """the intset"""
    for intset in l:
        start,end  = intset
        """the difference between start and end"""
        difference = start-end
        
        """get the binary float difference"""
        n_top    = c[start]
        n_bottom = c[end]
        
        """cell pair"""
        pair = [n_top,n_bottom]
        
        """get their size"""
        xt,yt = n_top.nonzero()
        size_top = len(xt)
        xb,yb = n_bottom.nonzero()
        size_bottom = len(xb)
        pairsize = [size_top,size_bottom]


        """index of largest cell"""
        index          = pairsize.index(max([size_top,size_bottom]))
        reference_cell = pair[index] 
        
        """get m-start to m-bottom, and copy coordinates largest cell"""
        xd,yd = (numpy.array(n_top,dtype = float) - numpy.array(n_bottom,dtype = float)).nonzero()
        pixel_average = len(xd)/2
        xr,yr = reference_cell.nonzero()
        rc = copy.copy(reference_cell)
        
        """small while loop to get the removed pixel number set"""
        stripped_pixels = 0
        while stripped_pixels < pixel_average:
            contour = []
            for i in range(len(xr)):
                try:
                    if all([rc[x,y] for x,y in neighbors((xr[i],yr[i]))]) != 1:
                        contour.append((xr[i],yr[i]))
                except IndexError:
                    pass
                    
            """add striped pixels to the dataset"""
            for x,y in contour:
                rc[x,y] = 0
            stripped_pixels += len(contour)  

        for z in range(start+1,end,1):
            cells[z] = copy.copy(rc)
            
    cells.update(c)
    return cells

def get_array_coordinates(c):
    """input is a dictionary with images, the key corresponds to a third z coordinates
    in: **arg dict
    key (integer)
    value (2D array)"""

    coordinates = []
    for z,coordinateset in c.items():
        x,y = coordinateset.nonzero()

        for i in range(len(x)):
            coordinates.append((x[i],y[i],z))
    return coordinates
    

def calculate_overlap(c):
    """calculate the overlap between two binary images expressed as a percentage
     in: **arg c is the array with boolean pixel blob
    out: fraction overlap between consecutive blobs"""
    f = {}
    """first check if the images overlap, filter out those that do not and move on with the rest"""
    print(len(c))
    for i in range(len(c)-1):
        f[(i,i+1)] = numpy.sum(numpy.logical_and(c[i] == True, c[i+1] == True))/(numpy.sum(numpy.logical_and(c[i] == True, c[i+1] == True)) + numpy.sum(numpy.logical_and(c[i] == True, c[i+1] == False)) + numpy.sum(numpy.logical_and(c[i] == False, c[i+1] == True)))
    return f

def filter_cell_layers(c):
    c = {k:numpy.array(v,dtype = bool) for k,v in c.items()}
    
    """get the indices that need to be deleted"""
    indices = []
    try:
        for i in range(len(c)-1):
            summed = numpy.sum(numpy.logical_and(c[i] == True, c[i+1] == True))/(numpy.sum(numpy.logical_and(c[i] == True, c[i+1] == True)) + numpy.sum(numpy.logical_and(c[i] == True, c[i+1] == False)) + numpy.sum(numpy.logical_and(c[i] == False, c[i+1] == True)) + 1)
            if summed < 0.5:
                if len(c[i].nonzero()[0]) > len(c[i+1].nonzero()[0]):
                    indices.append(i+1)
                else:
                    indices.append(i)
    except KeyError:
        pass
    
    """remove small objects"""
    cells = {}
    for index,image in c.items():
        thresh = skimage.morphology.remove_small_objects(image,min_size=300)
        if len(thresh.nonzero()[0]) == 0:
            pass
        else:
            cells[index] = image
    
    try:
        from itertools import groupby,count
        cn = count()
        lst = list(sorted([i for i in list(cells.keys()) if i not in indices]))
        indices = max((list(g) for _, g in groupby(lst, lambda x: x-next(cn))), key=len)
        cells = {i:v for i,v in cells.items() if i in indices} 
    except ValueError:
        print('Their are no cells in between index stacks')
    return cells
            
def averaged_coordinates(c):
    """this function averages the position of the contour pixels based on the current pixels set
    we do this by assigning a contemporary pixel partner i.e. every pixel is clustered based on its 
    neirest neighbor
    
     in: **arg c is the vector with coordinates of contour pixels in a list
    out: averaged out contourplot for the data""" 
    from skimage import feature
    from skimage.segmentation import flood, flood_fill
    
    f = calculate_overlap(c)
    r = [index for index,fraction in f.items() if fraction < 0.66]
    """remove the label in this index"""
    remove = []
    for x,y in r:
        remove.append(x)
        remove.append(y)

    
    """summed edge and the canny detector to get the edge of this final image"""
    summededge = sum([numpy.array(c[i],dtype = 'float32') for i in range(len(c)) if i not in remove])
    edges = feature.canny(summededge,sigma=3)
    x,y = edges.nonzero()
    nucleus = numpy.array(summededge,dtype = bool)
    
    # plt.imshow(nucleus)
    # plt.title('averaged')
    # plt.show()
    
    # print((nucleus,[(x[i],y[i]) for i in range(len(x))]))
    return (nucleus,[(x[i],y[i]) for i in range(len(x))])


def calculate_difference(ar_1,ar_2):
    """calculate the overlap between two binary arrays
    if overlap > 90% it counts and is added to the stack"""
    ar_1 = numpy.array(ar_1,dtype = bool)
    ar_2 = numpy.array(ar_2,dtype = bool)
    
    """the fraction of the arrays that overlap from the perspective of the cell slice arr_1"""
    overlap  = numpy.logical_and(ar_1 == True, ar_2 == True)
    summed_overlap = numpy.sum(overlap)
    
    mismatch = numpy.logical_and(ar_1 == True, ar_2 == False)
    summed_mismatch = numpy.sum(mismatch)
    
    """x_arr_1,y_arr_1"""
    x,y = overlap.nonzero()
    
    """get the contour 1 place outside actual contour"""
    contour = []
    for i in range(len(x)):
        n = neighbors((x[i],y[i]))
        try:
            boolean = [overlap[xn,yn] for xn,yn in n]
            if False in boolean:
                for m in n:
                    contour.append(m)
        except IndexError:
            pass
                
    """check if the overlap of the DAPI is inside! the cell label, if so automatically accepted"""          
    boolean = [arr_1[xc,yc] for xc,yc in contour]
    if False not in boolean:
        return True,overlap

    """calculate the fraction that overlaps"""
    score = summed_overlap/float(summed_mismatch+summed_overlap)
    """if 80% overlaps we can count it as part of the cell"""
    if score > 0.85:
        return True,overlap
    else:
        return False,overlap
    
    """get the sum of the overlap between states"""

def add_outer_layers(cells,DAPI):
    """this function takes two variables, the cells i.e. the 
    individual cluster of cell at the z-stack in the form of arrays in a dictionary
    
    dict: {z-stack(i):array with image containing single cel
    dict: {z-stack(i):array with full DAPI image
           
    OUT: self.cells i.e. the first input arg but this time it contains missing slices from the top and the bottom"""
    smallest,largest = min(list(cells.keys())),max(list(cells.keys()))
    
    """get the smallest and the largest dataset"""
    s = copy.copy(smallest)
    l = copy.copy(largest)
    
    """index addition i.e. the index where we duplicate the bottom or top slice or the overlapped slice"""
    for i in range(0,smallest,1):
        boolean,array = calculate_difference(cells[smallest],DAPI[i])
        if boolean:
            cells[i] = array
            smallest = i
            
    for i in range(largest,len(DAPI),1):
        boolean,array = calculate_difference(cells[largest],DAPI[i])
        if boolean:
            cells[i] = array
            largest = i
            
    from itertools import groupby,count
    c = count()
    """get the largest consecutive list of indices i.e. [1,2,3,5,6,7,8] the latter 4 would be largest"""
    indices = max((list(g) for _, g in groupby(list(sorted(cells.keys())), lambda x: x-next(c))), key=len)
    cells = {i:v for i,v in cells.items() if i in indices}

    # for k in sorted(cells.keys()):
    #     plt.title(k)
    #     plt.imshow(numpy.array(cells[k],dtype=float) *3 + numpy.array(DAPI[k],dtype = float)/numpy.amax(DAPI[k]))
    #     plt.colorbar()
    #     plt.show()
    """copy the cells at the cell z-stack"""    
    return cells
        
class CellFeature:
    def __init__(self,warped,label,coordinates,cutout,centroid,parameters = {}):
        """these classes serve as bins where we store the relevant infomration that is needed
        to obtain the distributions of targets within either the nucleus or cytoplasm, these
        can also be used to reconstruct a cell in 3D when we try for different parameters"""
        
        """the z-stack of the image"""
        self.warped      = warped
        self.label       = label
        self.coordinates = coordinates
        self.cutout      = cutout
        self.centroid    = centroid
        
        """segmentation parameters"""
        self.parameters  = parameters
        
        """coordinates of the cutout center"""
        self.x,self.y,self.z = self.coordinates
        
        
class ReconstructedNucleus:
    def __init__(self,labelmarker,labelset,raw,DAPI,SVM = '',AI = True):
        import itertools
        self.labelmarker = labelmarker
        
        """labelset contains the set of all labels"""
        self.x,self.y,self.z = [],[],[]
        """the labelset which account for the data in the set"""
        for n in labelset:
            x,y,z = zip(*n)
            for i in range(len(x)):
                self.x.append(x[i])
                self.y.append(y[i])
                self.z.append(z[i])
        
        self.sorted_labelset = {}
        for i in labelset:
            x,y,z = i[-1]
            if z not in self.sorted_labelset:
                self.sorted_labelset[z] = [i]
            else:
                self.sorted_labelset[z].append(i)

        """get the top and bottom of the cell (or atleast the detected top and bottom)"""       
        self.depth = (min(self.z),max(self.z))

        """segment not detected at this level of the z-stack"""
        missing = [i for i in range(self.depth[0],self.depth[1]+1,1) if i not in self.z]
        
        """check the overlap between labels and score it (coverage that overlaps / total coverage)
        then take the contours to fill in the cells at Z-stacks that are currently missing to avoid 
        having to perform a Qhull point detection or delauny point detection. This should work 100 times faster"""
        img = {}
        
        """get the contour of the cell then get array at this index and array at this index"""
        for z,labels in self.sorted_labelset.items():
            for i1 in range(len(labels)):
                
                il_1 = numpy.zeros((512,512),dtype = bool)
                X,Y,Z = zip(*labels[i1])
                il_1[X,Y] = True
                if z not in img:
                    img[z] = [il_1]
                else:  
                    img[z].append(il_1)

        """sort by zstack and interpolate the missing stacks, start by avereging 
        out the overlapping stacks in the dataset, then compare the ones missing"""
        averaged_contour = {}
        for z,img in img.items():
            # for i in img:
            #     plt.imshow(i)
            #     plt.show()
            try:
                averaged_contour[z] = averaged_coordinates(img)
            except:
                print("Averaged contour approximation failed")
                print(img,z)
                
        """calculate overlap between z-stack"""
        structs = []
        for i in sorted(averaged_contour.keys()):
            n,c = averaged_contour[i]
            structs.append(n)
            
        """remove the non-overlapping segments"""
        r = [index for index,fraction in calculate_overlap(structs).items() if fraction < 0.5]
        
        """the z-stacks that are linked"""
        linked = []
        
        """find missing z-stacks and link the z-stacks closest together from top to bottom"""
        for i in range(len(list(averaged_contour.keys()))-1):
            x,y = (sorted(list(averaged_contour.keys()))[i],sorted(list(averaged_contour.keys()))[i+1])
            if y-x != 1:
                if (x,y) not in r:
                    linked.append((x,y))
    
        if len(linked) == 0:
            """if this holds than all z-stacks are segmented and overlap"""
            self.cell = {}
            for z,data in averaged_contour.items():
                n,c = data
                self.cell[z] = n
            self.cell = add_outer_layers(self.cell,DAPI)
            self.cell = filter_cell_layers(self.cell)
            self.coordinates = get_array_coordinates(self.cell)
            
        else:
            self.cell = {}
            """interpolate the missing z-stacks"""
            for z,data in averaged_contour.items():
                n,c = data
                self.cell[z] = n
            self.cell = interpolate_contours(self.cell,linked)
            self.cell = add_outer_layers(self.cell,DAPI)
            self.cell = filter_cell_layers(self.cell)
            
            """get the coordinates of the cell"""
            self.coordinates = get_array_coordinates(self.cell) 

        """get the volume of the cell"""
        self.volume = len(self.coordinates)
        
        """decide later whether this cell will be accepted in the stack"""
        self.bool   = False
        
        """get the centroid of this cell"""
        self.centroid = (numpy.average(numpy.array(self.x)),numpy.average(numpy.array(self.y)),numpy.average(numpy.array(self.z)))
        
        """get the size profile of the labelset, we expect a single maximum! two
        is a problem as that indicates perfectly overlapping cells"""
        cell = numpy.array([self.cell[k] for k in sorted(self.cell.keys())])
        z,x,y = numpy.nonzero(cell)
        
        """cut the cell out from the arrays"""
        ymin,ymax   = numpy.min(y),numpy.max(y)
        xmin,xmax   = numpy.min(x),numpy.max(x)
        zmin,zmax   = numpy.min(z),numpy.max(z)
        
        """the cell in 3D"""
        cell_3D = []
        
        
        """collect the sideslice cell images and sum them to assess if they exist"""
        try:
            zcmin,zcmax = min(sorted(self.cell.keys())),max(sorted(self.cell.keys()))
            mx,my = (xmin+xmax)/2,int((ymin+ymax)/2.)
            for i in range(ymin,ymax,1):
                data = raw[zmin+zcmin:zmax+zcmax,xmin:xmax,i]/numpy.max(raw[zmin+zcmin:zmax+zcmax,xmin:xmax,i])
                if i == ymin:
                    vertical  = data
                else:
                    vertical += data
                    
                """add the slice to the 3D cell reconstruction"""
                cell_3D.append(raw[zmin+zcmin:zmax+zcmax,xmin:xmax,i])
        except:
            pass
        
        """segementataion of the nuclei"""
        gray     = numpy.array(vertical/numpy.amax(vertical)*255,dtype = numpy.uint8)
        gray     = skimage.morphology.area_opening(gray)
        initial  = copy.copy(gray)

        """check cutout of the image"""
        from PIL import Image
        from skimage.transform import resize
        import seaborn as sns
        
        """flatten the data to get distribution"""
        try:
            self.cellstate = (numpy.array(cell_3D),((xmin+xmax)/2.,(ymin+ymax)/2.,(zmin+zmax)/2.))
        except:
            self.cellstate = False
        
    
        if AI:
            AI_side,AI_top = SVM
            """loop through different z_stack sizes"""
            self.prediction = AI_side.SVM_predict(gray)
            if 1 in self.prediction:
                self.AI_side = True
            else:
                self.AI_side = False
            self.prediction = AI_top.SVM_predict(gray)
            if 1 in self.prediction:
                self.AI_top = True
            else:
                self.AI_top = False
        if self.AI_top == True:
            self.bool = True
        if self.AI_side == True:
            self.bool = True

        
        """the final segmentation figure stored"""
        """store = 'C:\\Users\\hanse\\Desktop\\Segmentation\\Nucleiisegments sideview\\All\\ALL\\'
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        length = len(os.listdir(store))

        from skimage.transform import resize
        image = resize(gray, (25, 25))
        plt.imshow(image,aspect='auto')
        plt.savefig(store + '\\' + 'A' + str(length) + '.png')
        plt.close()"""



class CellSetFeatures:
    def __init__(self,path = '',images = []):
        self.path = path
        
        """self.features i.e. the dict where the features will be stored"""
        self.cellfeatures      = {}
        self.sideviewfeatures  = {}
        
        if images != []:
            """raw images in dataset"""
            self.raw     = images
            """get the cytoplams versus the nucleii"""
            self.images  = {z:binarize_image(images[z]) for z in range(len(images))}
            """DAPI in array format"""
            self.DAPI = numpy.array([self.images[i] for i in sorted(self.images.keys())])
    
    def update_individual_features(self,cellfeatures):
        """the cellfeatures contains the information on individual cells in the 
        image and can be used to assign target to individual cells or to reconstruct
        a cell in 3D and obtain a final set"""
        for z, v in cellfeatures.items():
            if z in  self.cellfeatures: 
               for n in v:
                   self.cellfeatures[z].append(n)
            else:
                self.cellfeatures[z] = v
                
    def update_sideview_features(self,sideviewfeatures):
        """image and can be used to assign target to individual cells or to reconstruct
        a cell in 3D and obtain a final set, these are the sideview elements"""
        for k, v in sideviewfeatures.items():
            if k in  self.sideviewfeatures: 
               for n in v:
                   self.sideviewfeatures.append(n)
            else:
                self.sideviewfeatures[k] = v
        
    def return_data_object(self,):
        """the data in the set"""
        data = {}
    
        """fill up the dict with relevant dapidata"""
        data['Data']           = [(i.warped,i.label,i.coordinates,i.cutout,i.centroid,i.parameters,i.z)  for z,items in self.cellfeatures.items() for i in items]
        data["Raw Images"]     = self.raw
        data["Binary Images"]  = self.images
        return data

    def initiate_DAPI(self,data):
        """raw images in dataset"""
        self.raw     = data['Raw Images']
        """get the cytoplams versus the nucleii"""
        self.images  = data['Binary Images']
        """get the images in the dataset"""
        array = []
        for i in self.raw:
            array.append(binarize_image(i))
        """DAPI in array format"""
        self.DAPI = numpy.array(array)

    def initiate_cellfeatures(self,data):
        """plug the data into the features"""
        data = data['Data']
        """initiate the cell features"""
        for w,l,co,cu,ce,pa,z in data:
            if z in self.cellfeatures:
                self.cellfeatures[z].append(CellFeature(w,l,co,cu,ce,pa))
            else:
                self.cellfeatures[z] =     [CellFeature(w,l,co,cu,ce,pa)]
            
    def reconstruct_cells(self,AI = True):
        """we overlap the cells and try and map the structures presumed to be the same on top of each other
        by doing so we can create a novel reconstruction of the cell in 3D"""
        self.cells = [(i.label,i.centroid) for n in self.cellfeatures.values() for i in n]
        
        if AI:
            """get the sideview of the nucleii segment"""
            sideview = 'Segmentation\\Nucleiisegments sideview\\'
            """train an SVM on available data"""
            from MLclassification import MLtrainer
            SVM_sideview = MLtrainer(folder = sideview + 'Training data\\')  
            SVM_sideview.SegmentationData()
            SVM_sideview.SVM_Classifier()

            topview = 'Segmentation\\Nucleiisegments topview\\'
            SVM_topview = MLtrainer(folder = topview + 'Training data\\')   
            SVM_topview.SegmentationData()
            SVM_topview.SVM_Classifier()  

        """in this part we take the cells and attempt to reconstruct them, for now we feed in the labels with the centroid"""
        labels,centroids = zip(*self.cells)
        
        """integerize the lot"""
        centroids     = [(int(x),int(y),int(z)) for x,y,z in centroids]
        segmentlabels = {i:labels[i] for i in range(len(labels))}

        """cluster in a graph structure"""
        sets = build_coordinate_graph(centroids,distance = 10)   
        
        """get the overarching structures"""
        self.graphcount = len(sets)

        average,depth = {},{}
        """average the same segments out for iterative segmentation"""
        for l,centroid in sets.items():
            x,y,z = zip(*centroid)
            average[l] = (numpy.average(x),numpy.average(y),numpy.average(z)) 
            depth[l] = max(z) - min(z)
                  
        """Reconstruct the nucleus and test"""
        self.nuclei,self.labeldata  = [],[]
        for l,centroidsubset in sets.items():
            pop,li = [],[]
            for c in centroidsubset:
                if c not in pop:
                    pop.append(c)
                    for idx in [n for n, x in enumerate(centroids) if x == c]:
                        li.append(labels[idx])         
            try:
                self.labeldata.append((l,li))
                self.nuclei.append(ReconstructedNucleus(l,li,self.raw,self.DAPI,SVM = (SVM_sideview,SVM_topview)))
            
            except:
                print("There was an error in line 629, reconstructed nucleus class, please check, li probably empty")
              
        """take the deepest cells i.e. the ones that stack it the z-stack the most
        as the standard cell that is succesfully segmented and assign a true value"""
        sort  = reversed(numpy.argsort([len(i.cell) for i in self.nuclei]))
        sort  = [i for i in sort]

        """take 10 percent of the cells with the most penetration in the z-stack"""
        for i in range(int(len(self.nuclei)*0.01)):
            self.nuclei[sort[i]].bool = True

        """get volume reconstruct nucleus in 3D, take average"""
        volume  = numpy.average(numpy.array([i.volume for i in self.nuclei if i.bool == True]))
        
        """calculate number of cells based on average size cells (check number of 1's)"""
        x,y,z = self.DAPI.nonzero()
 
        """store raw signal count"""
        self.totalsignal = len(x)
        self.cellsize    = volume
        
        """divide this number by total"""
        self.signalcount = round(len(x)/float(volume),1)
        
        """known nuclei coordinates with cell label marker"""
        self.nucleus_coordinates = {}
        for i in self.nuclei:
            if i.bool == True:
                for c in i.coordinates:
                    self.nucleus_coordinates[c] = i.labelmarker
                    
        """cells as sorted by labelmarker"""
        self.cells = {i.labelmarker:i for i in self.nuclei} 
        
        """flattened Distributions are a sign of the replicative phase the cell is in"""
        self.cellstate = {i:self.nuclei[i].cellstate for i in range(len(self.nuclei)) if self.nuclei[i].bool == True}
        
        """average cellize"""
        average_cell_size = 0 
        for index,cell in self.cellstate.items():
            average_cell_size += len(cell[0].flatten())
        average_cell_size = average_cell_size/len(self.cellstate)
        
        """get the cell number out of the dataset, by dividing the average cellsize by the DAPI signal, we take half the image!"""
        number = int(len(self.DAPI)*0.05)
        
        """get the DAPI signal as a reference and divide by the total"""
        signalpixels = 0
        for i in range(number,len(self.DAPI)-number,1):
            dapicut = self.DAPI[i][255:511,0:511]
            x,y     = dapicut.nonzero()
            signalpixels += len(x)        
        self.cellnumber = signalpixels/average_cell_size   

    def return_celldata(self,):
        return (self.nucleus_coordinates,self.labeldata)
            
    def initiate_cell_reconstruction(self,celldata):
        self.nucleus_coordinates,self.labeldata = celldata
        """nuclei reconstructed"""
        self.nuclei = []
        for l,lset in self.labeldata:
            try:
                self.nuclei.append(ReconstructedNucleus(l,li,self.raw,self.DAPI))
            except:
                "Print There was an error in line 629, reconstructed nucleus class, please check, li probably empty"
              
        """cells as sorted by labelmarker"""
        self.cells = {i.labelmarker:i for i in self.nuclei}

    def global_position(self,data,datakeys = []):           
        """approximate the percentage of spots within the nucleus"""
        self.fraction = {}
        for m in datakeys:
            assign = {}
            """get the fraction of nuclear localized spots"""
            f = 0
            for c in data[m]:
                x,y,z = c
                try:
                    if self.DAPI[z,x,y] != 0:
                        f += 1
                except IndexError:
                    pass
            """assign nuclear the fraction"""
            self.fraction["Nuclear Fraction " + m] = f/float(len(data[m])+1)
            
    def global_position_targets(self,data):           
        """approximate the percentage of spots within the nucleus"""
        self.nucleus   = {i:0 for i in data.values()}
        self.cytoplasm = {i:0 for i in data.values()}
        
        for coordinate,target in data.items():
            assign = {}
            
            """get the fraction of nuclear localized spots"""
            x,y,z = coordinate
            try:
                if self.DAPI[z,x,y] != 0:
                    self.nucleus[target] += 1
                else:
                    self.cytoplasm[target] += 1
            except IndexError:
                pass
            
        """get the fraction expressed (nucleus versus cytoplasm)"""
        self.fraction = {'Nucleus':self.nucleus,'Cytoplasm':self.cytoplasm}            

    def local_position(self,data,datakeys = []):
        import pandas
        import seaborn
        """localization of the cells in the script"""
        self.localization = {}
        
        """get the metadata i.e. spots per cell"""
        self.metadata = {}
        
        for m in datakeys:
            """Loop through Spots
            1) see if its in a fully reconstructed nucleus
            2) see if its in a partially reconstructed nucleus
            3) see if its in the cytoplasm
            4) see if this part of the cytoplasm belongs to a reconstructed cell"""            
            assign = []
            print(data[m])
            """get the fraction of nuclear localized spots"""
            if type(data[m]) == list:
                data[m] = {i:False for i in data[m]}
            for c,target in data[m].items():
                
                x,y,z = c
                try:
                    index_test = self.DAPI[z,x,y]
                except IndexError:
                    index_test = False
                    
                """if index in DAPI move along (due to alignemnt some numbers may be outside bound)"""
                if index_test != False:
                    if self.DAPI[z,x,y] != 0.:           
                        f,location,distance = True,'Nucleus',False
                        try:
                            cellmarker = self.nucleus_coordinates[c]
                        except:
                            cellmarker = False
                    else:
                        f,location,distance = False,'Cytoplasm',False
                        cellmarker = False
                else:
                    f,cellmarker = True,False
 
                """assign spots to cells that can be reconstructed"""
                if f == False:
                    """generate coordinates in the circle"""
                    l = []
                    for i in numpy.array(circle(numpy.zeros((512,512)),x,y,radius = 30).nonzero()).T:
                        xg,yg = tuple(i)
                        if xg % 2 == 0 and yg % 2 == 0: 
                            l.append((xg,yg))

                    coordinates = []
                    for i in range(0,len(l),2):
                        xg,yg = tuple(l[i])  
                        for zg in range(-3,3,1):
                            if zg not in [1,-1]:
                                coordinates.append((xg,yg,z+zg))
                                
                    """create two vectors one with the original coordinate, one with the new coordinate"""
                    cv = numpy.array([numpy.ones((len(coordinates)))*x,numpy.ones((len(coordinates)))*y,numpy.ones((len(coordinates)))*z])
                    lc = numpy.array(coordinates)
                    
                    """dict with distance to original"""
                    corrected_distance = {i:False for i in coordinates}
                    """correct the distance, the per pixel distance is twice that of the distance along the xy direction"""
                    distances = (cv.T - lc)**2

                    """loop through LSE per axis and summate"""
                    for i in range(len(distances)):
                        """get SoS and take square root"""
                        corrected_distance[coordinates[i]] = math.sqrt(numpy.sum(distances[i]))

                    """check where the nucleus coordinates are"""
                    connected = []
                    for n in range(len(coordinates)):
                        x,y,z = coordinates[n]
                        try:
                            if (x,y,z) in self.nucleus_coordinates.keys():
                                connected.append((corrected_distance[(x,y,z)],(x,y,z)))
                            elif self.DAPI[z,x,y] != 0:
                                connected.append((corrected_distance[(x,y,z)],(x,y,z)))
                        except:
                            pass
                    try:
                        distance,closest = min(connected)
                        cellmarker = self.nucleus_coordinates[closest]
                    except:
                        pass

                if cellmarker != False:  
                    distance = calculate_distance(self.cells[cellmarker].centroid,c)
                    """assign the coordinate to the cell"""
                    assign.append({"Coordinate":c,
                                   'Location':location,
                                   'Cell':cellmarker,
                                   'Target':target,
                                   'Distance':distance,
                                   'Centroid':self.cells[cellmarker].centroid})
            
            """the location of individual spots in the dataset, i.e. the spots that could be assigned
            to the cells we were able to detect"""
            self.localization['Location ' + m] = copy.copy(assign)
            
            """metadata about localization of spots per cell in dataset including distance to cell center
            first we extract the cells we were able to assign coordinates to!"""
            from collections import Counter
            
            """flatten dataset from list of dicts to dict of lists"""
            flattened  = LDtoDL(self.localization['Location ' + m])
            
            """Group by target and cell to get the cellcount"""
            df = pandas.DataFrame(flattened)
            targetcount = df.groupby(['Cell', 'Target'])
            targetcount = targetcount.size()
            targetdict  = targetcount.to_dict() 
            targetcount.reset_index(name='Spots per Cell')
            
            """Group by location i.e. nucleus verus cytoplasm"""
            df = pandas.DataFrame(flattened)
            location = df.groupby(['Cell','Target','Location'])
            location = location.size()
            location.reset_index(name='Fraction')
            
                             
            """calculate fraction inside nucleus"""
            fraction =  {}
            for k,v in location.to_dict().items():
                cell,protein,location = k
                print(cell)
                if cell != False:
                    fraction[(cell,protein)] = 0
                    if location == 'Nucleus':
                        print(v,float(targetdict[(cell,protein)]))
                        fraction[(cell,protein)] = v/float(targetdict[(cell,protein)])
                    
            """calculate distance from centroid nucleus"""
            average_distance = {}
            for i in assign:
                if i['Cell'] != False:
                    if (i['Cell'],i['Target']) not in average_distance:
                        average_distance[(i['Cell'],i['Target'])] = []
                    average_distance[(i['Cell'],i['Target'])].append(i['Distance'])
            average_distance = {c:numpy.average(d) for c,d in average_distance.items()}

            """store metadata from individual coordinates"""
            metadata = {'Cell'                         :[],
                        'Target'                       :[],
                        'Cell Centroid'                :[],
                        "Spots Per Cell"               :[],
                        "Nuclear Fraction"             :[],
                        "Average Distance to Centroid" :[]}

            for cell,target in targetdict.keys():
                if cell != False:
                    metadata['Cell'].append(cell)
                    metadata['Target'].append(target)                   
                    metadata['Cell Centroid'].append(self.cells[cell].centroid)              
                    metadata["Spots Per Cell"].append(targetdict[(cell,target)])             
                    metadata["Nuclear Fraction"].append(fraction[(cell,target)])            
                    metadata["Average Distance to Centroid"].append(average_distance[(cell,target)]) 
                
            """store the metadata"""
            self.metadata['Location ' + m] = copy.copy(metadata)
            
            """split coordinate set"""
            x,y,z = zip(*flattened['Coordinate'])
            flattened["Cell"] = [str(i) for i in flattened["Cell"]]
            flattened['x'] = x
            flattened['y'] = y
            flattened['z'] = z
            
            """store the raw data in dataframe"""
            self.rawdata = pandas.DataFrame(flattened)
            import seaborn as sns
            plt.figure(figsize = (15,15))
            sns.scatterplot(data=self.rawdata,x=self.rawdata.x,y=self.rawdata.y,hue = 'Cell',style = 'Target',legend = False)
            plt.show()
            plt.figure(figsize = (15,15))
            sns.countplot(data=self.rawdata,y='Target',hue = 'Cell')
            plt.show()
            
