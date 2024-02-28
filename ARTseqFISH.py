# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:34:30 2022

@author: bob van sluijs
"""

import numpy
from scipy.spatial import distance
import os
import matplotlib.pylab as plt
import cv2 as cv2
from skimage import io
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm
from scipy import ndimage
from skimage.feature import peak_local_max
import itertools as it
from skimage.morphology import extrema
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from scipy.signal import argrelextrema
import copy
import scipy
import pickle
from PIL import Image as Im
from SpotDetection import Image
from skimage import draw
import itertools
from itertools import groupby
from operator import itemgetter
import random
import copy
import cProfile
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.ckdtree import cKDTree
from mpl_toolkits.mplot3d import Axes3D
from NucleusSegmentation import RenderVisCoordinates
from NucleusSegmentation import Segmentation
import copy as copy   
from memory_profiler import profile
from ImageAnalysisOperations import *
from Alignment import *


def SeqFishThread(position,path,imsize):
    return {position:Position(folder = position,imsize = imsize)}

def UpdateTargetCoordinateSet(factor,coordinates):
    return [(x+(512*factor),y,z) for x,y,z in coordinates]

def UpdateCellCoordinateSet(factor,coordinates):
    coordinates,boolean = zip(*coordinates.items())
    c = [(x+512*factor,y,z) for x,y,z in coordinates]
    return {c[i]:boolean[i] for i in range(len(c))}

def UpdateDistanceCoordinateSet(factor,coordinates):
    return [(x+(512*factor),y,z) for x,y,z in coordinates]


def ImportBenchmark(path):
    """the counts in the individually counted datasets""" 
    import pandas as pd
    df    = pd.read_excel(path).to_dict() 
      
    count = {}
    for ID,info in df.items():
        for i,c in info.items():
            count[ID] = int(c)
    return count
        
def CodePrune(p,index,rIDs):
    split = p.split('_')
    """delete rsplot"""
    delete = []
    for i in split:
        rsplit = i.split('.')
        if rsplit[-1] == '':
            delete.append(i)   
    """check in which round the signal was deleted"""
    segment = p[0:index]
        
    s = False
    for i in rIDs:
        if i in segment:
            s = copy.copy(i)
            
    split = [i + '_' for i in split if i not in delete]
    code  = ''.join(split)[0:-1]
    return code,s

def EdgeRange(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))


#assign probabilities to faulty potential signals
def FilterSignalSet(distance = 1,):
    labels = ['Targets','Predicted Targets','Potential Targets','Potential Predicted Targets']    
    "0) unpack the signals and create a dependency graph to find duplicates" 
    
    coordinates = list(targets.keys())
    cts = {i:True for i in list(coordinates)}
    
    "1) build k-d tree"
    kdt = cKDTree(numpy.array(coordinates))
    edges = kdt.query_pairs(2)
    
    "2) create graph"
    G = nx.from_edgelist(edges)
    
    "3) Find Connections"
    ccs = nx.connected_components(G)
    node_component = {v:k for k,vs in enumerate(ccs) for v in vs}
    df = pd.DataFrame(coordinates, columns=['x','y','z'])
    df['c'] = pd.Series(node_component)
    
    "4) extract features"
    sets = {}
    for k,v in node_component.items():
        if v in sets:
            sets[v].append((df['x'][k],df['y'][k],df['z'][k]))
        else:
            sets[v] = []
            sets[v].append((df['x'][k],df['y'][k],df['z'][k]))          
    
    "5) filter out the duplicate signals i.e. just pick one!"""
    duplicates = {}
    for i,potential_duplicate_clusters in sets.items():
        cluster = [targets[j].signalcode for j in potential_duplicate_clusters]
        duplicates[i] = [n for n in ListDuplicates(cluster)]

    """6) remove the duplicates in the set"""
    remove,keep,combined = [],[],[]
    for i,d in duplicates.items():
        dup = list(d)
        if dup != []:
            sgn,crd = zip(*list(dup))
            crd = list(crd)
            for indices in crd:
                selected = indices[0]
                for n in range(0,len(indices)):
                    index = indices[n]
                    remove.append(sets[i][index])   
                combined.append([sets[i][m] for m in indices])    
    
    """ 7) del duplicates in the targetset"""
    duplicateset = [targets[i].signalcode for i in remove]
    for i in remove:
        del targets[i] 
    return targets         

def Duplicates(l):
    for ID,items in l.items():
        l[ID] = list(set(items))
    return l

def flatten(dct):
    return [i for j in dct.values() for i in j]

def find_nearest(array, value):
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return array[idx]

def CodeDistance(c,r):
    if c == r:
        return 0
    """the code that we have"""
    code      = {}
    
    """the reference set"""
    reference = {}
    
    rounds = []
    """the split in the code"""
    rsplit = c.split('_')
    for i in rsplit:
        rnd,clrs = i.split('.')
        rounds.append(rnd)
        code[rnd] = clrs
    
    rsplit = r.split('_')
    for i in rsplit:
        rnd,clrs = i.split('.')
        rounds.append(rnd)
        reference[rnd] = clrs    
        
    """the rounds in the set"""
    rnd = list(set(rounds))
    
    diff = 0
    for i in rnd:
        clc = False
        clr = False
        try:
            clc = code[i]
        except KeyError:
            pass
        try:
              clr = reference[i] 
        except KeyError:
            pass
        
        if clc != False and clr == False:
            diff += len(clc)
            
        elif clc == False and clr != False:
            diff += len(clr)
            
        elif clc != False and clr != False:
            length  = 2*len([i for i in clc if i in clr])
            diff   += len(clc) + len(clr) - length
    return diff  
 
def ProportionalAssignment(targets):
    from collections import Counter
    probabilities = {}
    """counttanble and potential"""
    counttable,potential = [],[]
    for i in range(len(targets)):
        if type(targets[i]) == list:
            potential.append(targets[i])
        if type(targets[i]) == str:
            counttable.append(targets[i])
    datadict = Counter(counttable)
    
    """calculate the probability the assigned target is one or the other"""
    for p in potential:
        if tuple(sorted(p)) in probabilities.keys():
            pass
        else:
            pairwise_probability,pair = [],[]
            for name in sorted(p):
                pair.append(name)
                if name in datadict.keys():
                    pairwise_probability.append(datadict[name])   
                else:
                    datadict[name] = 1
                    pairwise_probability.append(datadict[name])
            percentage = [int((i/float(sum(pairwise_probability)))*100) for i in pairwise_probability]
            key = tuple(sorted(pair))
            probabilities[key] = copy.copy(percentage)
            
    """assign a new target proportionally"""
    for i in range(len(targets)):
        if type(targets[i]) == list:
            p = tuple(sorted(targets[i]))
            draw = random.choices(list(sorted(targets[i])), weights=probabilities[p])[0]
            targets[i] = copy.copy(draw)

    return targets

def compute_centroid(c):
    x,y,z = zip(*c)
    return (int(numpy.average(x)),int(numpy.average(y)),int(numpy.average(z)))

def metrics(metrics):
    """the m in the metrics"""
    m = []
    for i in metrics:
        m += i
        
    """the rounds,radius and rotation"""    
    rounds,radius,rotation = zip(*m)
    
    """store the rotation per round"""
    data = {i:[] for i in list(set(rounds))}
    for i in range(len(rotation)):
        data[rounds[i]].append(rotation[i]*radius[i])
        
    """calculate the average rotaion"""
    for r,rotations in data.items():
        data[r] = numpy.average(rotations)
        
    """where is the bias"""
    bias = {}
    for r,ave in data.items():
        bias[r] = int(ave/abs(ave))
    return bias

def ColocolizationGraph(targets, distance = 1):
    "0) unpack the signals and create a dependency graph to find duplicates" 
    coordinates = list(targets.keys())
    cts = {i:True for i in list(coordinates)}
    
    "1) build k-d tree"
    kdt = cKDTree(numpy.array(coordinates))
    edges = kdt.query_pairs(distance)
    
    "2) create graph"
    G = nx.from_edgelist(edges)

    "3) Find Connections"
    ccs = nx.connected_components(G)
    node_component = {v:k for k,vs in enumerate(ccs) for v in vs}
    df = pd.DataFrame(coordinates, columns=['x','y','z'])
    df['c'] = pd.Series(node_component)

    "4) extract features"
    sets = {}
    for k,v in node_component.items():
        if v in sets:
            sets[v].append((df['x'][k],df['y'][k],df['z'][k]))
        else:
            sets[v] = []
            sets[v].append((df['x'][k],df['y'][k],df['z'][k]))          

    "5) filter out the duplicate signals i.e. just pick one!"""
    colocolization = []
    for i,potential_clusters in sets.items():
        colocolization.append(list(set([targets[j].target for j in potential_clusters])))
    return colocolization 

def AssociationMapping(dataset):
    from collections import Counter
    from itertools import combinations
    countset  = Counter()
    for pair in dataset:
        if len(dataset) < 2:
            continue
        pair.sort()
        for c in combinations(pair,2):
            countset[c] += 1
    return countset.most_common()

def ListDuplicates(seq):
    from collections import defaultdict
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>1)

class CodeTopology:
    def __init__(self,c):
        self.distance = 1
        self.round    = []
        
        """the bastardized version of the code"""
        self.code = c
        self.N = []
        
    def update(self,n,s):
        self.N.append(n)
        self.round.append(s)

class Metrics:
    def __init__(self,detectedcode,targetcode,distance):
        self.code       = detectedcode
        self.distance   = distance

        """the rounds split from the colors"""
        rounds = targetcode.split("_")
        
        """similarly break the detected code down in parts"""
        targetcode = {}
        for i in rounds:
            r,cd = i.split('.')
            colors = cd.replace("", " ")[1: -1]
            targetcode[r] = colors.split(' ')
        self.size = len(flatten(targetcode))
            
        deletion = {}
        for r,colors in detectedcode.items():
            if r in targetcode:
                d = [i for i in colors if i not in targetcode[r]]
            elif r not in targetcode:
                d = colors
            deletion[r] = copy.copy(d)
        
        addition = {}
        for r,colors in targetcode.items():
            if r in detectedcode:
                d =  [i for i in colors if i not in detectedcode[r]]
            elif r not in detectedcode:
                d = targetcode[r]
            if d != []:
                addition[r] = copy.copy(d)  
        
        """addition and deletion of signals added to the detected code before its a valid target"""
        self.addition = addition
        self.deletion = deletion

        self.deconvolute,self.append = False,False
        """check if all the signals that are in the target are in here as well"""
        if len(flatten(addition)) == 0 and len(flatten(deletion)) > 0:
            self.deconvolute = True   
        if len(flatten(addition)) > 0 and len(flatten(deletion)) == 0:
            self.append = True           

class PairwiseDistance:
    def __init__(self,detectedcode):
        """the code"""
        self.code     = detectedcode
        """distance matrix"""
        self.distance = {}
        """split the code into two"""
        rounds = self.code.split("_")
        
        """break the code down into its consitutuent parts"""
        self.detectedcode = {}
        for i in rounds:
            r,c = i.split('.')
            colors = c.replace("", " ")[1: -1]
            self.detectedcode[r] = colors.split(' ')
            
        """length of the code"""
        self.size = len(flatten(self.detectedcode))            
        """the targetbook """
        self.targetbook = {}
        
    def update(self,targetcode,t,d):
        self.targetbook[targetcode] = Metrics(self.detectedcode,targetcode,d)
        """break the code down and see wich colors would have to be deleted to decode the signal successfully"""
        self.distance[targetcode] = (targetcode,t,d)        

    def sort(self,):
        self.shortest_path = ''
        prune = {c:obj for c,obj in self.targetbook.items() if obj.deconvolute == True}
        if len(prune) > 0:
            minimum = min([obj.distance for obj in prune.values()])
            self.shortest_path = [c for c,obj in prune.items() if obj.distance == minimum]
            if self.shortest_path  == []:
                self.shortest_path = ''

class Codebook:
    def __init__(self,path,codemap = {}):
        """Path of the codebook"""
        import itertools as it
        import pandas as pd
        import string
        
        """download the codebook"""
        alphabet = list(string.ascii_lowercase) + list(string.ascii_uppercase)
        df       = pd.read_excel(path)
        codedict = df.to_dict()
        spots    = df["Class"]  
        rounds   = df.columns
        
        """the rounds in the set"""
        r = [rounds[i] for i in range(1,len(rounds))]

        """function that creates the code """
        self.codebook = {}
        self.codemap  = codemap
    
        """the spots in the set"""
        for number,spotID in spots.items():
            code = ''
            for rnd in r:
                if codedict[rnd][number] != '-':
                    sortstring = '.'
                    for i in sorted(codedict[rnd][number]):
                        sortstring += i
                    code += rnd + sortstring
                    code += '_'
            
            """the codes assigned as set in the codebook"""
            if code[-1] == '_':
                code = code[0:-1]
            self.codebook[code] = spotID
        codekeys = list(self.codebook.keys())
        
        """find everything that is not a number and not a round"""
        letters = []
        for code in codekeys:
            for i in code:
                if i.isnumeric() == False:
                    if i in alphabet:
                        letters.append(i)
            
        """get the letterset"""    
        letterset = list(set(letters))
        
        N = []
        
        """get the individual letterset"""
        for code in codekeys:
            for letter in letterset:
                indices = [pos for pos, char in enumerate(code) if char == letter]
                
                """copy code of indices"""
                for index in indices:
                    try:
                        if not code[index+1].isnumeric():
                            p = copy.copy(code)
                            p = list(p)
                            del p[index]
                            p = ''.join(p)
                            p,s = CodePrune(p,index,r)
                            N.append((copy.copy(p),copy.copy(code),copy.copy(s)))
                            
                    except IndexError:
                            p = copy.copy(code)
                            p = list(p)
                            del p[index]
                            p = ''.join(p)
                            p,s = CodePrune(p,index,r)
                            N.append((copy.copy(p),copy.copy(code),copy.copy(s)))

        """the 1S removed codes"""
        self.topology = {}

        """get the distance n removed in codebook"""
        b,c,s = zip(*N)
        for i in range(len(b)):
            try:
                self.topology[b[i]].update(c[i],s[i])
            except:
                self.topology[b[i]] = CodeTopology(b[i])
                self.topology[b[i]].update(c[i],s[i])
  
        colors = {}
        """extract individual colors"""
        for i in range(1,len(rounds)):
            colors[i] = sorted(codemap.values())

        """map the distances of large codes to real codes"""
        self.map = {}
        
        """the colors in the set and strike out the duplicates and make all possible combinations"""
        for rnd in sorted(colors.keys()):
            self.map[rnd],clr = [],[]
            for i in range(1,len(colors[rnd])+1,1):
                for n in itertools.combinations(colors[rnd], i):
                    clr.append(list(sorted([j for j in n])))
            
            """the potentialcodes in the rounds"""
            m = []
            for i in clr:
                if len(i) > 0:
                    temp = 'R' + str(rnd) + '.'
                    for j in i:
                        temp += j
                    temp += '_'
                    m.append(temp)
    
                """the list of the code""" 
                m = [i for i in m if i != 'R' + str(rnd)]
                m.append('')
                m = list(set(m))
                    
                """get the list"""
                self.map[rnd] = copy.copy(m)

        """create the potential codes that are possible by unpacking them, combining them with it.product"""
        elements  = [self.map[i] for i in range(1,len(rounds))]
        codes     = list(set([''.join(code)[0:-1] for code in list(itertools.product(*elements)) if ''.join(code)[0:-1] != '']))
        
        c_combination,r_combination = {},{}
        for i in it.combinations(list(self.codebook.keys()),2):
            c1,c2 = i
            
            """copy the first """
            base  = copy.copy(c1)
        
            """the rounds split from the colors"""
            rounds = base.split("_")
            
            """similarly break the detected code down in parts"""
            combined = {}
            for i in rounds:
                r,cd = i.split('.')
                colors = cd.replace("", " ")[1: -1]
                combined[r] = colors.split(' ')
            
            """the rounds split from the colors"""
            rounds = c2.split("_")
            
            """similarly break the detected code down in parts"""
            referencecode = {}
            for i in rounds:
                r,cd = i.split('.')
                colors = cd.replace("", " ")[1: -1]
                referencecode[r] = colors.split(' ')      
                
            for k,v in referencecode.items():
                if k in combined:
                    combined[k] += v
                else:
                    combined[k] = v
                
            code,rc = '',''
            for k in sorted(combined.keys()):
                """stripped code"""
                code += k + '.' 
                code += ''.join(list(sorted(set(combined[k]))))
                code += '_'
                
                """the raw code"""
                rc += k + '.' 
                rc += ''.join(list(sorted(combined[k])))
                rc += '_'
                  
            code = code[0:-1]
            if code not in c_combination:
                c_combination[code] = []
            c_combination[code].append((self.codebook[base],self.codebook[c2]))

            rc = rc[0:-1]
            if rc not in r_combination.keys():
                r_combination[rc] = []
            r_combination[rc].append((self.codebook[base],self.codebook[c2]))

        self.combined_codebook     = c_combination
        self.raw_combined_codebook = r_combination
        
        """"get the distance of these codes to the real codes and to each other"""
        self.association = {}
        for detectedcode in codes:
            d = PairwiseDistance(detectedcode)
            for targetcode in self.codebook.keys():
                diff = CodeDistance(targetcode,detectedcode)
                d.update(targetcode,self.codebook[targetcode],diff)
            self.association[detectedcode] = d
     
        """the distance matrix for the dataset and probe sequence"""   
        self.prune   = {}
        self.metrics = {}

        """plug in dataframe and connect the network"""
        data = {'cost':[],'reference':[],'node':[]}
        for r,o in self.association.items():
            self.metrics[r] = o.targetbook
            o.sort()

        """
            if o.smallest < 3:
                if o.code not in self.codebook.keys():
                    if o.extended_code:
                        self.prune[obj.code] = obj.extended_code
                        
            for c,d in obj.distance.items():
                if d[2] < 3:
                    data['cost'].append(d)
                    data['reference'].append(r)
                    data['node'].append(c)
        """

    def decode_string(self,code):
        """the codebook object contains maps the sequence of hybridizations
        to a target, we check if the aligned sequences found in the images
        match the codebook e.g. is R1.GB_R2.R in self.codebook"""
        if code in self.codebook:
            return self.codebook[code]
        else:
            return ''   
        
    def decode_optional_string(self,code):
        """the codebook object contains maps the sequence of hybridizations
        to a target, we check if the aligned sequences found in the images
        match the codebook e.g. is R1.GB_R2.R in self.codebook"""
        if code in self.association:
            return self.association[code].shortest_path
        else:
            return ''    
        
    def decode_combined_string(self,code):
        if code in self.combined_codebook:
            return self.combined_codebook[code]
        else:
            return ''  
        
    def decode_raw_combined_string(self,code):
        if code in self.raw_combined_codebook:
            return self.raw_combined_codebook[code]
        else:
            return ''  

    def visualize(self,):
        import pandas
        import networkx as nx
        df = pandas.DataFrame(data, columns=['cost','reference','node'])
        plt.figure(figsize=(10,10),dpi = 500)
        G=nx.from_pandas_edgelist(df, 'reference', 'node', ['cost'])
        nx.draw_networkx(G,with_labels = False,node_size=5,width=1)
        nx.drawing.layout.spring_layout(G)


class Channel:
    def __init__(self,c,r,image,folder = ''):
        from SpotDetection import Image
        """the channel in the list"""
        self.channel = c      
        self.round   = r
        self.path    = image
        """check if pickled GPS coordinates exists, if not than re-analyze"""
        path = folder + '\\SpotDetection\\'
        """create the folder for spotdetection"""
        try:
            os.mkdir(folder)
        except:
            pass
        try:
            os.mkdir(path)
        except:
            pass

        try:
            """see if the path that loads the spotdetection data exists
            if so load the data, including the pointspreadfunction and
            the AI based detection of spots in the dataaset"""
            loadpath  = path + str(c) + '.pickle'
            pickle_in = open(loadpath,"rb")
            self.data = pickle.load(pickle_in)
            
            """unpack the relevant parts of the data"""
            self.coordinates = self.data['coordinates'] 
            self.psf         = self.data['psf']
            self.bool        = self.data['bool']
            
        except:
            """Run the analysis, if it exists than store it!"""
            self.image = Image(image,plot = False)
            unpack     = self.image.GPS
            
            data = []
            """unpack the datasets"""
            for coordinate,obj in unpack.items():
                try:
                    data.append((coordinate,obj.bool,obj.psf,obj.gaussfit))
                except:
                    pass
            """store the coordinates in a temp file, otherwise it takes too long"""  
            c,b,n,f   = zip(*data)
            del data
            
            """unpack and store the data in dictionary"""
            self.data = {'coordinates':c,
                         'bool'       :b,
                         'psf'        :n}
        
            """unpack the setlist"""
            self.coordinates = self.data['coordinates'] 
            self.psf         = self.data['psf']
            self.bool        = self.data['bool']
            
            """pickle the data dict and store it"""
            with open(loadpath, 'wb') as handle:
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)   
        
        """all signals that make up a code"""
        self.signals  = [Signal(self.round,
                                self.channel,
                                
                                self.coordinates[i],
                                self.bool[i],
                                self.psf[i]) for i in range(len(self.coordinates))]
       
class Round:
    def __init__(self,r,roundset,folder = '',size = (512,512)):
        """store the information needed to analyse the data"""
        import pickle
        listdir = os.listdir(folder)
        """the repository must have a results path"""
        result_path = folder + '\\Results\\'  
        try:
            os.makedirs(alignment_path)
        except:
            pass
        
        """the round coordinate where the hybrdization took place"""
        self.round = r
        
        """get the channel lists"""
        self.channels = {}
        
        """the round """    
        for i in roundset:
            """the number of the channel"""
            number = i.split('.')[0][-1]
            
            """the Nuclei in the images"""
            if 'DAPI' in i:
                self.DAPIpath = folder + i
            elif number == '0':
                self.DAPIpath = folder + i
            else:
                if "Results" not in i:
                    self.channels[number] = Channel(number,r,folder + i,folder = result_path)

        ############  SEGMANTATION  #############  
        from CellReconstruction import CellSetFeatures
        """the DAPI segment set"""
        DAPI_path = result_path + '\\DAPI\\'  
        try: 
            os.makedirs(DAPI_path)
        except FileExistsError:
            pass
        """this is segmentation for the aligment and cellreconstruction, see if file exists"""
        if os.path.isfile(DAPI_path + '\\' + 'DAPI {}.pickle'.format(str(r))):  
            """load the files"""
            loadpath   = DAPI_path  + '\\' + 'DAPI {}.pickle'.format(str(r))
            pickle_in  = open(loadpath,"rb")
            data       = pickle.load(pickle_in)
            
            """build the cells"""
            self.CSF = CellSetFeatures()
            self.CSF.initiate_DAPI(data)
            self.CSF.initiate_cellfeatures(data)  
        try:
            """load the files"""
            loadpath   = DAPI_path  + '\\' + 'DAPI {}.pickle'.format(str(r))
            pickle_in  = open(loadpath,"rb")
            data       = pickle.load(pickle_in)
            
            """build the cells"""
            self.CSF = CellSetFeatures()
            self.CSF.initiate_DAPI(data)
            self.CSF.initiate_cellfeatures(data)     

        except:
            self.segmentation = Segmentation(self.DAPIpath,AIenhanced = True,SVM = True,segmentsides = False)
            self.CSF = self.segmentation.CSF
            storepath = DAPI_path + '\\' + 'DAPI {}.pickle'.format(str(r))
            with open(storepath, 'wb') as handle:
                pickle.dump(self.CSF.return_data_object(), handle, protocol=pickle.HIGHEST_PROTOCOL)  

class Signal:
    def __init__(self,r,channel,coordinate,predicted,psf):  
        """the signal contains the necessary information to extract the data as needed"""
        color = ['k','g','r','b','gold','cyan',"purple","DarkRed","DarkGreen","DarkBLue"]
        
        """coordinates of the signals position in experiment"""
        self.channel     = channel
        self.fluorophore = color[int(channel)]
        self.plotcolor   = color[int(r)]
        self.round       = r

        """the signals and coordinates"""
        self.coordinate  = coordinate
        self.psf         = psf
        self.bool        = predicted
        
        """coordinate alignment"""
        self.intraset  = {}
        
        """get the coordinates"""
        self.xc,self.yc,self.zc = coordinate
                  
    def shift_correction(self,shift,reference):
        """correct the shift in the signal"""
        x,y,z = 0,0,0
        if reference != self.round:
            x,y,z = shift[(reference,self.round)]      
            
        """the xyz in the reference"""
        self.x = self.xc + y
        self.y = self.yc + x
        self.z = self.zc - z
        
        """plot coordinate is just the tuple version of shift"""
        self.adjusted_coordinate = (self.x,self.y,self.z)
        
        """c the z independent version"""
        self.c = (self.x,self.y)
        
    def calculate_relative_position(self,reference):
        """the coordinate of the current meta pixel"""
        xr,yr = reference
        
        """reference coordinate"""
        x,y   = self.c 
        
        """difference between coordinates"""
        dx = xr - x
        dy = yr - y

        """the distance and the distance between spots"""
        self.radius   = math.sqrt(dx**2 + dy**2)
        self.rotation = math.degrees(math.atan2(dy, dx))
             

class Barcode:
    def __init__(self,c,signals):
       """the centroid of the set of signal coordinates c"""
       self.c  = c
       self.centroid = compute_centroid(c) 
       
       """get the signals"""
       self.signals = signals

       """signals sorted per round"""
       self.barcode = {i.round:[] for i in signals}
       for i in signals:
           self.barcode[i.round].append(i)
           
       """the excluded signal is the signal that is present in AI unfiltered set exlusively"""  
       self.excluded_signal = [] 
           
       """AI filtered barcode i.e. those with a perfect gaussian pointspread"""
       self.filtered_barcode = {i.round:[] for i in signals}
       for i in self.signals:
           if i.bool == True:
               self.filtered_barcode[i.round].append(i)   
           else:
               self.excluded_signal.append((i.round,i))
 
       """the signal colors are markers to see which fluorophore is present
       i.e. we unpack the barcode which contains information about the signal"""
       self.signalcolors = {}

       """relative position = relative position i.e. coordinate within the barcode
       so we see how signals are shifted within the set and how we need to assign them"""
       self.relative_position = {}
             
       """difference between individial sets"""
       for r in sorted(self.barcode.keys()):
           signals = self.barcode[r]
           colors    = [i.channel for i in self.barcode[r]] 
           if colors:
               self.signalcolors[r] = copy.copy(colors)
         
           difference = {} 
           """calculate distance between spots"""
           for n,m in itertools.combinations(signals,2):
               xn,yn = n.c
               xm,ym = m.c
               
               """check the alignment within the set"""
               difference[(n,m)] = math.sqrt(((xn - xm)**2 + (yn-ym)**2))

           """the relative position of the signals"""
           self.relative_position[r] = difference
       
    def decode(self,codebook,codemap = {}):
        """in the decode step we make a one to one translation of our signal
        into the target signal, with this we aim to assign an initial signal code
        for this signal code we will generate N number of potential targets"""
        self.target = ''
        
        """the signalcode for the combined signals"""
        self.initial_signal_code      = ''
        
        """filterd code for the combined signals of AI"""
        self.filtered_signal_code     = ''
        
        """the signalcode for the combined signals"""
        self.raw_initial_signal_code  = ''
        
        """filterd code for the combined signals of AI"""
        self.raw_filtered_signal_code = ''
        
        """hybdata, get metadata for statistics on hybridization"""
        from collections import defaultdict
        self.hybridization_data = defaultdict(list)
        
        self.codelength = 0
        """get the signal code"""
        for r in sorted(self.barcode.keys()):
            """codemap not defined is directly defined by channel"""
            if codemap == {}:
                codemap = {int(i.channel):int(i.channel) for i in self.barcode[r]}
                
            """assign the colors, this is a mapping step for reference
            i.e letter markers for a signals instead of a number"""
            initial_colors   = [codemap[int(i.channel)] for i in self.barcode[r]]
            filtered_colors  = [codemap[int(i.channel)] for i in self.filtered_barcode[r]]
            
            """append the metadata to see if signal is too crowded (and where)"""
            self.hybridization_data[r] += initial_colors
            
            """codelength of the detected signal"""
            self.codelength += len(initial_colors)

            # print(initial_colors,filtered_colors)
            if initial_colors != []:
                code    = 'R' + str(r) + '.' + ''.join(sorted(set(initial_colors)))  + '_'
                rawcode = 'R' + str(r) + '.' + ''.join(sorted(initial_colors))      + '_'
                self.initial_signal_code     += code  
                self.raw_initial_signal_code += rawcode

            if filtered_colors != []:
                code    = 'R' + str(r) + '.' + ''.join(sorted(set(filtered_colors))) + '_'
                rawcode = 'R' + str(r) + '.' + ''.join(sorted(filtered_colors))      + '_'
                self.filtered_signal_code     += code   
                self.raw_filtered_signal_code += rawcode
                
        """remove the filtered signal"""
        self.initial_signal_code      = self.initial_signal_code[0:-1]
        self.filtered_signal_code     = self.filtered_signal_code[0:-1]    
        self.raw_initial_signal_code  = self.raw_initial_signal_code[0:-1] 
        self.raw_filtered_signal_code = self.raw_filtered_signal_code[0:-1]    
                
        """bool marking whether we accept the code"""
        self.accept = False
        
        """we deconvolute the signal and assign potential targets to the deconvolution
        this means we already assign different targets to the dataset. We will
        subsequently use the metadata from previous decoding to further deconvolute the
        remaining signals i.e. if green fluorophore is least efficient 
        assign higher likelihood to codes with a missing green signal etc."""
        self.deconvoluted_target       = []
        self.potential_target          = []
        self.combined_potential_target = []

        """decode the spots using the codebook"""
        self.initial_target  = codebook.decode_string(self.initial_signal_code)
        self.filtered_target = codebook.decode_string(self.filtered_signal_code)
        
        """check if they are the same"""
        match = False
        if self.initial_target == self.filtered_target:
            match = True
        
        """get the target option from the inital signal code """
        self.potential  = codebook.association[self.initial_signal_code].targetbook

        """see if the code is accepted by the codebook"""
        if match == True and self.initial_target != '':
            self.deconvoluted_target = self.initial_target
            self.accept = True
            
        elif self.initial_target != '':
            self.deconvoluted_target = self.initial_target
            self.accept = True
            
        elif self.filtered_target != '':
            self.deconvoluted_target = self.filtered_target
            self.accept = True
  
        self.metrics = []
        """if accepted get the metrics i.e. see if there is a shift"""
        x,y = zip(*[i.c for i in self.signals])
        if self.accept == True:
            """unpack it to get the rotation and distance from this centroid"""
            xr,yr,zr = self.centroid
            
            for i in range(len(x)):
                """difference between coordinates"""
                dx = xr - x[i]
                dy = yr - y[i]
    
                """the distance and the distance between spots"""
                radius   = math.sqrt(dx**2 + dy**2)
                rotation = math.degrees(math.atan2(dy, dx))
                
                """if the radius is smaller than 1 then apply rotaion"""
                if radius < 0.5:
                    rotation = 0
                    
                """the metrics i.e. where is the signal located relative to its own centroid """
                self.metrics.append((self.signals[i].round,radius,rotation))

    def deconvolute(self,codebook,codemap = {},metric = {}):
        """if we have an accepted code we pass it, if not 
        we check if we can deconvolute it a bit we do this in 3 steps
        we add signals that are structurally lower should only 1 be missing
        we strip signals if there are too many based on their local position
        in the aggregated signal"""
        
        """see if we can get some clusterd targets out i.e. a direct combination of two codes"""
        if self.accept == False:    
            ct_initial  = codebook.decode_raw_combined_string(self.raw_initial_signal_code)
            ct_filtered = codebook.decode_raw_combined_string(self.raw_filtered_signal_code)
     
            if ct_initial == ct_filtered:
                if ct_filtered != '':
                    self.combined_potential_target = ct_filtered
                    self.accept = True
                    
            if ct_initial != ct_filtered:
                if ct_filtered != '':
                    self.combined_potential_target = ct_filtered
                    self.accept = True
                    
            if ct_initial != '':
                self.combined_potential_target = ct_initial
                self.accept = True

            if self.accept == True:
                self.target = self.combined_potential_target
                 
        """start checking if we need to delete signals to make a code fit
        we choose to do this for the closest signals we can delete"""
        if self.accept == False:
            """store the deletion"""
            accepted = []
            """check the distance from the target to the core"""
            for target,obj in self.potential.items():
                if obj.distance == 1 and obj.deconvolute == True:
                    for hyb_round,to_be_deleted in obj.deletion.items():
                        if to_be_deleted:
                            accepted.append(codebook.codebook[target]) 
                                
            """accept this deletion"""
            if accepted:
                self.accept = True
                self.potential_target = accepted
                
        """start checking if we need to delete signals to make a code fit
        we choose to do this for the closest signals we can delete"""
        if self.accept == False:
            """store the deletion"""
            provisionally_accepted = []
            """check the distance from the target to the core"""
            for target,obj in self.potential.items():
                if obj.distance < 4 and obj.deconvolute == True:
                    collected = True
                    for hyb_round,to_be_deleted in obj.deletion.items():
                        if hyb_round not in target:
                            pass
                        else:
                            collected = False
                    if collected:
                        provisionally_accepted.append((obj.distance,codebook.codebook[target]))
                        
            if provisionally_accepted:
                distance,accepted = zip(*sorted(provisionally_accepted))
                accepted = [accepted[i] for i in range(len(distance)) if distance[i] == min(distance)]
                                
            """accept this round deletion"""
            if accepted:
                self.accept = True
                self.potential_target = accepted              
            
        """start checking if we need to add signals to make a code fit
        we choose to do this for the closest signals we can add i.e. a single color"""
        if self.accept == False:
            """store the deletion"""
            accepted = []
            """check the distance from the target to the core"""
            for target,obj in self.potential.items():
                if obj.distance == 1 and obj.append == True:
                    for hyb_round,to_be_added in obj.addition.items():
                        if to_be_added:
                            accepted.append(codebook.codebook[target]) 
                                
            """accept this color addition"""
            if accepted:
                self.accept = True
                self.potential_target = accepted 
                
        if self.accept == False:
            """store the deletion"""
            provisionally_accepted = []
            """check the distance from the target to the core"""
            for target,obj in self.potential.items():
                if obj.distance < 3 and obj.deconvolute == True:
                    provisionally_accepted.append((obj.distance,codebook.codebook[target]))
                        
            if provisionally_accepted:
                distance,accepted = zip(*sorted(provisionally_accepted))
                accepted = [accepted[i] for i in range(len(distance)) if distance[i] == min(distance)]
                                
            """accept this round deletion"""
            if accepted:
                self.accept = True
                self.potential_target = accepted 
                
    def assign_target(self,targes):
        """assign the target in de dataset"""
        self.target = ''
        
        """proportional assignment is needed"""
        self.proportional_assignment = False
        
        """the double signal is not present unless otherwise defined"""
        self.combined_assignment = False
        
        """accept the target ans assign it"""
        if self.accept:
            
            if self.deconvoluted_target:
                self.target = self.deconvoluted_target
                
            elif self.potential_target:
                if len(self.potential_target) == 1:
                    self.target = self.potential_target[-1] 
                if len(self.potential_target) > 1:
                    self.proportional_assignment = True
                    
            elif self.combined_potential_target:
                self.combined_assignment = True
                if len(self.combined_potential_target) == 1:
                    self.target = self.combined_potential_target[0]
                if len(self.combined_potential_target) > 1:
                    self.proportional_assignment = True

class Position:
    def __init__(self,folder = '',imsize = (512,512),paths = ('',''),codemap = {},codebook = {},benchmark = {},randomize = False,position = False,minimal_barcode_length = 2):
        from Plot import PlotARTseqFISH
        import pandas
        import pickle
        import os
        
        """We block the code into different sections:
            1) IF Hybridization detection?
                   pass  
               IF NOT:
                   Analyse image and store
            1b) IF the DAPI signal?
                   pass
               IF NOT:
                   Analyse image and store
            2) IF cell exist?
                   pass 
                IF NOT
                   Use DAPI and build cell
            3) IF alignment?
                    pass
                IF NOT
                    Align the images 
            4) IF decoding?
                   pass 
               IF NOT
                   Decode the data
               4b) Assign spots to cells"""
            
        """path results are stored"""
        result_path = folder + '\\Results\\'       
        plot        = PlotARTseqFISH(folder = result_path)

        """1) ANALYSE THE HYBRIDIZATION DETECTION 
              we loop through the rounds and its respective images
              and segment the DAPI and detect the spots"""
        
        """the repository must have a results path"""
        listdir = os.listdir(folder)
        try:
            os.makedirs(result_path)
        except:
            print("results folder already exists")

        """the hybridization rounds in the images"""
        rounds = {}  
        
        """the directories in the round folders"""
        for i in listdir:
            """find index of all folders except results, the images in the directory"""
            if 'R' in i:
                index  = i.index("R")
                if i[index+1].isnumeric():
                    rnd         = int(i[index+1])
                    images      = os.listdir(folder + i)
                    path        = folder + i + "\\"
                    rounds[rnd] = Round(rnd,images,folder = path) 
                    
   
        """2) RECONSTRUCT THE CELLS
              we take the repeated segmentations and reconstruct the cells
              with this we can assign spots to cells in the dataset!"""
        """we reference everything to round one so we take the DAPI from that round as well!"""
        
        self.CSF = rounds[min(rounds.keys())].CSF 
        self.CSF.reconstruct_cells()
        """pickle the data dict and store it"""
        with open(result_path + '\\Cells.pickle', 'wb') as handle:
            pickle.dump(self.CSF.return_celldata(), handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
        """get the cellstate distributions"""
        cellstate = self.CSF.cellstate
        with open(result_path + '\\cellstate2.pickle', 'wb') as handle:
            pickle.dump(cellstate, handle, protocol=pickle.HIGHEST_PROTOCOL) 

        if not os.path.isfile(result_path + '\\Decoding\\' + 'rawdata.pickle'):
            self.CSF = rounds[min(rounds.keys())].CSF 
            self.CSF.reconstruct_cells()
            """pickle the data dict and store it"""
            with open(result_path + '\\Cells.pickle', 'wb') as handle:
                pickle.dump(self.CSF.return_celldata(), handle, protocol=pickle.HIGHEST_PROTOCOL) 
            
            """get the cellstate distributions"""
            cellstate = self.CSF.cellstate
            with open(result_path + '\\cellstate.pickle', 'wb') as handle:
                pickle.dump(cellstate, handle, protocol=pickle.HIGHEST_PROTOCOL) 

        """3) ALIGN THE IMAGES
              we take the repeated segmentations use these to align the 
              images between rounds, on 'bad' microscopes we need to stitch
              images, this allows us to decode""" 
        """global image alignment based on local features"""
        segmentset   = {i:rounds[i].CSF.cellfeatures for i in rounds.keys()}
        binarization = {i:rounds[i].CSF.images for i in rounds.keys()}
        
        """if we use multiple parameter sets for segmentation stick to 1
        for the alignment otherwise it takes an eternity"""
        parameters   = []
        for r,data in segmentset.items():
            for z,features in data.items():
                for obj in features:
                    parameters.append(str(obj.parameters))
                    
        from collections import Counter
        count   = Counter(parameters)
        optimal = max(count, key= lambda x: count[x]) 
        for r,data in segmentset.items():
            for z,features in data.items(): 
                segmentset[r][z] = [i for i in segmentset[r][z] if str(i.parameters) == (str(optimal))]

        """reference round which we align to"""
        reference = min(list(rounds.keys()))
        alignment_path    = result_path + '\\Alignment\\'  
        
        """see if folder is there"""
        loadpath = alignment_path + 'shift.pickle'
        pickle_in = open(loadpath,"rb")
        
        """load it into the system"""
        shift = pickle.load(pickle_in) 
        paths = {i:rounds[i].DAPIpath for i in sorted(rounds.keys())}
        print(shift)
        StoreImageCorrectionOverlap(shift,paths,path = alignment_path + "\\")   
        
        try: 
            try:
                alignment_path    = result_path + '\\Alignment\\'  
                df                = pd.read_excel(alignment_path + 'manual correction.xlsx')
                manual_correction = {r:s[0] for r,s in df.to_dict().items()}
            except:
                manual_correction = {i:0 for i in rounds.keys()}
            try:
                alignment_path    = result_path + '\\Alignment\\'  
                df                = pd.read_excel(alignment_path + 'manual correction.xls')
                manual_correction = {r:s[0] for r,s in df.to_dict().items()}
            except:
                pass
                              
            """see if folder is there"""
            loadpath = alignment_path + 'shift.pickle'
            pickle_in = open(loadpath,"rb")
            
            """load it into the system"""
            shift = pickle.load(pickle_in) 
            paths = {i:rounds[i].DAPIpath for i in sorted(rounds.keys())}
            StoreImageCorrectionOverlap(shift,paths,path = alignment_path + "\\")   
            
        except:             
            """find the shift"""
            alignment_path = result_path + '\\Alignment\\'
            shift,sf  = {},[]
            
            try:
                loadpath = alignment_path + 'global shift.pickle'
                pickle_in = open(loadpath,"rb")
                
                """load it into the system"""
                shift = pickle.load(pickle_in) 
                for pair,s in shift.items():
                    shift[pair] = (int(s[0]),int(s[1]),int(s[2]))
                        
                """update with manual alignemnt"""
                for roundpair,coordinate in shift.items():
                    reference,r = roundpair
                    x,y,z       = coordinate
                    if r in manual_correction:
                        z           = manual_correction[r]
                    
                    """the new roundpair"""
                    shift[roundpair] = (x,y,z)
                    
            except:
                for c1 in segmentset.keys():
                    if c1 == min(segmentset.keys()):
                        for c2 in segmentset.keys():
                            if c2 != c1:

                                """the x and y average"""
                                x_ave,y_ave,z_ave,shiftset = AlignFeatures(segmentset[c1],segmentset[c2],binarization[c1],binarization[c2])
                                """the shiftset"""
                                sf.append(shiftset)
                                """the alignment of the signal set"""
                                shift[(c1,c2)] = (int(x_ave),int(y_ave),int(z_ave))
                 
                """store shifts in the main round folder"""
                storepath = alignment_path + '\\' + 'global shift.pickle'
                
                """this is a manaul override of the automated alignment script
                by doing this it forces the shfit to accept this z-position"""        
                for roundpair,coordinate in shift.items():
                    reference,r = roundpair
                    x,y,z = coordinate
                    
                    """if there is a manual correction for the shift then
                    the previous zshift calculation will be adjusted"""        
                    if r in manual_correction:
                        z  = manual_correction[r]
                    
                    """the new roundpair"""
                    shift[roundpair] = (x,y,z)
                
                """pickle the data dict and store it"""
                with open(storepath, 'wb') as handle:
                    pickle.dump(shift, handle, protocol=pickle.HIGHEST_PROTOCOL) 

            """a local feature alignment to refine the alignment"""
            shift,calibrationscore = LocalFeatureRefinement(rounds,shift) 
            """store image overlap"""
            paths = {i:rounds[i].DAPIpath for i in sorted(rounds.keys())}
            """the shift with respect to the reference"""
            shift[(reference,reference)] = (0,0,0)            
            """store shifts in the main round folder"""
            storepath = alignment_path + '\\' + 'shift.pickle'

            """pickle the data dict and store it"""
            with open(storepath, 'wb') as handle:
                pickle.dump(shift, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            df = pandas.DataFrame(calibrationscore)
            df.to_excel(alignment_path + '\\' +'calibrationscore.xlsx')
           
        """this is a manaul override of the automated alignment script by doing this it forces the shfit to accept this z-position"""        
        for roundpair,coordinate in shift.items():
            reference,r = roundpair
            x,y,z = coordinate
            """if there is a manual correction for the shift then the previous zshift calculation will be adjusted"""
            if r in manual_correction:
                z = manual_correction[r]
                
            """the new roundpair"""
            shift[roundpair] = (x,y,z) 
            
        print(shift)
        """store the image correction overlap"""
        StoreImageCorrectionOverlap(shift,paths,path = alignment_path + "\\")   


            # shift[roundpair] = (10,10,10)
        """4) DECODE THE IMAGES
              we take the repeated segmentations use these to align the 
              images between rounds, on 'bad' microscopes we need to stitch
              images, this allows us to decode""" 
        """global image alignment based on local features"""
        
        loadpath    = result_path + '\\Decoding\\decode.pickle'   
        pickle_in   = open(loadpath,"rb")
        self.decode = pickle.load(pickle_in)
        Coordinates = {self.decode['Coordinates'][i]:self.decode["Targets"][i] for i in range(len(self.decode["Coordinates"])) if self.decode["Targets"][i] != ''}
        self.CSF = rounds[min(rounds.keys())].CSF 
        self.CSF.global_position(Coordinates)
        self.CSF.reconstruct_cells()
        
        number = self.CSF.cellnumber
        storepath = result_path + '\\Decoding\\' + 'number.pickle'
        """pickle the data dict and store it"""
        with open(storepath, 'wb') as handle:
            pickle.dump(number, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        state = self.CSF.cellstate
        storepath = result_path + '\\Decoding\\' + 'cellstate.pickle'
        """pickle the data dict and store it"""
        with open(storepath, 'wb') as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            
        try:
            loadpath    = result_path + '\\Decoding\\decode.pickle'   
            pickle_in   = open(loadpath,"rb")
            self.decode = pickle.load(pickle_in)
            Coordinates = {self.decode['Coordinates'][i]:self.decode["Targets"][i] for i in range(len(self.decode["Coordinates"])) if self.decode["Targets"][i] != ''}
            
            self.CSF.global_position({"Coordinates":Coordinates},datakeys = ["Coordinates"])
            self.fraction = self.CSF.fraction
            storepath = result_path + '\\Decoding\\' + 'fraction.pickle'
            with open(storepath, 'wb') as handle:
                pickle.dump(fraction, handle, protocol=pickle.HIGHEST_PROTOCOL)   
            plot.CountPlotData(self.decode)
            plot.HybridizationMetrics(self.decode)
        except:   
            imsize_x,imsize_y = imsize
            """get the round object and loop through the signals
            and place them in the alignment pillar
            if the signal is present add to list, 
            if it is not then pass,
            this needs to be sorted by z-stack this is an ugly function, 
            first i generate all the information
            in position -> round -> channel -> signal 
            then go back up from signal to positition 
            to extract the information """
            """making the directories for the decoding and the figures"""
            try:
                os.mkdir(result_path + '\\' + 'Decoding\\')
            except:
                print('decode path already exists') 
            
            """allowed PSF"""  
            loc = [(0,0),(-1,-1),(1,1),(-1,1),(1,-1),(0, 1), (0, -1),(-1, 0), (1, 0),(-2, 0), (0, 2), (-1, -2),
                    (-2, 1), (2, 0), (1, -2), (0, -2),(-1, 2),(2, -1), (1, 2), (2, 1), (-2, -1)]
            
            
            from collections import defaultdict 
            c_by_crd  = defaultdict(list)
            c_by_set  = defaultdict(list)
    
            """get the rounds and the signals within the rounds' channels"""
            for r,hybridization in rounds.items():
                for c,channel in hybridization.channels.items():
                    for signal in channel.signals:
                        signal.shift_correction(shift,reference)
                        c_by_crd[signal.adjusted_coordinate].append(signal) 
    
            """track what has been added already"""
            tracker  = defaultdict(lambda: False)
       
            """the set with neighboring coordinates for i"""
            for r,hybridization in rounds.items():
                for c,channel in hybridization.channels.items():
                    for i in channel.signals:
                        x,y,z = i.adjusted_coordinate
                        if tracker[(x,y,z)] == False:
                            neighborset = tuple(sorted([(x+xs,y+ys,z) for xs,ys in loc]))
                            for n in neighborset:
                                if tracker[n] == False:
                                    c_by_set[neighborset] += c_by_crd[n]
                                    tracker[n] = True
                                    
            """all the barcodes that we have decoded (succesfully or otherwise)"""
            barcodes = []
            
            """decode the samples"""
            i = 0
            for c,signals in c_by_set.items():
                if i%int(len(c_by_set)/100) == 0:
                    print(
                        '\rDecoding Sample at [%d%%]'%int((i/float(len(c_by_set)))*100), end=""
                        ) 
                if len(signals) > minimal_barcode_length:
                    """decode the signals"""
                    code = Barcode(c,signals)
                    code.decode(codebook,codemap)  
    
                    """get the list if codes"""
                    barcodes.append(code)
                i += 1
                
            """see what the rotation and average distances are 
            for provisionally accepted barcodes in the dataset"""        
            bias = metrics([barcode.metrics for barcode in barcodes if barcode.accept == True])
     
            """with the current succesful decode round 1) calculate the metrics 2) decode remaining signals"""
            i = 0
            for code in barcodes:
                if i%int(len(barcodes)/100) == 0:
                    print(
                        '\rDeconvolution Sample at [%d%%]'%int((i/float(len(barcodes)))*100), end=""
                        ) 
                code.deconvolute(codebook,codemap)
                """update the i"""
                i += 1
                
            """get the targets in the dataset deemed highly likely"""
            import random
            from collections import Counter
            import pandas as pd
            targets = Counter([code.deconvoluted_target for code in barcodes if code.deconvoluted_target])
            
            """assign a target to the dataset"""
            for code in barcodes:
                code.assign_target(targets)
    
            """the fraction of total"""
            count    = Counter([code.target for code in barcodes if code.target != ''])
            
            """the percentage change"""
            total = sum([i for i in count.values()])
            fraction = {target:c/float(total) for target,c in count.items()}
            
            """awnser to everything"""
            random.seed(42)
            
            """the defaultdict to count the frequency a code occurs""" 
            for code in barcodes:
                if code.proportional_assignment == True:
                    if code.combined_assignment == False:
                        probability = {}
                        for t in code.potential_target:
                            if t in fraction:
                                """get dict and select"""
                                probability[t] = fraction[t] 
                            else:
                                """get dict and select"""
                                probability[t] = 0
                                
                        """the selection"""
                        code.target = random.choices(list(probability.keys()), weights=probability.values(), k=1)[0]
          
            """the defaultdict to count the frequency a code occurs"""
            for code in barcodes:
                if code.proportional_assignment == True:
                    if code.combined_assignment == True:
                        
                        probability = {}
                        """calculate the proportional chance a target is present"""
                        for t_1,t_2 in code.combined_potential_target:
                            if t_1 in fraction:
                                p_1 = fraction[t_1]
                            else:
                                p_1 = 0
                            if t_2 in fraction:
                                p_2 = fraction[t_2]
                            else:
                                p_2 = 0   
                                
                            """combined target"""
                            probability[(t_1,t_2)] = p_1 + p_2
    
                        """the selection"""
                        code.target = random.choices(list(probability.keys()), weights=probability.values(), k=1)[0]
                        
            self.combined_assignment = []
            """split the doublet signals in two seperate signals with a single shift in coordinate"""
            for i in barcodes:
                shift = 1
                
                """get the combined assignment"""
                if i.combined_assignment == True:
                    """compute the centroid of the target in the sample"""
                    x,y,z = i.centroid
                    
                    """the targets in the dataset"""
                    t_1,t_2 = i.target
                    
                    """new barcodes in the dataset"""
                    barcode_1 = copy.copy(i)
                    barcode_1.centroid = (x,y,z)
                    barcode_1.target = t_1
                    barcode_1.combined_assignment = False
                    
                    """new barcodes in the dataset"""
                    barcode_2 = copy.copy(i)
                    barcode_2.centroid = (x+shift,y,z)
                    barcode_2.target = t_2
                    barcode_2.combined_assignment = False
                    
                    """append the barcodes"""
                    barcodes.append(barcode_1)
                    barcodes.append(barcode_2)
                    
                    self.combined_assignment.append((t_1,t_2))
                    
            barcodes    = [i for i in barcodes if not i.combined_assignment]
            coordinates = [i.centroid for i in barcodes]
            
            """print the counter i.e. number of targets present"""
            print(Counter([code.target for code in barcodes if code.target != '']))
            
            """get the targets and put them in a dict"""
            targets = dict(zip(coordinates,barcodes))
            
            "0) unpack the signals and create a dependency graph to find duplicates" 
            cts = {i:True for i in coordinates}
            
            "1  ) build k-d tree"
            kdt = cKDTree(numpy.array(coordinates))
            edges = kdt.query_pairs(1)
            
            "2) create graph"
            G = nx.from_edgelist(edges)
            
            "3) Find Connections"
            connnected_components = nx.connected_components(G)
            df = pd.DataFrame(coordinates, columns=['x','y','z'])
            node_component = {v:k for k,connect in enumerate(connnected_components) for v in connect}
            df['c'] = pd.Series()
            
            "4) extract features and build sets"
            sets = defaultdict(list)
            for k,v in node_component.items():
                sets[v].append((df['x'][k],df['y'][k],df['z'][k]))
                
            "5) filter out the duplicate signals i.e. stacked ontop in z-stack"""
            remove = []
            for i,potential_duplicate_clusters in sets.items():
                cluster     = [targets[j].target for j in potential_duplicate_clusters]
                duplicates  = [n for n in ListDuplicates(cluster)]
                for target,indices in duplicates:
                    remove.append(potential_duplicate_clusters[indices[0]])
    
            """ 6) del duplicates in the targetset"""
            for i in remove:
                del targets[i] 
    
            """ 7) extract the composition of the spots over time 
            this will allow us to assess if we misalign something"""
            hybridization_in_observed_signal  = {'Round':[],'Fluorophore':[],'Target':[]}
            for c,t in targets.items():
                for r, colors in t.hybridization_data.items():
                    for n in colors:
                        hybridization_in_observed_signal['Round'].append(r)
                        hybridization_in_observed_signal['Fluorophore'].append(n) 
                        if t.target != '':
                            hybridization_in_observed_signal['Target'].append(True)
                        else:
                            hybridization_in_observed_signal['Target'].append(False)
    
            """ 8) Repeat for colocolization"""
            colocolization = ColocolizationGraph(targets)
    
            """9) congratulations you completed your 9 step program now 
            self.data is dataset where all relevant data is stored 
            (easy PANDAS conversion for plotting or excell)"""
            self.decode = {}
            
            """where was the target found, its signal and the species"""
            self.decode["Coordinates"]        = list(targets.keys())
            self.decode["Codes"]              = [targets[i].raw_initial_signal_code for i in self.decode["Coordinates"] ] 
            self.decode["Targets"]            = [targets[i].target for i in self.decode["Coordinates"]]
            
            """the data relevant to the decoding and target assignment"""
            self.decode["Colocalization"]     = colocolization
    
            """the metrics to assess the decoding process"""
            self.decode['Hybridization Data'] = hybridization_in_observed_signal
    
            """the data and path where the data is stored"""
            import pickle
            """the shift benchmark is to check if the alignment was succesfull"""
            storepath = result_path + '\\Decoding\\' + 'decode.pickle'
            with open(storepath, 'wb') as handle:
                pickle.dump(self.decode, handle, protocol=pickle.HIGHEST_PROTOCOL)      
                
            """plotting the figures from the dataset"""
            plot.CountPlotData(self.decode)
            plot.HybridizationMetrics(self.decode)

            
            """count the entire combined dataset"""
            self.countable = Counter(self.combined_assignment)

            """get the count table from duplicate assignments"""
            heatmap = {}    
            
            targets = []
            for x,y in self.countable.keys():
                targets.append(x)
                targets.append(y)
            targets = list(set(targets))
                
            for i in targets:
                heatmap[i] = {n:0 for n in targets}

            for double,count in self.countable.items():
                x,y = double
                heatmap[x][y] = count
                heatmap[y][x] = count
            
            import pickle
            """the shift benchmark is to check if the alignment was succesfull"""
            storepath = result_path + '\\Decoding\\' + 'colocalizationheatmap.pickle'
            with open(storepath, 'wb') as handle:
                pickle.dump(heatmap , handle, protocol=pickle.HIGHEST_PROTOCOL)  
            

        """5) ASSIGN THE SPOTS TO CELLS AND NUCLEUS OR CYTOPLASM
              we assign the spots to cells and check if the target
              is detected in the nucleus and the cytoplasm"""
        import os
        if os.path.isfile(result_path + '\\Decoding\\data.pickle'):
            pickle_in = open(result_path   + '\\Decoding\\data.pickle' ,"rb")
            self.data = pickle.load(pickle_in)
            self.data = pandas.DataFrame(self.data)
            pickle_in = open(result_path   + '\\Decoding\\rawdata.pickle' ,"rb")
            self.rawdata = pickle.load(pickle_in)
            
            import seaborn as sns
            plt.figure(figsize = (4,4))
            sns.scatterplot(data=self.rawdata,x='y',y='x',hue = 'Cell',style = 'Target',legend = False)
            plt.savefig(result_path + '\\Cells.png',dpi = 650)
            plt.show()
            # sns.set_context('notebook', font_scale = 1.5)
            plt.figure(figsize = (15,15))
            sns.countplot(data=self.rawdata,y='Target',hue = 'Cell')
            plt.savefig(result_path + '\\Count.png',dpi = 650)
            plt.show()

        else:
            """define the new dataset where we check the local position of the data"""
            Coordinates = {self.decode['Coordinates'][i]:self.decode["Targets"][i] for i in range(len(self.decode["Coordinates"])) if self.decode["Targets"][i] != ''}
        
            """the CSF i.e. independent cell locations"""
            self.CSF.local_position({"Coordinates":Coordinates},datakeys = ["Coordinates"])
            self.data = self.CSF.metadata

            metadata = {"Cell"                         : [],
                        "Cell Centroid"                : [],
                        'Target'                       : [],
                        "Position"                     : [],
                        "Spots Per Cell"               : [],
                        "Nuclear Fraction"             : [],
                        "Average Distance to Centroid" : [],
                        'Detection Method'             : []}

            for t,data in self.data.items():
                for key,value in data.items():
                    for n in value:
                        if key == "Cell":
                            metadata["Cell"].append(n)
                            metadata["Detection Method"].append(t)
                            metadata["Position"].append(position)
                        else:
                            metadata[key].append(n)

            import pickle
            """the shift benchmark is to check if the alignment was succesfull"""
            storepath = result_path + '\\Decoding\\' + 'rawdata.pickle'
            with open(storepath, 'wb') as handle:
                pickle.dump(self.CSF.rawdata , handle, protocol=pickle.HIGHEST_PROTOCOL)  
            """the shift benchmark is to check if the alignment was succesfull"""
            storepath = result_path + '\\Decoding\\' + 'data.pickle'
            with open(storepath, 'wb') as handle:
                pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)   

#insert path where these can be found
codebook = ''
folder   = ''

"""build the codebook"""
code    = Codebook(codebook,{1:'G', 2:'R', 3:'B'})

"""start analysing the images in the folder
    the structure of the folder is organized around a position of the microscope:
        Position Folder
            Hybridization Round Folders
                Hybridization Round
                    Image Fluorophore 1
                    Image Fluorophore 2
                    Image Fluorophore 3
                    DAPI image for nuclei"""
                    
Position(folder,imsize = (512,512),codebook = code,codemap = {1:'G', 2:'R', 3:'B'})


