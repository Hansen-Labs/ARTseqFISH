# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:20:02 2022

@author: bobvansluijs
This is a script to count ARTSEQFISH spots in an image and localize them vis a vis the cell
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
from LocalFeatureAlignment import *

def getListOfFiles(dirName):
    import os
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def get_image_name(path):
    if '\\' in path:
        name = path.split('\\')[-1].split('.tif')[0]
    else:
        name = path.split('$\$')[-1].split('.tif')[0]
    return name

def FileByExtention(dirName,tag = ['0.tif']):
    listOfFiles = getListOfFiles(dirName)
    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    #sort by folder name where results need to be stored
    folders = {}
    for i in listOfFiles:
        for n in tag:
            if n in i:      
                directory = ''
                for k in [i.split('\\')[n] for n in range(len(i.split('\\'))-1)]:
                    directory += k + '\\' 
                if directory not in folders:
                    folders[directory] = [i]
                else:
                    folders[directory].append(i)
    return folders

def get_channel(s):
    return s.split('.')[0][-1]

def get_target(s):
    return get_image_name(s)

def Create_XLSX_files(dirName):
    import os
    import pandas
    import seaborn as sns
    
    """we need to reconstruct the cells and the BCG"""
    from CellReconstruction import ReconstituteCell
    from CellReconstruction import build_coordinate_graph
    
    """file by extention, get the spotcount"""
    spotcountfolder = FileByExtention(dirName,tag = ['1.pickle','2.pickle','3.pickle'])
    """get the DAPI file extentions"""
    dapifolder = FileByExtention(dirName,tag = ['0.pickle'])
    
    """get the pickle and create a pandas file for each subdirectory"""
    plotdata = {"Folder"                            :[],  
                'Target'                            :[],  
                "Channel"                           :[],

                "Hybridization"                     :[],
                "Hybridization (AI)"                :[],
                "Target"                            :[],
                "Target (AI)"                       :[],
                
                "Cellnumber Graph Method"           :[],
                "Cellnumber Reconstruction Method"  :[],
                
                "Spots/Cell"                        :[],
                "Spots/Cell (AI)"                   :[]}
                
    
    metadata  = {"Cell":[],
             	"Cell Centroid":[],
                "Spots Per Cell":[],
             	"Nuclear Fraction":[],
             	"Average Distance to Centroid":[],
             	"Target":[],
             	"Detection Method":[],
                "Channel":[]}
    
    
    print(spotcountfolder)
    """folders in the dataset"""
    for folder,subdirs in reversed(spotcountfolder.items()):
        try:
            cells = 0
            if 'data.xlsx' not in os.listdir(folder):
                backup = False
                try:
                    """load the pickles and build the dataset"""
                    pickle_in = open(dapifolder[folder][0],"rb")
                    celldata  = pickle.load(pickle_in)
                    
                    """the cells we can observe in the dataset"""
                    CFS = ReconstituteCell(celldata) 
                    CFS.reconstruct_cells()
                    
                    """dataformat for the dataset"""
                    dataset = copy.deepcopy(plotdata)
    
                except:
                    pass
    
                for subdir in set(subdirs):
                    splitset = subdir.split('\\')[-1]
                    splitset = splitset.split('.pickle')[0][0:-1]
                # try:
                    for i in range(len(dapifolder[folder])):
                        if splitset != backup:
                            if splitset in dapifolder[folder][i]:
                                backup = copy.copy(splitset)
                                
                                """load the pickles and build the dataset"""
                                pickle_in = open(dapifolder[folder][i],"rb")
                                celldata  = pickle.load(pickle_in)
                                
                                """the cells we can observe in the dataset"""
                                CFS = ReconstituteCell(celldata) 
                                CFS.reconstruct_cells()
                            
                    """load the dataset"""
                    pickle_in = open(subdir,"rb")
                    data      = pickle.load(pickle_in)
    
                    hyb   = build_coordinate_graph(data['Hybridization GPS'],distance = 1)
                    hybAI = build_coordinate_graph(data['AI Hybridization GPS'],distance = 1)
    
                    # """get the coordinates out"""
                    targets   = [list(sorted(hyb[k]))[0] for k in hyb.keys()]
                    targetsAI = [list(sorted(hybAI[k]))[0] for k in hybAI.keys()]
    
                    """print the nymber of hybridizations"""
                    
                    """the counts in the network"""
                    dataset["Hybridization" ].append(data['Hybridization Number'])
                    dataset["Hybridization (AI)"].append(data['Hybridizations AI number'])
                    
                    """the target counts (filtered by z-stack)"""
                    dataset["Target"].append(len(hyb))
                    dataset["Target (AI)"].append(len(hybAI))
                    dataset['Spots/Cell'].append(float(len(hyb))/float(CFS.signalcount))
                    dataset['Spots/Cell (AI)'].append(float(len(hybAI))/float(CFS.signalcount))
                    
                    """cell number based on two methods"""
                    dataset['Cellnumber Graph Method'].append(CFS.graphcount)
                    dataset["Cellnumber Reconstruction Method"].append(CFS.signalcount)
    
                    """get the relevant metadata"""
                    dataset["Folder"]  = folder
                    dataset["Target"].append(get_target(subdir))
                    dataset["Channel"].append(get_channel(subdir))
    
    
                    CFS.global_position(data,["AI Hybridization GPS"])
                    for key,fraction in CFS.fraction.items():
                        if key not in dataset.keys():
                            dataset[key] = [fraction]
                        else:
                            dataset[key].append(fraction)

                    CFS.local_position(data,datakeys = ["AI Hybridization GPS"])
                    for link,vector in CFS.metadata.items():
                        for keys,data in vector.items():
                            if keys == 'Cells':
                                for i in data:
                                    metadata[keys].append(cells)
                                    metadata['Channel'].append(get_channel(subdir))
                                    metadata['Target'].append(get_target(subdir))
                                    metadata['Detection Method'].append('AI Hybridization GPS')
                                    cells += 1
                            else:
                                for i in data:
                                    metadata[keys].append(i)  
                    # except:
                    #     pass
     
                # """store the metadata"""
                # for k,v in metadata.items():
                #     print(len(v))
                    
                # for k,v in dataset.items():
                #     print(len(v))
                print(dataset)
                with open(folder+'metadata.pickle', 'wb') as handle:
                    pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
                with open(folder+'data.pickle', 'wb') as handle:
                    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
                # try:
                print(dataset)
                df = pandas.DataFrame(dataset)
                df.to_excel(folder + 'data.xlsx')
                
                print(metadata)
                df = pandas.DataFrame(metadata)
                df.to_excel(folder + 'metadata.xlsx')
                # except:
                    # pass
        except:
            print('something did not work sataset not created')
    
def CountandAnalyse(dirName,dapitag = '0.tif'):
    listOfFiles = getListOfFiles(dirName)
    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    #sort by folder name where results need to be stored
    folders = {}
    for i in listOfFiles:
        if '.tif' in i:
  
            if '\\' in i:
                strset = ''
                for k in [i.split('\\')[n] for n in range(len(i.split('\\'))-1)]:
                    strset += k + '\\'
            else:
                strset = ''
                for k in [i.split('$\$')[n] for n in range(len(i.split('$\$'))-1)]:
                    strset += k + '\\'
                
            """check if folder is present"""
            if strset in folders:
                folders[strset].append(i)
            else:
                folders[strset] = [i]

    from SpotDetection import Image
    # for folder,images in reversed(folders.items()):
    for folder,images in folders.items():
        for path in images:
            try:
                if dapitag in path:
                    if path.split('.')[0] + '.pickle' not in [folder + i for i in os.listdir(folder)]:
                        segmentation = Segmentation(path,AIenhanced = True,SVM = True,segmentsides = False)
                        CSF  = copy.copy(segmentation.CSF)
                        
                        """get image name to store the data"""
                        name = get_image_name(path)
                        storepath = folder + '\\' + name + '.pickle'
                        with open(storepath, 'wb') as handle:
                            pickle.dump(CSF.return_data_object(), handle, protocol=pickle.HIGHEST_PROTOCOL) 
                            
                        """clean up the memory"""
                        del segmentation
                        del CSF
                else:
                    if path.split('.')[0] + '.pickle' not in [folder + i for i in os.listdir(folder)]:
                        image = Image(path,plot = False)
                        count = image.count
                        name = get_image_name(path)
                        
                        """store the count and channel"""
                        storepath = folder + '\\' + name + '.pickle'
                        with open(storepath, 'wb') as handle:
                            pickle.dump(count, handle, protocol=pickle.HIGHEST_PROTOCOL)
                            
                        """clean up the memory"""
                        del image
            except:
                print('this path caused some issues, please revise')
                print(path + ": double check")
                


# def open_metapickle(path):
#     with open(path + "data.pickle", "rb") as f:
#         dictname = pickle.load(f)
#     for k,v in dictname.items():
#         print(k,len(v))
        
#     del dictname['Target']
#     del dictname['Folder']
    
#     import pandas
#     df = pandas.DataFrame(dictname)
#     df.to_excel(path + 'data.xlsx')
    
    
# #     # del
# path ='D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\'
# # open_metapickle(path)

# import pandas
# data = []
# for i in os.listdir(path):
#     m = path + i + '\\'
#     m = m.replace('._','')
#     # print(m)
#     for n in os.listdir(m):
#         print(n)
        
#         if 'data.xlsx' in m + '\\' + n:
#             if 'metadata' not in m + '\\' + n:
                
#                 data.append(pandas.read_excel(m + '\\' + n).to_dict())

# print(data)
# appended_dataset = {i:[] for i in data[-1].keys() if i != 'Unnamed: 0'}
# for i in data:
#     for key,value in i.items():
#         if key != 'Unnamed: 0':
#             for index,datapoint in value.items():
#                 appended_dataset[key].append(datapoint)

# appended = pandas.DataFrame(appended_dataset)
# appended.to_excel(path + 'appended_dataset.xlsx')
                 
