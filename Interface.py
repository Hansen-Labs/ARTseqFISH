# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:47:51 2024

@author: bob van sluijs
"""

from SpotDetection import *
from SpotDetection import Image

"""Example of spot detection, set path of software"""
path = ''

# open the image and test (pick image_C1, image_C2 or image_C3 for color channel), the first 10 z stack only noise
"""GPS is number of detected maxima
    Spots number of connections through z-stack
    AI is number with a gaussian PSF as tested by the SVM output
"""
image         = OpenImage(path + '\\Spot Demo\\test_image_C2.tif')
image_object  = Image(image)


folder = path + '\\Spot Demo\\Analysis\\'
image_object.plot(folder)

"""Example of segmentation"""
from NucleusSegmentation import Segmentation
nucleus_path = path + "\\DAPI Demo\\DAPI_test.tif"
nuclei = Segmentation(nucleus_path,show_bbox = True)

 
# from ARTseqFISH import Position
# from ARTseqFISH import Codebook
#insert path where these can be found
# codebook = path + '\\Codebook (Example)\\Book.xls'
# folder   = ''

# """build the codebook"""
# code  = Codebook(codebook,{1:'G', 2:'R', 3:'B'})

# """start analysing the images in the folder
#     the structure of the folder is organized around a position of the microscope:
#         Position Folder
#             Hybridization Round Folders
#                 Hybridization Round
#                     Image Fluorophore 1
#                     Image Fluorophore 2
#                     Image Fluorophore 3
#                     DAPI image for nuclei"""                  
# Position(folder,imsize = (512,512),codebook = code,codemap = {1:'G', 2:'R', 3:'B'})
 
