# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 12:02:24 2022

@author: bob van sluijs
"""

"""call the spotcountscript and store the data
this script does the following 

it goes into folder depth n to find images marked with .tif extention
    it subsequently seperates DAPI files from spotdetection ARTseqfish files
    it stores the resuls in the same images as show"""
    
# CountandAnalyse()
# CountandAnalyse("C:\\Users\\hanse\\Desktop\\Need to count\\")  

# path = 'D:\\Spots Counting - Abseq_XYH_50\\Ab test\\AKT YAP CX2_01_20210512_113911\\AKT YAP CX2_01_w0000.pickle'
# path = 'D:\\Spots Counting - Abseq_XYH_50\\RNA Protein\\'
# path = 'D:\\Spots Counting - Abseq_XYH_50\\'


# path = "D:\\Spots Counting - Abseq_XYH_50\\"
# path = "D:\\Spots Counting - Abseq_XYH_50\\Colocalization Batch\\"
# path = "D:\\Spots Counting - Abseq_XHY_51_B\\"
import os
from CountSpots import *
# from Figures import *

"""count the spots in this folder"""
# path = "C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\Bra GATA6 CDX2\\"
# CountandAnalyse(path)
# path = 'C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\FOXA2 YAP SNAIL\\'
# CountandAnalyse(path)
# path = 'C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\Nanog\\'
# CountandAnalyse(path)
# path = 'C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\OCT4 Sox2 Sox17\\'
# CountandAnalyse(path)


# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\SMAD2 inhibitor activator\\3h ACTIVIN\\"
# CountandAnalyse(path)
# path = 'D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\SMAD2 inhibitor activator\\3h SB43542\\'
# CountandAnalyse(path)
# path = 'D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\SMAD2 inhibitor activator\\6h ACTIVIN\\'
# CountandAnalyse(path)
# path = 'D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\SMAD2 inhibitor activator\\6h SB43542\\'
# CountandAnalyse(path)

"""count the spots in this folder"""
# path = "C:\\Users\\hanse\\Desktop\\Psmad\\3T3\\"
# CountandAnalyse(path)
# path = 'C:\\Users\\hanse\\Desktop\\Psmad\\MCF7\\'
# CountandAnalyse(path)
# path = "C:\\Users\\hanse\\Desktop\\Psmad\\NEG 3T3\\"
# Create_XLSX_files(path)
# path = 'C:\\Users\\hanse\\Desktop\\Psmad\\MCF7\\'
# Create_XLSX_files(path)



# path = "D:\\Correlation\\Correlation_0H\\"
# CountandAnalyse(path)
# path = "D:\\Correlation\\Correlation_12H\\"
# CountandAnalyse(path)
# path = "D:\\Correlation\\Correlation_24H\\"
# CountandAnalyse(path)
# path = "D:\\Correlation\\Correlation_48H\\"
# CountandAnalyse(path)

# path = "D:\\Correlation\\Correlation_0H\\"
# Create_XLSX_files(path)
# path = "D:\\Correlation\\Correlation_12H\\"
# Create_XLSX_files(path)
# path = "D:\\Correlation\\Correlation_24H\\"
# Create_XLSX_files(path)
# path = "D:\\Correlation\\Correlation_48H\\"
# Create_XLSX_files(path)


# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\0h_20240110_181240\\"
# CountandAnalyse(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\3h_ACTIVIN_1_20240110_164931\\"
# CountandAnalyse(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\3h_ACTIVIN_2_20240110_171122\\"
# CountandAnalyse(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\3h_SB43542_1_20240110_173202\\"
# CountandAnalyse(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\3h_SB43542_2_20240110_175247\\"
# CountandAnalyse(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\6h_ACTIVIN_1_20240110_140254\\"
# CountandAnalyse(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\6h_ACTIVIN_2_20240110_142824\\"
# CountandAnalyse(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\6h_SB43542_1_20240110_133800\\"
# CountandAnalyse(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\6h_SB43542_2_20240110_145107\\"
# CountandAnalyse(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\12h_ACTIVIN_1_20240110_152337\\"
# CountandAnalyse(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\12h_ACTIVIN_2_20240110_162426\\"
# CountandAnalyse(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\12h_SB43542_1_20240110_155328\\"
# CountandAnalyse(path)


# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\0h_20240110_181240\\"
# Create_XLSX_files(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\3h_ACTIVIN_1_20240110_164931\\"
# Create_XLSX_files(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\3h_ACTIVIN_2_20240110_171122\\"
# Create_XLSX_files(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\3h_SB43542_1_20240110_173202\\"
# Create_XLSX_files(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\3h_SB43542_2_20240110_175247\\"
# Create_XLSX_files(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\6h_ACTIVIN_1_20240110_140254\\"
# Create_XLSX_files(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\6h_ACTIVIN_2_20240110_142824\\"
# Create_XLSX_files(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\6h_SB43542_1_20240110_133800\\"
# Create_XLSX_files(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\6h_SB43542_2_20240110_145107\\"
# Create_XLSX_files(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\12h_ACTIVIN_1_20240110_152337\\"
# Create_XLSX_files(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\12h_ACTIVIN_2_20240110_162426\\"
# Create_XLSX_files(path)
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\12h_SB43542_1_20240110_155328\\"
# Create_XLSX_files(path)


"""count the spots in this folder"""




# data = []
import pandas
# for path in ['C:\\Users\\hanse\\Desktop\\Psmad\\NEG 3T3\\',
#               ]:
#     for i in os.listdir(path):
#         m = path + i + '\\'
#         m = m.replace('._','')
#         # print(m)
#         for n in os.listdir(m):
#             if 'meta' not in  m + '\\' + n:
#                 if 'data.pickle' in m + '\\' + n:
#                     with open(m + '\\' + n, "rb") as f:
#                         d = pickle.load(f)
#                         data.append(d)
# appended_dataset = {i:[] for i in data[-1].keys() if i != 'Unnamed: 0'}
# for i in data:
#     for key,value in i.items():
#         if key == 'Target':
#             value = [i for i in value if type(i) != int]
#         if key != 'Folder':
#                 appended_dataset[key].extend(value)
#         else:
#                 appended_dataset[key].append(value)
#         if key == 'Folder':
#             for n in range(len(i['Channel'])-1):
#                 appended_dataset[key].append(value)
        

# appended = pandas.DataFrame(appended_dataset)
# appended.to_excel(path + 'appended_dataset.xlsx')
# def open_metapickle(path):
#     with open(path + "data.pickle", "rb") as f:
#         dictname = pickle.load(f)
    # print(dictname)
    
    # import pandas
    # df = pandas.DataFrame(dictname)
    # df.to_excel(path + 'data.xlsx')
    
# path = "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\SMAD2 inhibitor activator\\3h ACTIVIN\\"
# Create_XLSX_files(path)
# path = 'D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\SMAD2 inhibitor activator\\3h SB43542\\'
# Create_XLSX_files(path)
# path = 'D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\SMAD2 inhibitor activator\\6h ACTIVIN\\'
# Create_XLSX_files(path)
# path = 'D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\SMAD2 inhibitor activator\\6h SB43542\\'
# Create_XLSX_files(path)

    
# """count the spots in this folder"""
# path = "C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\Bra GATA6 CDX2\\"
# Create_XLSX_files(path)
# path = 'C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\FOXA2 YAP SNAIL\\'
# Create_XLSX_files(path)
# path = 'C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\Nanog\\'
# Create_XLSX_files(path)
# path = 'C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\OCT4 Sox2 Sox17\\'
# Create_XLSX_files(path)

# path = 'C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\Bra GATA6 CDX2\\Bra GATA6 CDX2_DMSO_24h_03_20231123_181502\\'
# open_metapickle(path)

data = []
# path = 'C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\Nanog\\'
# path = 'C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\OCT4 Sox2 Sox17\\'
# path = 'C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\FOXA2 YAP SNAIL\\'
# path = "C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\Bra GATA6 CDX2\\"

# for path in ["C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\Bra GATA6 CDX2\\",
#              'C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\FOXA2 YAP SNAIL\\',
#              'C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\OCT4 Sox2 Sox17\\',
#              'C:\\Users\\hanse\\Desktop\\Abseq_XYH_062\\Nanog\\']:
    
for path in [ "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\"]:
    for i in os.listdir(path):
        m = path + i + '\\'
        # m = m.replace('._','')
        # print(m)
        for n in os.listdir(m):

            if 'txt' not in m + '\\' + n + '\\' :
                for k in os.listdir(m + '\\' + n + '\\'):
                    if 'meta' not in  m + '\\' + n + '\\' + k:
                            print('asdfasdfadf',m + '\\' + n + '\\' + k)
                            if 'data.pickle' in m + '\\' + n + '\\' + k:
                                with open(m + '\\' + n + '\\' + k, "rb") as f:
                                    d = pickle.load(f)
                                    data.append(d)
                                    


appended_dataset = {i:[] for i in data[-1].keys() if i != 'Unnamed: 0'}
for i in data:
    for key,value in i.items():
        if key == 'Target':
            value = [i for i in value if type(i) != int]
        if key != 'Folder':
                appended_dataset[key].extend(value)
        else:
                appended_dataset[key].append(value)
        if key == 'Folder':
            for n in range(len(i['Channel'])-1):
                appended_dataset[key].append(value)
        

appended = pandas.DataFrame(appended_dataset)
appended.to_excel(path + 'appended_dataset.xlsx')
                 

# for path in [ "D:\\Figures Final Datasets (experiments)\\Revision (datasets)\\Abseq_XYH_064\\"]:
#     for i in os.listdir(path):
#         m = path + i + '\\'
#         # m = m.replace('._','')
#         # print(m)
#         for n in os.listdir(m):

#             if 'txt' not in m + '\\' + n + '\\' :
#                 for k in os.listdir(m + '\\' + n + '\\'):
#                     # if 'meta' not in  m + '\\' + n + '\\' + k:
#                     #         print('asdfasdfadf',m + '\\' + n + '\\' + k)
#                             if 'metadata.pickle' in m + '\\' + n + '\\' + k:
#                                 with open(m + '\\' + n + '\\' + k, "rb") as f:
#                                     d = pickle.load(f)
#                                     data.append(d)
                                    


# appended_dataset = {i:[] for i in data[-1].keys() if i != 'Unnamed: 0'}
# for i in data:
#     for key,value in i.items():
#         if key == 'Target':
#             value = [i for i in value if type(i) != int]
#         if key != 'Folder':
#                 appended_dataset[key].extend(value)
#         else:
#                 pass
#         if key == 'Folder':
#             for n in range(len(i['Channel'])-1):
#                 appended_dataset[key].append(value)
        

# appended = pandas.DataFrame(appended_dataset)
# appended.to_excel(path + 'appended_dataset_meta.xlsx')

# for i in range(len(os.listdir(path))):
# paths = os.listdir(path)
# p = path + paths[10]
# print(p)
# print(p)
# Create_XLSX_files(p)
# CountandAnalyse(path)
# datasets = CreateDataset_Figure_1(path)

# """create the excell files that are needed"""
# paths = os.listdir(path)
# for i in range(len(paths)):
#     i = paths[i]
#     if '.txt' not in i:
#         p = path + i
#         CreateXLSXfiles(p)
        
# path = "D:\\Spots Counting - Abseq_XYH_51\\"