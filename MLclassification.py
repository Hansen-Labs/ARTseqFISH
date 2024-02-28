# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 17:57:01 2021

@author: bob van sluijs
"""
import os
import numpy
import matplotlib.pylab as plt
from ImageAnalysisOperations import *
#imports
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      The print function has been:
          modified from: Habernal, Ivan, et al. "Mining legal arguments in court decisions." Artificial Intelligence and Law (2023): 1-38.
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)

    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.1), (oText._x, oText._y), (oText._x, oText._y+0.1)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        #print '\n'

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 0.38]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 0.38]
        facecolors[posi] = carr

    else:
        if(per > 0):
            txt = '%s\n%.2f%%' %(cell_val, per)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #main diagonal
        if(col == lin):
            #set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del


def insert_totals(df_cm):
    """ insert total column and line (the last ones)
       modified from Habernal, Ivan, et al. "Mining legal arguments in court decisions." Artificial Intelligence and Law (2023): 1-38."""
    
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col


def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[9,9], show_null_values=0, pred_val_axis='y'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
       modified from Habernal, Ivan, et al. "Mining legal arguments in court decisions." Artificial Intelligence and Law (2023): 1-38.
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array(df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) 
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    # ax.set_xlim(bottom + 0.5, top - 0.5) # set the ylim to bottom, top# set the ylim to bottom, top
    plt.tight_layout()  #set layout slim
    plt.savefig('D:\\Appendix Figures\\confusion.png',dpi = 600)
    # plt.show()
    plt.close()

def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",
      fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[5,5], show_null_values=0, pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    #data
    if(not columns):
        #labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        #labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Oranges';
    fz = 11;
    figsize =[6,6];
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis)

         
class MLtrainer:
    def __init__(self,
                 folder = "Segmentation\\Nucleiisegments\\Training data\\"
                 ):
        
        """this class calls the test and training data files, we assume the relevant folder
        is present on the desktop of the user, the test and training data is provided"""
        
        """get the model images needed for training"""            
        desktop  = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop') + '\\'
        
        """nuclei segmentation is the default spot detection method"""
        folder   = desktop + folder
            
        """classified folders"""
        self.folders    = os.listdir(folder)
        self.folderpath = [folder + i for i in self.folders]
        
        """the folderpaths contain images resize the images in the folder paths"""
        self.datasets   = {}
        
    def SegmentationData(self,):
        from sklearn.model_selection import train_test_split
        for i in range(len(self.folderpath)):
            self.datasets[self.folders[i]] = resize_all(self.folderpath[i],"",self.folderpath[i])
            
        """training labels"""
        labels = {self.folders[i] :i for i in range(len(self.folders))}
        
        """"create test and training data"""
        self.images         = []
        self.classification = []
        
        """loop through the datasets"""
        for k,v in self.datasets.items():
            for i in v['images']:
                self.images.append(i)
                self.classification.append(labels[k])
      
        """training labels"""
        labels = {self.folders[i] :int(self.folders[i]) for i in range(len(self.folders))}
        
        """"create test and training data"""
        self.images         = []
        self.classification = []
        
        """loop trough the datasets"""
        for k,v in self.datasets.items():
            for i in v['images']:
                self.images.append(i)
                self.classification.append(labels[k])

        """the data and the samples"""
        data      = numpy.array(self.images)
        n_samples = len(self.images)
        
        """flatten the images"""
        self.flattened_images = data.reshape((n_samples, -1))
        self.classification   = numpy.array(self.classification)
        
        """create the dataset test, train etc."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.flattened_images, self.classification, test_size=0.25, shuffle=True)
        
    def PSFData(self,):
        from sklearn.model_selection import train_test_split
        """create thest and trainingdata"""
        self.images         = []
        self.classification = []
        
        """the folders that serve as classification"""
        for i in range(len(self.folderpath)):
            subpath = os.listdir(self.folderpath[i])
            for j in subpath:
                array = numpy.load(self.folderpath[i] + "\\" + j)
                if array.shape == (49,):
                    self.images.append(array)
                    self.classification.append(int(self.folders[i]))
                
        """flatten and create the dataset"""
        self.flattened_images = numpy.array(self.images).reshape((len(self.images), -1))
        self.classification   = numpy.array(self.classification)

        """create the dataset test, train etc."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.flattened_images, self.classification, test_size=0.5, shuffle=True)
        
    def manualPSFData(self,
                           psf
                      ):
        """function applies if you want to create test and training data on the fly
        you give it a tuple with lists containing the pointspread functions
        INPUT
            psf: (False:[ARR1,ARR2....ARRn],(True:[ARR1,ARR2....ARRn])"""
        """play around with the datasets"""
        from sklearn.model_selection import train_test_split
        
        """get the pointspreadfunctions of th data"""
        psf_0,psf_1 = psf

        """create thest and trainingdata"""
        self.images         = []
        self.classification = []

        """The psf_0 and the Psf_1 for a specific colorchannel"""        
        for i in psf_0:
            if i.shape == (7,7):
                self.images.append(numpy.array(i.reshape(len(i),-1)))
                self.classification.append(0)
            
        for i in psf_1:
            if i.shape == (7,7):
                self.images.append(numpy.array(i.reshape(len(i),-1)))
                self.classification.append(1)   
                
        """create the dataset test, train etc."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.images, self.classification, test_size=0.5, shuffle=True)
        

    def SVM_Classifier(self,
                       plot = False
                       ):
         #Standard scientific Python imports
        import matplotlib.pyplot as plt
        
        # Import datasets, classifiers and performance metrics
        from sklearn import datasets, svm, metrics
        from sklearn.model_selection import train_test_split
        
        # flatten the images
        n_samples = len(self.images)
        data = numpy.array(self.images)
        data = data.reshape((n_samples, -1))
        classification = self.classification
        
        # Create a classifier: a support vector classifier
        self.clf = svm.SVC(gamma=0.01)        
        # Learn the digits on the train subset
        self.clf.fit( self.X_train,  self.y_train)        
        # Predict the value of the digit on the test subset
        predicted = self.clf.predict(self.X_test)
        misclassified_samples =  self.X_test[ self.y_test != predicted]
        #the confusion and the matrix
        plot_confusion_matrix_from_data( self.y_test,predicted)
        
        
    def ANN_PSF(self,):
        import warnings
        import matplotlib.pyplot as plt
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import MinMaxScaler
        from sklearn import datasets
        from sklearn.exceptions import ConvergenceWarning
        from sklearn import datasets, svm, metrics
        from sklearn.model_selection import train_test_split
            
        # different learning rate schedules and momentum parameters
        params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
                  'learning_rate_init': 0.2},
                  {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
                  'nesterovs_momentum': False, 'learning_rate_init': 0.2},
                  {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
                  'nesterovs_momentum': True, 'learning_rate_init': 0.2},
                  {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
                  'learning_rate_init': 0.2},
                  {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
                  'nesterovs_momentum': True, 'learning_rate_init': 0.2},
                  {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
                  'nesterovs_momentum': False, 'learning_rate_init': 0.2},
                  {'solver': 'adam', 'learning_rate_init': 0.01}]
        
        labels = ["constant learning-rate", "constant with momentum",
                  "constant with Nesterov's momentum",
                  "inv-scaling learning-rate", "inv-scaling with momentum",
                  "inv-scaling with Nesterov's momentum", "adam"]
        
        plot_args = [{'c': 'red', 'linestyle': '-'},
                    {'c': 'green', 'linestyle': '-'},
                    {'c': 'blue', 'linestyle': '-'},
                    {'c': 'red', 'linestyle': '--'},
                    {'c': 'green', 'linestyle': '--'},
                    {'c': 'blue', 'linestyle': '--'},
                    {'c': 'black', 'linestyle': '-'}]
        
        def plot_on_dataset(
                            X, y, name
                            ):
            # for each dataset, plot learning for each learning strategy
            print("\nlearning on dataset %s" % name)
            plt.title(name)
        
            X = MinMaxScaler().fit_transform(self.X_train)
            mlps = []
        
            max_iter = 400
            for label, param in zip(labels, params):
                print("training: %s" % label)
                self.mlp = MLPClassifier(random_state=0,max_iter=max_iter, **param)
        
                # some parameter combinations will not converge as can be seen on the
                # plots so they are ignored here
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                            module="sklearn")
                    self.mlp.fit(X, self.y_train)
        
                mlps.append(self.mlp)
                print("Training set score: %f" % self.mlp.score(X, self.y_train))
                print("Training set loss: %f" % self.mlp.loss_)
            for mlp, label, args in zip(mlps, labels, plot_args):
                plt.plot(mlp.loss_curve_, label=label, **args)
            return mlps

        mlps = plot_on_dataset(self.images,self.classification,name='ANN training regimes')
        
        
        for mlp in mlps:
            predicted = mlp.predict(self.X_test)
            disp = metrics.confusion_matrix(self.y_test, predicted)

            
    def SVM_predict(self,
                    image
                    ):
        """resize the image and predict of the
        Nucleus as segemented is a false positive (wrong segementation) or true positive (correct segmentatation)"""
        from skimage.transform import resize
        
        """images are resized to 25x25 pixel objects"""
        image = numpy.array(resize(image, (25, 25)))
        
        """flatten the matrix"""
        data = image.reshape((1, -1))
        
        """return the prediction of the clf model prediction"""
        return self.clf.predict(data/numpy.max(data))
        
    def ANN_Classifier(self,):
        import warnings
        import matplotlib.pyplot as plt
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import MinMaxScaler
        from sklearn import datasets
        from sklearn.exceptions import ConvergenceWarning
        from sklearn import datasets, svm, metrics
        from sklearn.model_selection import train_test_split
            
        # different learning rate schedules and momentum parameters
        params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
                  'learning_rate_init': 0.2},
                  {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
                  'nesterovs_momentum': False, 'learning_rate_init': 0.2},
                  {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
                  'nesterovs_momentum': True, 'learning_rate_init': 0.2},
                  {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
                  'learning_rate_init': 0.2},
                  {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
                  'nesterovs_momentum': True, 'learning_rate_init': 0.2},
                  {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
                  'nesterovs_momentum': False, 'learning_rate_init': 0.2},
                  {'solver': 'adam', 'learning_rate_init': 0.01}]
        
        labels = ["constant learning-rate", "constant with momentum",
                  "constant with Nesterov's momentum",
                  "inv-scaling learning-rate", "inv-scaling with momentum",
                  "inv-scaling with Nesterov's momentum", "adam"]
        
        plot_args = [{'c': 'red', 'linestyle': '-'},
                    {'c': 'green', 'linestyle': '-'},
                    {'c': 'blue', 'linestyle': '-'},
                    {'c': 'red', 'linestyle': '--'},
                    {'c': 'green', 'linestyle': '--'},
                    {'c': 'blue', 'linestyle': '--'},
                    {'c': 'black', 'linestyle': '-'}]
        
        def plot_on_dataset(
                            X, y, name
                            ):
            # for each dataset, plot learning for each learning strategy
            print("\nlearning on dataset %s" % name)
            plt.title(name)
        
            X = MinMaxScaler().fit_transform(self.X_train)
            mlps = []
        
            max_iter = 400
            for label, param in zip(labels, params):
                print("training: %s" % label)
                self.mlp = MLPClassifier(random_state=0,max_iter=max_iter, **param)
        
                # some parameter combinations will not converge as can be seen on the
                # plots so they are ignored here
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                                            module="sklearn")
                    self.mlp.fit(X, self.y_train)
        
                mlps.append(self.mlp)
                print("Training set score: %f" % self.mlp.score(X, self.y_train))
                print("Training set loss: %f" % self.mlp.loss_)
            for mlp, label, args in zip(mlps, labels, plot_args):
                plt.plot(mlp.loss_curve_, label=label, **args)
            return mlps

        """show the results of the trained models"""
        mlps = plot_on_dataset(self.flattened_images,self.classification,name='ANN training regimes')
        
        for mlp in mlps:
            predicted = mlp.predict(self.X_test)
            disp = metrics.confusion_matrix(mlp, self.X_test, self.y_test)
            disp.figure_.suptitle("Confusion Matrix")
            plt.show()

    def SVM_NucleiiTest(self,
                        paths = []
                        ):
        from NucleusSegmentation import SegmentNucleus
        """the paths in the list"""
        for path in paths:
            """path of the image"""
            imagename = path.split("\\")[-1].split(".")[0]
            """get the paths"""
            desktop  = os.path.join(os.path.join(os.path.expanduser('~')), 'Onedrive\\Desktop') + '\\Segmentation\\'
            dataloc  = desktop + "Nucleiisegments\\Test data\\" + imagename + "\\"
            try:
                 os.makedirs(dataloc)
            except:
                print('image file exists')
            images   = os.listdir(dataloc)            
                
            """do the segmentation"""
            nucleus = SegmentNucleus(path,AIenhanced = True)
            
            """add the classified images to a new folder"""
            classes = set([i.prediction[0] for i in nucleus.classification])
            
            """location of the classes"""
            classpath = {}
            for i in classes:
                try:
                    os.makedirs(dataloc + str(i))
                    classpath[i] = dataloc + str(i)
                except:
                    classpath[i] = dataloc + str(i)
       
            for i in nucleus.classification:
                fig = plt.figure(frameon=False)
                plt.margins(0,0)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                path = classpath[i.prediction[0]]
                length = len(os.listdir(path))
                plt.imsave(path + '\\' + str(length) + '.png',i.warped)
                plt.close()

    def SVM_MembraneTest(self,
                         paths = []
                         ):
        
        from NucleusSegmentation import SegmentNucleus
        """the paths in the list"""
        for path in paths:
            """path of the image"""
            imagename = path.split("\\")[-1].split(".")[0]
            """get the paths"""
            desktop  = os.path.join(os.path.join(os.path.expanduser('~')), 'Onedrive\\Desktop') + '\\Segmentation\\'
            dataloc  = desktop + "MembraneSegments\\Test data\\" + imagename + "\\"
            try:
                 os.makedirs(dataloc)
            except:
                print('image file exists')
            images   = os.listdir(dataloc)            
                
            """do the segmentation"""
            nucleus = SegmentMembrane(path,AIenhanced = True)
            
            """add the classified images to a new folder"""
            classes = set([i.prediction[0] for i in nucleus.classification])
            
            """location of the classes"""
            classpath = {}
            for i in classes:
                try:
                    os.makedirs(dataloc + str(i))
                    classpath[i] = dataloc + str(i)
                except:
                    classpath[i] = dataloc + str(i)
       
            for i in nucleus.classification:
                
                fig = plt.figure(frameon=False)
                plt.margins(0,0)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                path = classpath[i.prediction[0]]
                length = len(os.listdir(path))
                plt.imsave(path + '\\' + str(length) + '.png',i.warped)
                plt.close()
