import os
import fnmatch
import string
import itertools
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
from skimage import color
from skimage import io
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import NearestCentroid
from sklearn.externals import joblib
from sklearn import feature_extraction

xtraindata = []
ytraindata = []
xtestdata = []
ytestdata = []
filemapper = []
coords = []
isTrain = "True"
isClassify = "True"
basePath = "..\\14501500\\"
f = open(basePath + 'PRLog.txt','w')

class Model:
    def SetVec(self, value):
        self.Vec = value
    def SetModelData(self, value):
        self.ModelData = value

def getContourVerts(cn):
    contours = None    
    for cc in cn.collections:
        paths = []        
        for pp in cc.get_paths():                           
            contours = (max(pp.vertices[:,0]) + min(pp.vertices[:,0])) / 2., (max(pp.vertices[:,1]) + min(pp.vertices[:,1])) / 2.
            break
        break
    return contours

def createModel(trainingModel,filename):      
    model = Model()
    trainingModel.fit(trainXVec, np.array(ytraindata))
    model.SetModelData(trainingModel)
    model.SetVec(vec)
    joblib.dump(model,basePath + filename + ".pkl", compress=9, cache_size=1000)   

def processFeatures(facetdata):
    data = []
    for facets in facetdata:
        ones = string.split(facets)
        allFacets = dict((x,1) for x in ones)
        data.append(allFacets)
    return data 

def predict(trainingModel,filename,xtestdata):      
    modelLoaded = Model()
    modelLoaded = joblib.load(basePath + filename + ".pkl")    
    xtestdata = processFeatures(xtestdata)
    trainX = [i for i in xtestdata]    
    vec = modelLoaded.Vec
    clf = modelLoaded.ModelData
    testXVec = vec.transform(trainX)
    predicted = clf.predict(testXVec)    
    return predicted


def createFeatures(coords,gridpoints,label):
    #Actual fearture generation is blackboxed as of now, hope you could crack this!! best of luck!! ;)
    return features


if isTrain == "True":
    for root, dirnames, filenames in os.walk(basePath + 'train'):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            img = color.rgb2gray(io.imread(os.path.join(root, filename)))
            # Resize image to 10% of the original size
            face = sp.misc.imresize(img, 0.10) / 255.
            X = np.reshape(face, (-1, 1))            
            connectivity = grid_to_graph(*face.shape)
            # number of clusters
            n_clusters = 4
            ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',connectivity=connectivity)
            ward.fit(X)
            label = np.reshape(ward.labels_, face.shape)                                    
            ytraindata.append(np.array((str(os.path.basename(root)))))            
            coords = []
            for l in range(n_clusters):                
                cset2 = plt.contour(label == l, contours=1,colors=[plt.cm.spectral(l / float(n_clusters)),])        
                xy = getContourVerts(cset2)                                
                coords.append([xy[0] , xy[1]])            
            plt.clf()            
            ax = plt.gca()
            ax.grid(True)            
            ax.xaxis.set_ticks(np.arange(10, 200, 10))
            ax.yaxis.set_ticks(np.arange(10, 200, 10))
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            gridpoints = list(itertools.product(xticks, yticks))            
            # to save cluster points as image file for analysis
            #=======================================
            #for cc in coords:
            #    plt.scatter(cc[0] , cc[1],c='red')
            #plt.savefig(basePath+ 'meshgrid\\' +os.path.basename(root) + filename )
            #=======================================            
            xtraindata.append(createFeatures(np.array(coords),gridpoints,str(os.path.basename(root))))                                 
    vec = feature_extraction.DictVectorizer()
    xtraindata = processFeatures(xtraindata)
    trainX = [i for i in xtraindata]
    trainXVec = vec.fit_transform(trainX)
    createModel(NearestCentroid(),"NearestCentroid")
    createModel(MultinomialNB(alpha=.01),"MultinomialNB")
    createModel(LogisticRegression(),"LogisticRegression")
        

if isClassify == "True":
    for root, dirnames, filenames in os.walk(basePath + 'test'):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            filemapper.append(os.path.basename(root) + "_" + filename)
            img = color.rgb2gray(io.imread(os.path.join(root, filename)))            
            face = sp.misc.imresize(img, 0.10) / 255.
            X = np.reshape(face, (-1, 1))            
            connectivity = grid_to_graph(*face.shape)            
            n_clusters = 60 
            ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                                            connectivity=connectivity)
            ward.fit(X)
            label = np.reshape(ward.labels_, face.shape)            
            tempxtraindata = []
            ytestdata.append(np.array((str(os.path.basename(root)))))
            coords = []
            for l in range(n_clusters):                
                cset2 = plt.contour(label == l, contours=1,colors=[plt.cm.spectral(l / float(n_clusters)),])        
                xy = getContourVerts(cset2)                
                tempxtraindata.append((xy[0] * xy[1]) + xy[1])
                coords.append([xy[0] , xy[1]])                             
            plt.clf()            
            ax = plt.gca()
            ax.grid(True)            
            ax.xaxis.set_ticks(np.arange(10, 200, 10))
            ax.yaxis.set_ticks(np.arange(10, 200, 10))
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            gridpoints = list(itertools.product(xticks, yticks))                        
            tempxtraindata1 = createFeatures(np.array(coords),gridpoints,str(os.path.basename(root)))
            xtestdata.append(tempxtraindata1)            
    p1 = predict(NearestCentroid(),"NearestCentroid",xtestdata)
    p2 = predict(MultinomialNB(alpha=.01),"MultinomialNB",xtestdata)    
    p3 = predict(LogisticRegression(),"LogisticRegression",xtestdata)    
    data = ""
    for i in range(0,len(filemapper)):                
        data += str(filemapper[i]) + "\t" + str(p1[i]) + "\t" + str(p2[i]) + "\t" + str(p3[i]) + "\n"        
    f.write(data)
    f.close()
