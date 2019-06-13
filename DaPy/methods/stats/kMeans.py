# user/bin/python
#########################################
# Author         : Feichi Yang
# Edited by      : Xuansheng Wu    
# Email          : wuxsmail@163.com 
# Created        : 2018-12-01 00:00 
# Last modified  : 2018-12-01 11:09
# Filename       : DaPy.stats.KMeans
# Description    : kMeans method in DaPy                    
#########################################

from collections import namedtuple
from math import sqrt
from warnings import warn

from DaPy.core import DataSet, Frame
from DaPy.core import Matrix as mat
from DaPy.core import is_math, is_seq
from DaPy.matlib import _abs as abs
from DaPy.matlib import _sum as sum
from DaPy.matlib import corr, log, mean
from DaPy.methods.activation import UnsupportTest
from DaPy.methods.tools import engine2str, str2engine

warn('this model is developing, it is still unstable right now!')

class kMeans(object):
    def __init__(self, engine):
        pass

    def distEclud(self, vecA, vecB):
        return dp.sqrt(sum((vecA - vecB) ** 2))

    def fit(self, k, data):
        dataMat = mat(data)
        m, n = dataMat.shape
        # create centroid mat
        centroids = mat(zeros((k, n)))
        # create random cluster centers, within bounds of each dimension
        for j in range(n):  
            minJ = min(dataMat[:, j])
            rangeJ = float(max(dataMat[:, j]) - minJ)
            centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))

        # create mat to assign data points to a centroid, also holds SE of each point
        clusterAssment = self._engine.zeros((m, 2))  
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            # for each data point assign it to the closest centroid
            for i in range(m):  
                minDist = inf
                minIndex = -1
                for j in range(k):
                    distJI = self.distEclud(centroids[j, :], dataMat[i, :])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                if clusterAssment[i, 0] != minIndex:
                    clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2

            # recalculate centroids
            for cent in range(k):
                # get all the point in this cluster
                ptsInClust = dataMat[nonzero(clusterAssment[:, 0].A == cent)[0]]
                # assign centroid to mean
                centroids[cent, :] = mean(ptsInClust, axis=0)  
        return centroids, dp.SeriesSet(clusterAssment[:, 0].tolist())



