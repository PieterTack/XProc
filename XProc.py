# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:07:35 2020

@author: prrta
"""
from PyMca5.PyMca import FastXRFLinearFit
from PyMca5.PyMcaPhysics.xrf import ClassMcaTheory
from PyMca5.PyMcaPhysics.xrf import Elements
from PyMca5.PyMcaIO import ConfigDict
import numpy as np
from scipy.interpolate import griddata
import h5py
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib import container
import itertools

import multiprocessing
from functools import partial


class Cnc():
    def __init__(self):
        self.name = ''
        self.z = 0 # atomic number
        self.conc = 0 # [ppm or ug/g]
        self.err = 0 # [ppm or ug/g]
        self.density = 0 # [mg/cm^3]
        self.mass = 0 # [mg]
        self.thickness = 0 # [micron]
        
class Spc():
    def __init__(self, spcfile):
        # file content: spc_dtype = \
        #     [  # data offset (bytes)
        #         ('fVersion', '<f4'),  # 0
        #         ('aVersion', '<f4'),  # 4
        #         ('fileName', '8i1'),  # 8
        #         ('collectDateYear', '<i2'),  # 16
        #         ('collectDateDay', '<i1'),  # 17
        #         ('collectDateMon', '<i1'),
        #         ('collectTimeMin', '<i1'),
        #         ('collectTimeHour', '<i1'),
        #         ('collectTimeHund', '<i1'),
        #         ('collectTimeSec', '<i1'),
        #         ('fileSize', '<i4'),  # 24
        #         ('dataStart', '<i4'),  # 28
        #         ('numPts', '<i2'),  # 32
        #         ('intersectingDist', '<i2'),  # 34
        #         ('workingDist', '<i2'),  # 36
        #         ('scaleSetting', '<i2'),  # 38
    
        #         ('filler1', 'V24'),  # 40
    
        #         ('spectrumLabel', '256i1'),  # 64
        #         ('imageFilename', '8i1'),  # 320
        #         ('spotX', '<i2'),  # 328
        #         ('spotY', '<i2'),  # 330
        #         ('imageADC', '<i2'),  # 332
        #         ('discrValues', '<5i4'),  # 334
        #         ('discrEnabled', '<5i1'),  # 354
        #         ('pileupProcessed', '<i1'),  # 359
        #         ('fpgaVersion', '<i4'),  # 360
        #         ('pileupProcVersion', '<i4'),  # 364
        #         ('NB5000CFG', '<i4'),  # 368
    
        #         ('filler2', 'V12'),  # 380
    
        #         ('evPerChan', '<i4'),  # 384 **
        #         ('ADCTimeConstant', '<i2'),  # 388
        #         ('analysisType', '<i2'),  # 390     #4:live, 1:clock
        #         ('preset', '<f4'),  # 392           #set measurement time in s
        #         ('maxp', '<i4'),  # 396
        #         ('maxPeakCh', '<i4'),  # 400
        #         ('xRayTubeZ', '<i2'),  # 404
        #         ('filterZ', '<i2'),  # 406
        #         ('current', '<f4'),  # 408          #Tube current in uA?
        #         ('sampleCond', '<i2'),  # 412       #0:air, 1:vacuum
        #         ('sampleType', '<i2'),  # 414
        #         ('xrayCollimator', '<u2'),  # 416
        #         ('xrayCapilaryType', '<u2'),  # 418
        #         ('xrayCapilarySize', '<u2'),  # 420     #"Beam spot"
        #         ('xrayFilterThickness', '<u2'),  # 422
        #         ('spectrumSmoothed', '<u2'),  # 424
        #         ('detector_Size_SiLi', '<u2'),  # 426   #detector area in mm²
        #         ('spectrumReCalib', '<u2'),  # 428
        #         ('eagleSystem', '<u2'),  # 430
        #         ('sumPeakRemoved', '<u2'),  # 432
        #         ('edaxSoftwareType', '<u2'),  # 434
    
        #         ('filler3', 'V6'),  # 436
    
        #         ('escapePeakRemoved', '<u2'),  # 442
        #         ('analyzerType', '<u4'),  # 444
        #         ('startEnergy', '<f4'),  # 448 **
        #         ('endEnergy', '<f4'),  # 452
        #         ('liveTime', '<f4'),  # 456 **      #Live time
        #         ('tilt', '<f4'),  # 460 **
        #         ('takeoff', '<f4'),  # 464          #beam incidence angle
        #         ('beamCurFact', '<f4'),  # 468
        #         ('detReso', '<f4'),  # 472 **
        #         ('detectType', '<u4'),  # 476
        #         ('parThick', '<f4'),  # 480
        #         ('alThick', '<f4'),  # 484          #Al light shield thickness (in um?)
        #         ('beWinThick', '<f4'),  # 488       #Be window thickness (in um?)
        #         ('auThick', '<f4'),  # 492          #Au light shield thickness (in um?)
        #         ('siDead', '<f4'),  # 496
        #         ('siLive', '<f4'),  # 500
        #         ('xrayInc', '<f4'),  # 504
        #         ('azimuth', '<f4'),  # 508 **
        #         ('elevation', '<f4'),  # 512 **
        #         ('bCoeff', '<f4'),  # 516
        #         ('cCoeff', '<f4'),  # 520
        #         ('tailMax', '<f4'),  # 524
        #         ('tailHeight', '<f4'),  # 528
        #         ('kV', '<f4'),  # 532 **            #kV source setting
        #         ('apThick', '<f4'),  # 536
        #         ('xTilt', '<f4'),  # 540
        #         ('yTilt', '<f4'),  # 544
        #         ('yagStatus', '<u4'),  # 548
    
        #         ('filler4', 'V24'),  # 552
    
        #         ('rawDataType', '<u2'),  # 576
        #         ('totalBkgdCount', '<f4'),  # 578
        #         ('totalSpectralCount', '<u4'),  # 582
        #         ('avginputCount', '<f4'),  # 586
        #         ('stdDevInputCount', '<f4'),  # 590
        #         ('peakToBack', '<u2'),  # 594
        #         ('peakToBackValue', '<f4'),  # 596
    
        #         ('filler5', 'V38'),  # 600
    
        #         ('numElem', '<i2'),  # 638 **
        #         ('at', '<48u2'),  # 640 **
        #         ('line', '<48u2'),  # 736
        #         ('energy', '<48f4'),  # 832
        #         ('height', '<48u4'),  # 1024
        #         ('spkht', '<48i2'),  # 1216
    
        #         ('filler5_1', 'V30'),  # 1312
    
        #         ('numRois', '<i2'),  # 1342
        #         ('st', '<48i2'),  # 1344
        #         ('end', '<48i2'),  # 1440
        #         ('roiEnable', '<48i2'),  # 1536
        #         ('roiNames', '(24,8)i1'),  # 1632
    
        #         ('filler5_2', 'V1'),  # 1824
    
        #         ('userID', '80i1'),  # 1825
    
        #         ('filler5_3', 'V111'),  # 1905
    
        #         ('sRoi', '<48i2'),  # 2016
        #         ('scaNum', '<48i2'),  # 2112
    
        #         ('filler6', 'V12'),  # 2208
    
        #         ('backgrdWidth', '<i2'),  # 2220
        #         ('manBkgrdPerc', '<f4'),  # 2222
        #         ('numBkgrdPts', '<i2'),  # 2226
        #         ('backMethod', '<u4'),  # 2228
        #         ('backStEng', '<f4'),  # 2232
        #         ('backEndEng', '<f4'),  # 2236
        #         ('bg', '<64i2'),  # 2240
        #         ('bgType', '<u4'),  # 2368
        #         ('concenKev1', '<f4'),  # 2372
        #         ('concenKev2', '<f4'),  # 2376
        #         ('concenMethod', '<i2'),  # 2380
        #         ('jobFilename', '<32i1'),  # 2382
    
        #         ('filler7', 'V16'),  # 2414
    
        #         ('numLabels', '<i2'),  # 2430
        #         ('label', '<(10,32)i1'),  # 2432
        #         ('labelx', '<10i2'),  # 2752
        #         ('labely', '<10i4'),  # 2772
        #         ('zListFlag', '<i4'),  # 2812
        #         ('bgPercents', '<64f4'),  # 2816
        #         ('IswGBg', '<i2'),  # 3072
        #         ('BgPoints', '<5f4'),  # 3074
        #         ('IswGConc', '<i2'),  # 3094
        #         ('numConcen', '<i2'),  # 3096
        #         ('ZList', '<24i2'),  # 3098
        #         ('GivenConc', '<24f4'),  # 3146
    
        #         ('filler8', 'V598'),  # 3242
    
        #         ('spectrum', '<4096i4'),  # 3840 #actual spectrum
        #         ('longFileName', '<256i1'),  # 20224
        #         ('longImageFileName', '<256i1'),  # 20480
        #     ]
        
        # We'll only look for the parts we need
        self.rv = {}
        with open(spcfile, 'rb') as h:
            h.seek(408)
            self.rv['Current'] = np.fromfile(h, dtype = [ ('current', '<f4')], count=1)['current'][0]
            h.seek(456)
            self.rv['LiveTime'] = np.fromfile(h, dtype = [ ('liveTime', '<f4')], count=1)['liveTime'][0]
            h.seek(3840)
            self.rv['Data'] = np.fromfile(h, dtype = [ ('spectrum', '<4096i4')], count=1)['spectrum'][0]
            self.rv['ICR'] = np.sum(self.rv['Data'])
            self.rv['OCR'] = self.rv['ICR'] # ICR and OCR are identical in this case as data is already deadtime corrected (i.e. acquired for given livetime)
            
            # Notes: no info on motor positions is found in this file
            # We could also consider importing the Eagle-quantified data, for further usage...
            # To read in all data fields: h.seek(0); header = np.fromfile(h, dtype = spc_dtype, count=1)self.rv[]; for name in header.dtype.names: self.rv[name] = header[name][0] if len(header[name]) == 1 else header[name]
            


    
##############################################################################
def read_cnc(cncfile):
    """
    Read in the data of a concentration (.cnc) file.

    Parameters
    ----------
    cncfile : String
        .cnc file path.

    Returns
    -------
    rv : Cnc() Class
        Cnc class containing the data contained within the .cnc file.

    """
    rv = Cnc()
    line = ""
    with open(cncfile, "r") as f:
        f.readline() #Standard_Name
        rv.name = f.readline() # name of the standard
        f.readline() #Density(mg/cm^3)	Mass(mg)	Sample_thickness(micron)
        line = [float(i) for i in f.readline().split("\t") if i.strip()] #should contain 3 elements
        rv.density = line[0]
        rv.mass = line[1]
        rv.thickness = line[2]
        f.readline() #Number of elements
        size = int(f.readline())
        f.readline() #Z	Cert conc(ppm)	Standard_error(ppm)
        z = np.zeros(size)
        conc = np.zeros(size)
        err = np.zeros(size)
        for i in range(0,size):
            line = [float(i) for i in f.readline().split("\t") if i.strip()] #should contain 3 elements
            z[i] = int(line[0])
            conc[i] = line[1]
            err[i] = line[2]
    rv.z = z
    rv.conc = conc
    rv.err = err
    
    return rv

##############################################################################
def PCA(rawdata, nclusters=5, el_id=None):
    """
    returns: data transformed in 5 dims/columns + regenerated original data
    pass in: data as 2D NumPy array of dimension [M,N] with M the amount of observations and N the variables/elements
    

    Parameters
    ----------
    rawdata : array
        2D NumPy array of dimension [M,N] with M the amount of observations and N the variables/elements.
    nclusters : integer, optional
        The amount of PCA clusters to reduce the data to. The default is 5.
    el_id : list of integers, optional
        List of integers indexing the N variables/elements to be used for PCA analysis. The default is None, denoting the usage of all variables.

    Returns
    -------
    scores : float
        PCA scores (images).
    evals : float
        PCA eigenvalues (RVE, not yet normalised).
    evecs : float
        PCA eigenvectors (loading values).

    """

    if rawdata.ndim == 3:
        # assumes first dim is the elements
        data = rawdata.reshape(rawdata.shape[0], rawdata.shape[1]*rawdata.shape[2]).T #transform so elements becomes second dimension
    else:
        # we assume rawdata is properly oriented
        data = rawdata

    if el_id is not None:
        data = data[:, el_id]
    
    data[np.isnan(data)] = 0.
    # mean center the data
    data -= data.mean(axis=0)
    data = data/data.std(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = np.linalg.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or nclusters)
    evecs = evecs[:, :nclusters]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    if rawdata.ndim == 3:
        scores = np.moveaxis(np.dot(evecs.T, data.T).T.reshape(rawdata.shape[1], rawdata.shape[2], nclusters), -1, 0)
    else:
        scores = np.dot(evecs.T, data.T).T
        
    return scores, evals, evecs
##############################################################################
def h5_pca(h5file, h5dir, nclusters=5, el_id=None, kmeans=False):
    """
    Perform PCA analysis on a h5file dataset. 
    Before clustering, the routine performs a sqrt() normalisation on the data to reduce intensity differences between elements
      a selection of elements can be given in el_id as their integer values corresponding to their array position in the dataset (first element id = 0)
      kmeans can be set as an option, which will perform Kmeans clustering on the PCA score images and extract the respective sumspectra


    Parameters
    ----------
    h5file : string
        File directory path to the H5 file containing the data.
    h5dir : string
        Data directory within the H5 file containing the data to be analysed.
    nclusters : integer, optional
        The amount of PCA clusters to reduce the data to. The default is 5.
    el_id : List of integers, optional
        List of integers indexing the N variables/elements to be used for PCA analysis. The default is None, denoting the usage of all variables.
    kmeans : Boolean, optional
        If True, will perform Kmeans clustering on the PCA score images and extract the respective sumspectra. The default is False.

    Returns
    -------
    None.

    """
    # read in h5file data, with appropriate h5dir
    file = h5py.File(h5file, 'r+', locking=True)
    data = np.asarray(file[h5dir])
    if el_id is not None:
        names = [n.decode('utf8') for n in file['/'.join(h5dir.split("/")[0:-1])+'/names']]
    if 'channel00' in h5dir:
        if kmeans is not None:
            spectra = np.asarray(file['raw/channel00/spectra'])
        channel = 'channel00'
    elif 'channel01' in h5dir:
        if kmeans is not None:
            spectra = np.asarray(file['raw/channel01/spectra'])
        channel = 'channel01'
    elif 'channel02' in h5dir:
        if kmeans is not None:
            spectra = np.asarray(file['raw/channel02/spectra'])
        channel = 'channel02'
    
    # perform PCA clustering
    scores, evals, evecs = PCA(data, nclusters=nclusters, el_id=el_id)
    PCA_names = []
    for i in range(nclusters):
        PCA_names.append("PC"+str(i))
    
    # save the cluster image , as well as the elements that were clustered (el_id), loading plot data (eigenvectors) and eigenvalues (explained variance sum)
    try:
        del file['PCA/'+channel+'/el_id']
        del file['PCA/'+channel+'/nclusters']
        del file['PCA/'+channel+'/ims']
        del file['PCA/'+channel+'/names']
        del file['PCA/'+channel+'/RVE']
        del file['PCA/'+channel+'/loadings']
    except Exception:
        pass
    if el_id is not None:
        file.create_dataset('PCA/'+channel+'/el_id', data=[n.encode('utf8') for n in names[el_id]])
    else:
        file.create_dataset('PCA/'+channel+'/el_id', data='None')        
    file.create_dataset('PCA/'+channel+'/nclusters', data=nclusters)
    file.create_dataset('PCA/'+channel+'/ims', data=scores, compression='gzip', compression_opts=4)
    file.create_dataset('PCA/'+channel+'/names', data=[n.encode('utf8') for n in PCA_names])
    file.create_dataset('PCA/'+channel+'/RVE', data=evals[0:nclusters]/np.sum(evals))
    file.create_dataset('PCA/'+channel+'/loadings', data=evecs, compression='gzip', compression_opts=4)
    dset = file['PCA']
    dset.attrs["LastUpdated"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
    # if kmeans option selected, follow up with Kmeans clustering on the PCA clusters
    if kmeans is True:
        clusters, centroids = Kmeans(scores, nclusters=nclusters, el_id=None)
        
        # calculate cluster sumspectra
        #   first check if raw spectra shape is identical to clusters shape, as otherwise it's impossible to relate appropriate spectrum to pixel
        if spectra.shape[0] == clusters.size:
            sumspec = []
            for i in range(nclusters):
                sumspec.append(np.sum(spectra[np.where(clusters.ravel() == i),:], axis=0))
        
        # save the cluster image and sumspectra, as well as the elements that were clustered (el_id)
        try:
            del file['kmeans/'+channel+'/nclusters']
            del file['kmeans/'+channel+'/data_dir_clustered']
            del file['kmeans/'+channel+'/ims']
            del file['kmeans/'+channel+'/el_id']
            for i in range(nclusters):
                del file['kmeans/'+channel+'/sumspec_'+str(i)]
        except Exception:
            pass
        file.create_dataset('kmeans/'+channel+'/nclusters', data=nclusters)
        file.create_dataset('kmeans/'+channel+'/data_dir_clustered', data=('PCA/'+channel+'/ims').encode('utf8'))
        file.create_dataset('kmeans/'+channel+'/ims', data=clusters, compression='gzip', compression_opts=4)
        file.create_dataset('kmeans/'+channel+'/el_id', data=[n.encode('utf8') for n in PCA_names])     
        dset = file['kmeans']
        dset.attrs["LastUpdated"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        if spectra.shape[0] == clusters.size:
            for i in range(nclusters):
                file.create_dataset('kmeans/'+channel+'/sumspec_'+str(i), data=sumspec[i,:], compression='gzip', compression_opts=4)    
    file.close()    

##############################################################################
def Kmeans(rawdata, nclusters=5, el_id=None, whiten=True):
    """
    Perform Kmeans clustering on a dataset.

    Parameters
    ----------
    rawdata : Float array
        2D or 3D array which will be clustered. Variables/elements are the first dimension.
    nclusters : integer, optional
        The amount of Kmeans clusters to reduce the data to. The default is 5.
    el_id : List of integers, optional
        List of integers indexing the N variables/elements to be used for Kmeans clustering. The default is None, denoting the usage of all variables.
    whiten : Boolean, optional
        If True, data is whitened using the whiten() scipy function. The default is True.

    Returns
    -------
    clusters : integer array
        1D or 2D array matching the rawdata shape, for which the integer value denotes the cluster to which each pixel was attributed.
    centroids : float array
        Kmeans cluster centroid values.

    """
    from scipy.cluster.vq import kmeans2, whiten

    if rawdata.ndim == 3:
        # assumes first dim is the elements
        data = rawdata.reshape(rawdata.shape[0], rawdata.shape[1]*rawdata.shape[2]).T #transform so elements becomes second dimension
    else:
        # we assume rawdata is properly oriented
        data = rawdata

    if el_id is not None:
        data = data[:, el_id]

    # first whiten data (normalises it)
    data[np.isnan(data)] = 0.
    if whiten is True:
        data = whiten(data) #data should not contain any NaN or infinite values

    # then do kmeans
    centroids, clusters = kmeans2(data, nclusters, iter=100, minit='points')
    
    if rawdata.ndim == 3:
        clusters = clusters.reshape(rawdata.shape[1], rawdata.shape[2])
    
    return clusters, centroids
##############################################################################
def h5_kmeans(h5file, h5dir, nclusters=5, el_id=None, nosumspec=False):
    """
    Perform Kmeans clustering on a h5file dataset.
      Before clustering data is whitened using the Scipy whiten() function.

    Parameters
    ----------
    h5file : string
        File directory path to the H5 file containing the data.
    h5dir : string
        Data directory within the H5 file containing the data to be analysed.
    nclusters : integer, optional
        The amount of Kmeans clusters to reduce the data to. The default is 5.
    el_id : List of integers, optional
        List of integers indexing the N variables/elements to be used for Kmeans clustering. The default is None, denoting the usage of all variables.
    nosumspec : Boolean, optional
        If True, no sumspectra are calculated and stored in the H5 file for each Kmeans cluster. The default is False.

    Returns
    -------
    None.

    """
    # read in h5file data, with appropriate h5dir
    file = h5py.File(h5file, 'r+', locking=True)
    data = np.asarray(file[h5dir])
    if el_id is not None:
        names = [n.decode('utf8') for n in file['/'.join(h5dir.split("/")[0:-1])+'/names']]
    if 'channel00' in h5dir:
        spectra = np.asarray(file['raw/channel00/spectra'])
        channel = 'channel00'
    elif 'channel01' in h5dir:
        spectra = np.asarray(file['raw/channel01/spectra'])
        channel = 'channel01'
    elif 'channel02' in h5dir:
        spectra = np.asarray(file['raw/channel02/spectra'])
        channel = 'channel02'
    spectra = spectra.reshape((spectra.shape[0]*spectra.shape[1], spectra.shape[2]))
    
    # perform Kmeans clustering
    clusters, centroids = Kmeans(data, nclusters=nclusters, el_id=el_id)
    
    # calculate cluster sumspectra
    #   first check if raw spectra shape is identical to clusters shape, as otherwise it's impossible to relate appropriate spectrum to pixel
    # Warning: in case of timetriggered scans cluster.ravel indices may not match the appropriate cluster point!
    if nosumspec is False:
        if spectra.shape[0] == clusters.size:
            sumspec = []
            for i in range(nclusters):
                sumspec.append(np.sum(np.squeeze(spectra[np.where(clusters.ravel() == i),:]), axis=0))
    
    # save the cluster image and sumspectra, as well as the elements that were clustered (el_id)
    try:
        del file['kmeans/'+channel]
    except Exception:
        pass
    file.create_dataset('kmeans/'+channel+'/nclusters', data=nclusters)
    file.create_dataset('kmeans/'+channel+'/data_dir_clustered', data=h5dir.encode('utf8'))
    file.create_dataset('kmeans/'+channel+'/ims', data=clusters, compression='gzip', compression_opts=4)
    dset = file['kmeans']
    dset.attrs["LastUpdated"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if el_id is not None:
        file.create_dataset('kmeans/'+channel+'/el_id', data=[n.encode('utf8') for n in np.asarray(names)[el_id]])
    else:
        file.create_dataset('kmeans/'+channel+'/el_id', data='None')        
    if nosumspec is False:
        if spectra.shape[0] == clusters.size:
            for i in range(nclusters):
                dset = file.create_dataset('kmeans/'+channel+'/sumspec_'+str(i), data=np.asarray(sumspec)[i,:], compression='gzip', compression_opts=4)    
                dset.attrs["NPixels"] = np.asarray(np.where(clusters.ravel() == i)).size
    file.close()
    
##############################################################################
def div_by_cnc(h5file, cncfile, channel=None):
    """
    Divide quantified images by the corresponding concentration value of the same element in the cncfile to obtain relative difference images
      If an element in the h5file is not present in the cncfile it is omitted from further processing.
    

    Parameters
    ----------
    h5file : string
        File directory path to the H5 file containing the data.
    cncfile : string
        File directory path to the CNC file containing the reference material composition information.
    channel : string, optional
        H5 file (detector) channel in the quant/ directory for which data should be processed. The default is None, which results in the calculation being performed on the lowest indexed channel (channel00).

    Returns
    -------
    None.

    """
    # read in h5file quant data
    file = h5py.File(h5file, 'r+', locking=True)
    if channel is None:
        channel = list(file['quant'].keys())[0]
    h5_ims = np.asarray(file['quant/'+channel+'/ims'])
    h5_imserr = np.asarray(file['quant/'+channel+'/ims_stddev'])
    h5_names = np.asarray([n.decode('utf8') for n in file['quant/'+channel+'/names']])
    h5_sum = np.asarray(file['quant/'+channel+'/sum/int'])
    h5_sumerr = np.asarray(file['quant/'+channel+'/sum/int_stddev'])
    h5_z = [Elements.getz(n.split(" ")[0]) for n in h5_names]

    # read in cnc file
    cnc = read_cnc(cncfile)

    #convert errors to relative errors
    cnc.err /= cnc.conc
    h5_sumerr /= h5_sum
    for z in range(0, h5_ims.shape[0]):
        h5_imserr[z, :, :] /= h5_ims[z, :, :]

    # loop over h5_z and count how many times there's a common z in h5_z and cnc.z
    cnt = 0
    for z in range(0,len(h5_z)):
        if h5_z[z] in cnc.z:
            cnt+=1

    # make array to store rel_diff data and calculate them
    rel_diff_ims = np.zeros((cnt, h5_ims.shape[1], h5_ims.shape[2]))
    rel_diff_sum = np.zeros((cnt))
    rel_diff_imserr = np.zeros((cnt, h5_ims.shape[1], h5_ims.shape[2]))
    rel_diff_sumerr = np.zeros((cnt))
    rel_names = []

    cnt = 0
    for z in range(0, len(h5_z)):
        if h5_z[z] in cnc.z:
            rel_diff_ims[cnt, :, :] = h5_ims[z,:,:] / cnc.conc[list(cnc.z).index(h5_z[z])]
            rel_diff_sum[cnt] = h5_sum[z] / cnc.conc[list(cnc.z).index(h5_z[z])]
            rel_diff_imserr[cnt, :, :] = np.sqrt(h5_imserr[z,:,:]**2 + cnc.err[list(cnc.z).index(h5_z[z])]**2)
            rel_diff_sumerr[cnt] = np.sqrt(h5_sumerr[z]**2 + cnc.err[list(cnc.z).index(h5_z[z])]**2)
            rel_names.append(h5_names[z])
            cnt+=1

    # convert relative errors to absolute
    for z in range(0, rel_diff_ims.shape[0]):
        rel_diff_imserr *= rel_diff_ims
    rel_diff_sumerr *= rel_diff_sum

    # save rel_diff data
    try:
        del file['rel_dif/'+channel+'/names']
        del file['rel_dif/'+channel+'/ims']
        del file['rel_dif/'+channel+'/ims_stddev']
        del file['rel_dif/'+channel+'/sum/int']
        del file['rel_dif/'+channel+'/sum/int_stddev']
        del file['rel_dif/'+channel+'/cnc']
    except Exception:
        pass
    file.create_dataset('rel_dif/'+channel+'/names', data=[n.encode('utf8') for n in rel_names])
    file.create_dataset('rel_dif/'+channel+'/ims', data=rel_diff_ims, compression='gzip', compression_opts=4)
    file.create_dataset('rel_dif/'+channel+'/sum/int', data=rel_diff_sum, compression='gzip', compression_opts=4)
    file.create_dataset('rel_dif/'+channel+'/ims_stddev', data=rel_diff_imserr, compression='gzip', compression_opts=4)
    file.create_dataset('rel_dif/'+channel+'/sum/int_stddev', data=rel_diff_sumerr, compression='gzip', compression_opts=4)
    file.create_dataset('rel_dif/'+channel+'/cnc', data=cncfile.split("/")[-1].encode('utf8'))
    dset = file['rel_dif']
    dset.attrs["LastUpdated"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    file.close()

##############################################################################
def quant_with_ref(h5file, reffiles, channel='channel00', norm=None, absorb=None, snake=False, mask=None, density=None, thickness=None, composition=None):
    """
    Quantify XRF data, making use of elemental yields as determined from reference files
      h5file and reffiles should both contain a norm/ data directory as determined by norm_xrf_batch()
      The listed ref files should have had their detection limits calculated by calc_detlim() before
          as this function also calculates element yields.
      If an element in the h5file is not present in the listed refs, its yield is estimated through linear interpolation of the closest neighbouring atoms with the same linetype.
          if Z is at the start of end of the reference elements, the yield will be extrapolated from the first or last 2 elements in the reference
          if only 1 element in the reference has the same linetype as the quantifiable element, but does not have the same Z, the same yield is used nevertheless as inter/extrapolation is impossible
      A mask can be provided. This can either be a reference to a kmeans cluster ID supplied as a string or list of strings, e.g. 'kmeans/CLR2' or ['CLR2','CLR4'],
          or a string data path within the h5file containing a 2D array of size equal to the h5file image size, where 0 values represent pixels to omit
          from the quantification and 1 values are pixels to be included. Alternatively, a 2D array can be directly supplied as argument.
     In order to calculate the concentration as parts per million rather than the default areal concentration, one can supply a density, thickness and composition (as a .cnc file).
         The composition file is used to calculate the fluorescence escape depth for each element to quantify, which is then compared to the (grain) thickness for further quantification.
         Note that if the density keyword is set, so too should thickness. If composition is None, the provided thickness is used for all elements, irrespective of escape depth.

    Parameters
    ----------
    h5file : string
        File directory path to the H5 file containing the data to be quantified.
    reffiles : (list of) string(s).
        File directory path(s) to the H5 file(s) containing the data corresponding to measurement(s) on referene material(s).
    channel : string, optional
        H5 file (detector) channel in the quant/ directory for which data should be processed. The default is 'channel00'.
    norm : String, optional
        String of element name as it is included in the /names data directory within the H5 file. 
        If keyword norm is provided, the elemental yield is corrected for the intensity of this signal
        the signal has to be present in both reference and XRF data fit. The default is None.
    absorb : tuple (['element'], 'cnc file'), optional
        If keyword absorb is provided, the fluorescence signal in the XRF data is corrected for absorption through sample matrix.
        The element will be used to find Ka and Kb line intensities and correct for their respective ratio
        using concentration values from the provided cnc files. The default is None, performing no absorption correction.
    snake : Boolean, optional
        If the scan was performed following a snake-like pattern, set to True to allow for appropriate image reconstruction. The default is False.
    mask : String, list of strings, 2D binary integer array or None, optional
        A data mask can be provided. This can either be a reference to a kmeans cluster ID supplied as a string or list of strings, e.g. 'kmeans/CLR2' or ['CLR2','CLR4'],
            or a string data path within the H5 file containing a 2D array of size equal to the H5 file image size, where 0 values represent pixels to omit
            from the quantification and 1 values are pixels to be included. Alternatively, a 2D array can be directly supplied as argument. The default is None.
    density : float or None, optional
        If keyword density and thickness are not None, the calculated areal concentration is divided by a density [g/cm³]*thickness [cm] value. The default is None.
    thickness : float or None, optional
        If keyword density and thickness are not None, the calculated areal concentration is divided by a density [g/cm³]*thickness [cm] value. The default is None.
    composition: string or None, optional
        File directory path to the CNC file containing the (approximate) material composition information.
        If composition is not None, and density and thickness are set, the CNC file is used to calculate the escape depth from this material for each element to quantify. 
        The lower value between escape depth (99% absorption) and user-supplied thickness is used for further processing. Note that for all linetypes, the alfa line is used 
        (i.e. if it is a K-line, Ka fluorescence energy will be considered).
    
    Yields
    ------
    bool
        Returns False on error.

    """
    import Xims

    if (density is not None and thickness is None) or (density is None and thickness is not None):
            print("ERROR: quant_with_ref: density and thickness should both be not-None values if set.")
            return False

    # first let's go over the reffiles and calculate element yields
    #   distinguish between K and L lines while doing this
    reffiles = np.asarray(reffiles)
    if reffiles.size == 1:
        reff = h5py.File(str(reffiles), 'r', locking=True)
        ref_yld = [yld for yld in reff['elyield/'+[keys for keys in reff['elyield'].keys()][0]+'/'+channel+'/yield']] # elemental yields in (ug/cm²)/(ct/s)
        ref_yld_err = [yld for yld in reff['elyield/'+[keys for keys in reff['elyield'].keys()][0]+'/'+channel+'/stddev']] # elemental yield errors in (ug/cm²)/(ct/s)
        ref_names = [n.decode('utf8') for n in reff['elyield/'+[keys for keys in reff['elyield'].keys()][0]+'/'+channel+'/names']]
        ref_z = [Elements.getz(n.split(" ")[0]) for n in ref_names]
        ref_yld_err = np.asarray(ref_yld_err) / np.asarray(ref_yld) #convert to relative error
        if norm is not None:
            names = [n.decode('utf8') for n in reff['norm/'+channel+'/names']]
            if norm in names:
                sum_fit = np.asarray(reff['norm/'+channel+'/sum/int'])
                sum_fit[np.where(sum_fit < 0)] = 0
                ref_yld_err = np.sqrt(ref_yld_err*ref_yld_err + 1./sum_fit[names.index(norm)])
                ref_yld = [yld*(sum_fit[names.index(norm)]) for yld in ref_yld]
            else:
                print("ERROR: quant_with_ref: norm signal not present for reference material in "+str(reffiles))
                return False
        reff.close()        
        ref_yld = np.asarray(ref_yld)
        ref_yld_err = np.asarray(ref_yld_err)
        ref_names = np.asarray(ref_names)
        ref_z = np.asarray(ref_z)
    else:
        ref_yld = []
        ref_yld_err = []
        ref_names = []
        ref_z = []
        for i in range(0, reffiles.size):
            reff = h5py.File(str(reffiles[i]), 'r', locking=True)
            ref_yld_tmp = [yld for yld in reff['elyield/'+[keys for keys in reff['elyield'].keys()][0]+'/'+channel+'/yield']] # elemental yields in (ug/cm²)/(ct/s)
            ref_yld_err_tmp = [yld for yld in reff['elyield/'+[keys for keys in reff['elyield'].keys()][0]+'/'+channel+'/stddev']] # elemental yield errors in (ug/cm²)/(ct/s)
            ref_names_tmp = [n.decode('utf8') for n in reff['elyield/'+[keys for keys in reff['elyield'].keys()][0]+'/'+channel+'/names']]
            ref_yld_err_tmp = np.asarray(ref_yld_err_tmp) / np.asarray(ref_yld_tmp) #convert to relative error
            if norm is not None:
                names = [n.decode('utf8') for n in reff['norm/'+channel+'/names']]
                if norm in names:
                    ref_sum_fit = np.asarray(reff['norm/'+channel+'/sum/int'])
                    ref_sum_bkg = np.asarray(reff['norm/'+channel+'/sum/bkg'])
                    ref_rawI0 = np.sum(np.asarray(reff['raw/I0']))
                    ref_sum_fit[np.where(ref_sum_fit < 0)] = 0
                    ref_sum_bkg[np.where(ref_sum_bkg < 0)] = 0
                    ref_normto = np.asarray(reff['norm/I0'])
                    ref_sum_fit = ref_sum_fit / ref_normto
                    ref_sum_bkg = ref_sum_bkg / ref_normto
                    ref_yld_err_tmp = np.sqrt(ref_yld_err_tmp*ref_yld_err_tmp + np.sqrt((ref_sum_fit[names.index(norm)]+2.*ref_sum_bkg[names.index(norm)])*ref_rawI0)/(ref_sum_fit[names.index(norm)]*ref_rawI0))
                    ref_yld_tmp = [yld*(ref_sum_fit[names.index(norm)]) for yld in ref_yld_tmp]
                else:
                    print("ERROR: quant_with_ref: norm signal not present for reference material in "+reffiles[i])
                    return False
            for j in range(0, np.asarray(ref_yld_tmp).size):
                ref_yld.append(ref_yld_tmp[j])
                ref_yld_err.append(ref_yld_err_tmp[j])
                ref_names.append(ref_names_tmp[j])
                ref_z.append(Elements.getz(ref_names_tmp[j].split(" ")[0]))
            reff.close()
        # find unique line names, and determine average yield for each of them
        ref_yld = np.asarray(ref_yld)
        ref_yld_err = np.asarray(ref_yld_err)
        ref_names = np.asarray(ref_names)
        ref_z = np.asarray(ref_z)
        unique_names, unique_id = np.unique(ref_names, return_index=True)
        unique_z = ref_z[unique_id]
        unique_yld = np.zeros(unique_z.size)
        unique_yld_err = np.zeros(unique_z.size)
        for j in range(0, unique_z.size):
            name_id = [i for i, x in enumerate(ref_names) if x == unique_names[j]]
            unique_yld[j] = np.average(ref_yld[name_id])
            unique_yld_err[j] = np.sqrt(np.sum(np.asarray(ref_yld_err[name_id])*np.asarray(ref_yld_err[name_id])))
        # order the yields by atomic number
        ref_names = unique_names[np.argsort(unique_z)]
        ref_yld = unique_yld[np.argsort(unique_z)]
        ref_yld_err = unique_yld_err[np.argsort(unique_z)]
        ref_z = unique_z[np.argsort(unique_z)]
    
    # read in h5file norm data
    #   normalise intensities to 1s acquisition time as this is the time for which we have el yields
    file = h5py.File(h5file, 'r', locking=True)
    h5_ims = np.asarray(file['norm/'+channel+'/ims'])
    h5_ims_err = np.asarray(file['norm/'+channel+'/ims_stddev'])/h5_ims[:]
    h5_names = np.asarray([n.decode('utf8') for n in file['norm/'+channel+'/names']])
    h5_sum = np.asarray(file['norm/'+channel+'/sum/int'])
    h5_sum_err = np.asarray(file['norm/'+channel+'/sum/int_stddev'])/h5_sum[:]
    h5_normto = np.asarray(file['norm/I0'])
    tmnorm = str(file['norm'].attrs["TmNorm"])
    if tmnorm == "True":
        tmnorm = True
    else:
        tmnorm = False
    if absorb is not None:
        h5_spectra = np.asarray(file['raw/'+channel+'/spectra'])
        try:
            h5_cfg = file['fit/'+channel+'/cfg'][()].decode('utf8')
        except AttributeError:
            h5_cfg = file['fit/'+channel+'/cfg'][()]
        if len(h5_spectra.shape) == 2:
            h5_spectra = h5_spectra.reshape((h5_spectra.shape[0], 1, h5_spectra.shape[1]))
        elif len(h5_spectra.shape) == 1:
            h5_spectra = h5_spectra.reshape((1, 1, h5_spectra.shape[0]))
    if snake is True:
        mot1 = np.asarray(file['mot1'])
        mot2 = np.asarray(file['mot2'])
    file.close()

    # add option to provide a mask from which new sumspec etc is calculated. 
    if mask is not None:
        # check whether mask is a string to a h5file path, or an array
        if type(mask) == type(str()):
            if 'clr' in mask.lower():
                # a cluster ID is provided as mask, so change mask to the appropriate path
                clrid = mask.split('/')[-1][3:]
                with h5py.File(h5file, 'r', locking=True) as file:
                    if ('kmeans/'+channel+'/ims' in file) is True:
                        mask = np.asarray(file['kmeans/'+channel+'/ims'])
                        mask = np.where(mask==float(clrid), 1, 0)
                    else:
                        print("Warning: No kmeans/"+channel+"/ims path in "+h5file)
                        print("    No mask was applied to further processing")     
                        mask = False
            else:
                # 'clr' or 'CLR' was not in the string, so likely another h5file path was provided
                with h5py.File(h5file, 'r', locking=True) as file:
                    if (mask in file) is True:
                        mask = np.asarray(file[mask])
                    else:
                        print("Warning: provided mask "+mask+" is not a path in "+h5file)
                        print("    No mask was applied to further processing")      
                        mask = False
        elif type(mask) == type(list()) and type(mask[0]) == type(str()):
            # in this case kmeans cluster paths should have been supplied.
            with h5py.File(h5file, 'r', locking=True) as file:
                if ('kmeans/'+channel+'/ims' in file) is True:
                    k_ims = np.asarray(file['kmeans/'+channel+'/ims'])
                    temp = np.zeros(k_ims.shape)
                    for clr in mask:
                        clrid = clr.split('/')[-1][3:]
                        temp += np.where(k_ims == float(clrid), 1, 0)
                    mask = temp 
                else:
                    print("Warning: No kmeans/"+channel+"/ims path in "+h5file)
                    print("    No mask was applied to further processing")  
                    mask = False
        # mask is now likely an array. Check whether the dimensions match the h5_ims shape and whether values are binary
        if mask is not False:
            mask = np.asarray(mask)            
            if len(mask.shape) != 2 or mask.shape[0] != h5_ims.shape[1] or mask.shape[1] != h5_ims.shape[2]:
                print('Warning: mask shape does not match h5file ims shape: ', mask.shape)
                print("    No mask was applied to further processing")
            elif ((mask==0) | (mask==1)).all() is False:
                print("Warning: user supplied mask does not contain binary values (0 or 1).")
                print("    No mask was applied to further processing")
            else:
                # Mask should be appropriate for further processing
                with h5py.File(h5file, 'r', locking=True) as file:
                    I0 = np.asarray(file["raw/I0"])*mask
                    tm = np.asarray(file["raw/acquisition_time"])*mask
                for i in range(0, h5_ims.shape[0]):
                    h5_ims[i,:,:] *= mask
                    h5_ims_err[i,:,:] *= mask
                    if tmnorm is True:
                        h5_sum[i] = np.sum(h5_ims[i,:,:])/(np.sum(I0*tm))*h5_normto
                    else:
                        h5_sum[i] = np.sum(h5_ims[i,:,:])/(np.sum(I0))*h5_normto
                    h5_sum_err[i] = np.sqrt(np.sum(h5_ims_err[i,:,:]**2))
        
    h5_ims = (h5_ims / h5_normto)  #These are intensities for 1 I0 count.
    h5_sum = (h5_sum / h5_normto)
    # remove Compt and Rayl signal from h5, as these cannot be quantified
    names = h5_names
    ims = h5_ims
    ims_err = h5_ims_err
    sumint = h5_sum
    sumint_err = h5_sum_err
    if 'Compt' in list(names):
        ims = ims[np.arange(len(names))!=list(names).index('Compt'),:,:]
        sumint = sumint[np.arange(len(names))!=list(names).index('Compt')]
        ims_err = ims_err[np.arange(len(names))!=list(names).index('Compt'),:,:]
        sumint_err = sumint_err[np.arange(len(names))!=list(names).index('Compt')]
        names = names[np.arange(len(names))!=list(names).index('Compt')]
    if 'Rayl' in list(names):
        ims = ims[np.arange(len(names))!=list(names).index('Rayl'),:,:]
        sumint = sumint[np.arange(len(names))!=list(names).index('Rayl')]
        ims_err = ims_err[np.arange(len(names))!=list(names).index('Rayl'),:,:]
        sumint_err = sumint_err[np.arange(len(names))!=list(names).index('Rayl')]
        names = names[np.arange(len(names))!=list(names).index('Rayl')]

    # Normalise for specified roi if required
    #   Return Warning/Error messages if roi not present in h5file
    if norm is not None:
        if norm in h5_names:
            # print(np.nonzero(h5_ims[list(h5_names).index(norm),:,:]==0)[0])
            for i in range(0, ims.shape[0]):
                ims[i,:,:] = ims[i,:,:] / h5_ims[list(h5_names).index(norm),:,:] #Note: we can get some division by zero error here...
                sumint[i] = sumint[i] / h5_sum[list(h5_names).index(norm)]
                ims_err[i,:,:] = np.sqrt(ims_err[i,:,:]**2 + h5_ims_err[list(h5_names).index(norm),:,:]**2)
                sumint_err[i] = np.sqrt(sumint_err[i]**2 + h5_sum_err[list(h5_names).index(norm)]**2)
        else:
            print("ERROR: quant_with_ref: norm signal not present in h5file "+h5file)
            return False

    # perform self-absorption correction based on Ka-Kb line ratio
    if absorb is not None:
        cnc = read_cnc(absorb[1])
        config = ConfigDict.ConfigDict()
        try:
            config.read(h5_cfg)
        except Exception:
            config.read('/'.join(h5file.split('/')[0:-1])+'/'+h5_cfg.split('/')[-1])
        cfg = [config['detector']['zero'], config['detector']['gain']]
        absorb_el = absorb[0]
        try:
            import xraylib
            # calculate absorption coefficient for each element/energy in names
            mu = np.zeros(names.size)
            for n in range(0, names.size):
                el = xraylib.SymbolToAtomicNumber(names[n].split(' ')[0])
                line = names[n].split(' ')[1]
                if line[0] == 'K':
                    line = xraylib.KL3_LINE #Ka1
                elif line[0] == 'L':
                    line = xraylib.L3M5_LINE #La1
                elif line[0] == 'M':
                    line = xraylib.M5N7_LINE #Ma1
                for i in range(0, len(cnc.z)):
                    mu[n] += xraylib.CS_Total(int(cnc.z[i]), xraylib.LineEnergy(el, line)) * cnc.conc[i]/1E6
            mu_ka1 = np.zeros(len(absorb_el))
            mu_kb1 = np.zeros(len(absorb_el))
            rate_ka1 = np.zeros(len(absorb_el))
            rate_kb1 = np.zeros(len(absorb_el))
            for j in range(len(absorb_el)):
                for i in range(0, len(cnc.z)):
                    mu_ka1[j] += xraylib.CS_Total(int(cnc.z[i]), xraylib.LineEnergy(xraylib.SymbolToAtomicNumber(absorb_el[j]),xraylib.KL3_LINE)) * cnc.conc[i]/1E6
                    mu_kb1[j] += xraylib.CS_Total(int(cnc.z[i]), xraylib.LineEnergy(xraylib.SymbolToAtomicNumber(absorb_el[j]),xraylib.KM3_LINE)) * cnc.conc[i]/1E6
                # determine the theoretical Ka - Kb ratio of the chosen element (absorb[0])
                rate_ka1[j] = xraylib.RadRate(xraylib.SymbolToAtomicNumber(absorb_el[j]), xraylib.KL3_LINE)
                rate_kb1[j] = xraylib.RadRate(xraylib.SymbolToAtomicNumber(absorb_el[j]), xraylib.KM3_LINE)
        except ImportError: # no xraylib, so use PyMca instead
            # calculate absorption coefficient for each element/energy in names
            mu = np.zeros(names.size)
            for n in range(0, names.size):
                el = names[n].split(' ')[0]
                line = names[n].split(' ')[1]
                if line[0] == 'K':
                    line = 'KL3' #Ka1
                elif line[0] == 'L':
                    line = 'L3M5' #La1
                elif line[0] == 'M':
                    line = 'M5N7' #Ma1
                for i in range(0, len(cnc.z)):
                    mu[n] += Elements.getmassattcoef(Elements.getsymbol(cnc.z[i]), Elements.getxrayenergy(el, line))['total'][0] * cnc.conc[i]/1E6
            mu_ka1 = np.zeros(len(absorb_el))
            mu_kb1 = np.zeros(len(absorb_el))
            rate_ka1 = np.zeros(len(absorb_el))
            rate_kb1 = np.zeros(len(absorb_el))
            for j in range(len(absorb_el)):
                for i in range(0, len(cnc.z)):
                    mu_ka1[j] += Elements.getmassattcoef(Elements.getsymbol(cnc.z[i]), Elements.getxrayenergy(absorb_el[j],'KL3'))['total'][0] * cnc.conc[i]/1E6
                    mu_kb1[j] += Elements.getmassattcoef(Elements.getsymbol(cnc.z[i]), Elements.getxrayenergy(absorb_el[j],'KM3'))['total'][0] * cnc.conc[i]/1E6
                # determine the theoretical Ka - Kb ratio of the chosen element (absorb[0])
                rate_ka1[j] = Elements._getUnfilteredElementDict(absorb_el[j], None)['KL3']['rate']
                rate_kb1[j] = Elements._getUnfilteredElementDict(absorb_el[j], None)['KM3']['rate']
        rhot = np.zeros((len(absorb_el), ims.shape[1], ims.shape[2]))
        for j in range(len(absorb_el)):
            # calculate Ka-Kb ratio for each experimental spectrum
                # Ka1 and Kb1 channel number
            idx_ka1 = max(np.where(np.arange(h5_spectra.shape[-1])*cfg[1]+cfg[0] <= Elements.getxrayenergy(absorb_el[j],'KL3'))[-1])
            idx_kb1 = max(np.where(np.arange(h5_spectra.shape[-1])*cfg[1]+cfg[0] <= Elements.getxrayenergy(absorb_el[j],'KM3'))[-1])
            # remove 0 and negative value to avoid division errors. On those points set ka1/kb1 ratio == rate_ka1/rate_kb1
            int_ka1 = np.sum(h5_spectra[:,:,int(np.round(idx_ka1-3)):int(np.round(idx_ka1+3))], axis=2)
            int_ka1[np.where(int_ka1 < 1.)] = 1.
            int_kb1 = np.sum(h5_spectra[:,:,int(np.round(idx_kb1-3)):int(np.round(idx_kb1+3))], axis=2)
            int_kb1[np.where(int_kb1 <= 1)] = int_ka1[np.where(int_kb1 <= 1)]*(rate_kb1[j]/rate_ka1[j])
            ratio_ka1_kb1 = int_ka1 / int_kb1
            # also do not correct any point where ratio_ka1_kb1 > rate_ka1/rate_kb1
            #   these points would suggest Ka was less absorbed than Kb
            ratio_ka1_kb1[np.where(ratio_ka1_kb1 > rate_ka1[j]/rate_kb1[j])] = rate_ka1[j]/rate_kb1[j]
            ratio_ka1_kb1[np.where(ratio_ka1_kb1 <= 0.55*rate_ka1[j]/rate_kb1[j])] = rate_ka1[j]/rate_kb1[j] #Note: this value may be inappropriate...
            ratio_ka1_kb1[np.isnan(ratio_ka1_kb1)] = rate_ka1[j]/rate_kb1[j]
            # calculate corresponding layer thickness per point through matrix defined by cncfiles
            rhot[j,:,:] = (np.log(ratio_ka1_kb1[:,:]) - np.log(rate_ka1[j]/rate_kb1[j])) / (mu_kb1[j] - mu_ka1[j]) # rho*T for each pixel based on Ka1 and Kb1 emission ratio
            print('Average Rho*t: ',absorb_el[j], np.average(rhot[j,:,:]))
        rhot[np.where(rhot < 0.)] = 0. #negative rhot values do not make sense
        rhot[np.isnan(rhot)] = 0.
        rhot = np.amax(rhot, axis=0)
        
        # if this is snakescan, interpolate ims array for motor positions so images look nice
        #   this assumes that mot1 was the continuously moving motor
        if snake is True:
            print("Interpolating rho*T for motor positions...", end=" ")
            pos_low = min(mot1[:,0])
            pos_high = max(mot1[:,0])
            for i in range(0, mot1[:,0].size): #correct for half a pixel shift
                if mot1[i,0] <= np.average((pos_high,pos_low)):
                    mot1[i,:] += abs(mot1[i,1]-mot1[i,0])/2.
                else:
                    mot1[i,:] -= abs(mot1[i,1]-mot1[i,0])/2.
            mot1_pos = np.average(mot1, axis=0) #mot1[0,:]
            mot2_pos = np.average(mot2, axis=1) #mot2[:,0]
            mot1_tmp, mot2_tmp = np.mgrid[mot1_pos[0]:mot1_pos[-1]:complex(mot1_pos.size),
                    mot2_pos[0]:mot2_pos[-1]:complex(mot2_pos.size)]
            x = mot1.ravel()
            y = mot2.ravel()
            values = rhot.ravel()
            rhot = griddata((x, y), values, (mot1_tmp, mot2_tmp), method='cubic', rescale=True).T
            print("Done")
        rhot[np.where(rhot < 0.)] = 0. #negative rhot values do not make sense
        rhot[np.isnan(rhot)] = 0.
        for n in range(0, names.size):
            corr_factor = 1./np.exp(-1.*rhot[:,:] * mu[n])
            corr_factor[np.where(corr_factor > 1000.)] = 1. # points with very low correction factor are not corrected; otherwise impossibly high values are obtained
            ims[n,:,:] = ims[n,:,:] * corr_factor
            sumint[n] = sumint[n] * np.average(corr_factor)

    
    # convert intensity values to concentrations
    h5_z = [Elements.getz(n.split(" ")[0]) for n in names]
    h5_lt = [n.split(" ")[1][0] for n in names] #linteype: K, L, M, ... even if linetype is K$\alpha$
    ref_lt = [n.split(" ")[1][0] for n in ref_names]
    for i in range(0, names.size):
        if names[i] in ref_names:
            ref_id = list(ref_names).index(names[i])
            ims[i,:,:] = ims[i,:,:]/ref_yld[ref_id]
            sumint[i] = sumint[i]/ref_yld[ref_id]
            ims_err[i,:,:] = np.sqrt(ims_err[i,:,:]*ims_err[i,:,:]+ref_yld_err[ref_id]*ref_yld_err[ref_id])
            sumint_err[i] = np.sqrt(sumint_err[i]*sumint_err[i]+ref_yld_err[ref_id]*ref_yld_err[ref_id])
        else: # element not in references list, so have to interpolate...
            if h5_lt[i] == 'K':
                line_id = [j for j, x in enumerate(ref_lt) if x == 'K']
            elif h5_lt[i] == 'L':
                line_id = [j for j, x in enumerate(ref_lt) if x == 'L']
            else:
                line_id = [j for j, x in enumerate(ref_lt) if x == 'M']
            if len(line_id) < 2:
                # there is only 1 element or even none in the ref with this linetype.
                #   if none, then don't quantify (set all to -1)
                #   if only 1 element, simply use that same el_yield, although it will probably give very wrong estimations
                if len(line_id) < 1:
                    ims[i,:,:] = -1
                    sumint[i] = -1
                    ims_err[i,:,:] = 0
                    sumint_err[i] = 0
                else:
                    ims[i,:,:] = ims[i,:,:] / ref_yld[line_id]
                    sumint[i] = sumint[i] / ref_yld[line_id]
                    ims_err[i,:,:] = np.sqrt(ims_err[i,:,:]*ims_err[i,:,:]+ref_yld_err[line_id]*ref_yld_err[line_id])
                    sumint_err[i] = np.sqrt(sumint_err[i]*sumint_err[i]+ref_yld_err[line_id]*ref_yld_err[line_id])
            else:
                # find ref indices of elements neighbouring h5_z[i]
                z_id = np.searchsorted(ref_z[line_id], h5_z[i]) #h5_z[i] is between index z_id-1 and z_id
                # check if z_id is either 0 or len(ref_z[line_id])
                #   in that case, do extrapolation with next 2 or previous 2 lines, if that many present
                if z_id == 0:
                    yld_interpol = (ref_yld[line_id][z_id+1]-ref_yld[line_id][z_id]) / (ref_z[line_id][z_id+1]-ref_z[line_id][z_id]) * (h5_z[i]-ref_z[line_id][z_id]) + ref_yld[line_id][z_id]
                    yld_interpol_err = (ref_yld_err[line_id][z_id+1]-ref_yld_err[line_id][z_id]) / (ref_z[line_id][z_id+1]-ref_z[line_id][z_id]) * (h5_z[i]-ref_z[line_id][z_id]) + ref_yld_err[line_id][z_id]
                elif z_id == len(ref_z[line_id]):
                    yld_interpol = (ref_yld[line_id][z_id-1]-ref_yld[line_id][z_id-2]) / (ref_z[line_id][z_id-1]-ref_z[line_id][z_id-2]) * (h5_z[i]-ref_z[line_id][z_id-2]) + ref_yld[line_id][z_id-2]
                    yld_interpol_err = (ref_yld_err[line_id][z_id-1]-ref_yld_err[line_id][z_id-2]) / (ref_z[line_id][z_id-1]-ref_z[line_id][z_id-2]) * (h5_z[i]-ref_z[line_id][z_id-2]) + ref_yld_err[line_id][z_id-2]
                else: #there is an element in ref_yld with index z_id-1 and z_id
                    yld_interpol = (ref_yld[line_id][z_id-1]-ref_yld[line_id][z_id]) / (ref_z[line_id][z_id-1]-ref_z[line_id][z_id]) * (h5_z[i]-ref_z[line_id][z_id]) + ref_yld[line_id][z_id]
                    yld_interpol_err = (ref_yld_err[line_id][z_id-1]-ref_yld_err[line_id][z_id]) / (ref_z[line_id][z_id-1]-ref_z[line_id][z_id]) * (h5_z[i]-ref_z[line_id][z_id]) + ref_yld_err[line_id][z_id]
                ims[i,:,:] = ims[i,:,:] / yld_interpol
                sumint[i] = sumint[i] / yld_interpol
                ims_err[i,:,:] = np.sqrt(ims_err[i,:,:]*ims_err[i,:,:]+yld_interpol_err*yld_interpol_err)
                sumint_err[i] = np.sqrt(sumint_err[i]*sumint_err[i]+yld_interpol_err*yld_interpol_err)
        # split rhot keyword into rho and t, and calculate the escape depth.
        if density is not None and thickness is not None:
            if composition is None:
                thick = thickness
            else:
                # read in cnc file data
                cnc = read_cnc(composition)
                el_name = names[i].split(" ")[0]
                # Check if the perceived sample thickness is bigger than the escape depth for said element.
                #   If K line, consider KL3 information depth, if L line the L3M5 ...
                if h5_lt[i] == 'K':
                    line = 'KL3' #Ka1
                elif h5_lt[i] == 'L':
                    line = 'L3M5' #La1
                elif h5_lt[i] == 'M':
                    line = 'M5N7' #Ma1
                mu = 0
                for j in range(0, cnc.z.size):
                    mu += Elements.getmassattcoef(Elements.getsymbol(cnc.z[j]), Elements.getxrayenergy(el_name, line))['total'][0] * cnc.conc[j]/1E6
                escape_depth = (np.log(100)/(density*mu)) #in cm
                if escape_depth < thickness:
                    thick = escape_depth
                else:
                    thick = thickness
            div_by_rhot = float(density*thick)
            ims[i,:,:] /= div_by_rhot
            sumint[i] /= div_by_rhot


    # # check which relative errors are largest: sumint_err or np.average(ims_err) or np.std(ims)/np.average(ims)
    # #   then use this error as the sumint_err
    # for i in range(sumint_err.size):
    #     sumint_err[i] = np.max(np.asarray([sumint_err[i], np.average(ims_err[i,:,:]), np.std(ims[i,:,:])/np.average(ims[i,:,:])]))

    # set appropriate concentration unit
    conc_unit = "ug/cm²"
    if density is not None and thickness is not None:
        conc_unit = "ug/g"
        
    # convert relative errors to absolute errors
    ims_err = ims_err*ims
    sumint_err = sumint_err*sumint
            
    # save quant data
    file = h5py.File(h5file, 'r+', locking=True)
    try:
        del file['quant/'+channel]
    except Exception:
        pass
    file.create_dataset('quant/'+channel+'/names', data=[n.encode('utf8') for n in names])
    dset = file.create_dataset('quant/'+channel+'/ims', data=ims, compression='gzip', compression_opts=4)
    dset.attrs["Unit"] = conc_unit
    dset = file.create_dataset('quant/'+channel+'/sum/int', data=sumint, compression='gzip', compression_opts=4)
    dset.attrs["Unit"] = conc_unit
    dset = file.create_dataset('quant/'+channel+'/ims_stddev', data=ims_err, compression='gzip', compression_opts=4)
    dset.attrs["Unit"] = conc_unit
    dset = file.create_dataset('quant/'+channel+'/sum/int_stddev', data=sumint_err, compression='gzip', compression_opts=4)
    dset.attrs["Unit"] = conc_unit
    if reffiles.size > 1:
        ' '.join(reffiles)
    file.create_dataset('quant/'+channel+'/refs', data=str(reffiles).encode('utf8'))
    dset = file['quant']
    dset.attrs["LastUpdated"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if composition is not None:
        dset.attrs["Composition"] = composition
    if density is not None and thickness is not None:
        dset.attrs["density"] = density
        dset.attrs["thickness"] = thickness
    if absorb is not None:
        file.create_dataset('quant/'+channel+'/ratio_exp', data=ratio_ka1_kb1, compression='gzip', compression_opts=4)
        file.create_dataset('quant/'+channel+'/ratio_th', data=rate_ka1/rate_kb1)
        file.create_dataset('quant/'+channel+'/rhot', data=rhot, compression='gzip', compression_opts=4)
    file.close()

    # plot images
    data = Xims.ims()
    data.data = np.zeros((ims.shape[1],ims.shape[2], ims.shape[0]+1))
    for i in range(0, ims.shape[0]):
        data.data[:, :, i] = ims[i, :, :]
    if absorb is not None:
        data.data[:,:,-1] = rhot[:,:]
    names = np.concatenate((names,[r'$\rho T$']))
    data.names = names
    cb_opts = Xims.Colorbar_opt(title=r'Conc.;[$\mu$g/cm²]')
    nrows = int(np.ceil(len(names)/4)) # define nrows based on ncols
    colim_opts = Xims.Collated_image_opts(ncol=4, nrow=nrows, cb=True)
    Xims.plot_colim(data, np.arange(len(names)), 'viridis', cb_opts=cb_opts, colim_opts=colim_opts, save=os.path.splitext(h5file)[0]+'_ch'+channel[-1]+'_quant.png')

    return True

##############################################################################
def plot_detlim(dl, el_names, tm=None, ref=None, dl_err=None, bar=False, save=None, ytitle="Detection Limit (ppm)"):
    """
    Create detection limit image that is of publishable quality, including 3sigma error bars.
    

    Parameters
    ----------
    dl : float array
        Float array of dimensions ([n_ref, ][n_tm, ]n_elements) containing the detection limit data.
    el_names : string array
        String array containing the element labels for which data is provided.
    tm : float array or list, optional
        Measurement times, including the unit, associated with the provided detection limits. The default is None.
    ref : string array or list, optional
        Labels of the reference materials for which data is provided. The default is None.
    dl_err : float array, optional
        Error values (1sigma) corresponding to the detection limit values. The default is None.
    bar : Boolean, optional
        If True the default scatter plot is replaced by a bar-plot (histogram plot). The default is False.
    save : String, optional
        File path directory in which the detection limit plot will be saved. The default is None, meaning plot will not be saved.
    ytitle : String, optional
        Label to be used for the y-axis title. The default is "Detection Limit (ppm)".

    Returns
    -------
    bool
        returns False on error.

    """
    tickfontsize = 20
    toptickfontsize = 8
    titlefontsize = 22

    # check shape of dl. If 1D, then only single curve selected. 2D array means several DLs
    dl = np.asarray(dl, dtype='object')
    el_names = np.asarray(el_names, dtype='object')
    if tm:
        tm = np.asarray(tm)
    if ref:
        ref = np.asarray(ref)
    if dl_err is not None:
        dl_err = np.asarray(dl_err, dtype='object')[:]*3. # we plot 3sigma error bars.
    # verify that el_names, dl_err are also 2D if provided
    if len(el_names.shape) != len(dl.shape):
        print("Error: el_names should be of similar dimension as dl")
        return False
    if dl_err is not None and len(dl_err.shape) != len(dl.shape):
        print("Error: dl_err should be of similar dimension as dl")
        return False

    marker = itertools.cycle(('o', 's', 'd', 'p', '^', 'v')) #some plot markers to cycle through in case of multiple curves
    # make el_z array from el_names
    if len(el_names.shape) == 1 and type(el_names[0]) is type(str()):
        el_names = np.asarray(el_names, dtype='str')
        all_names = np.asarray([str(name) for name in np.nditer(el_names)])
        all_dl = np.nditer(dl)
    else:
        all_names = []
        all_dl = []
        for i in range(0,len(el_names)):
            for j in range(0, len(el_names[i])):
                all_names.append(el_names[i][j])
                all_dl.append(dl[i][j])
        all_names = np.asarray(all_names, dtype='str')
        all_dl = np.asarray(all_dl, dtype='float')
    el_z = np.asarray([Elements.getz(name.split(" ")[0]) for name in all_names])
    # count unique Z's
    unique_z = np.unique(el_z)
    #set figure width and height
    height = 5.
    if (unique_z.size+1)*0.65 > 6.5:
        width = (unique_z.size+1)*0.65
    else:
        width = 6.5
    plt.figure(figsize=(width, height), tight_layout=True)
    
    # add axis on top of graph with element and line names
    unique_el, unique_id = np.unique(all_names, return_index=True) # sorts unique_z names alphabetically instead of increasing Z!!
    dl_av = [np.average(all_dl[np.where(all_names == tag)]) for tag in unique_el]
    el_labels = ["-".join(n.split(" ")) for n in unique_el]
    z_temp = el_z[unique_id]
    unique_el = np.asarray(el_labels)[np.argsort(z_temp)]
    dl_av = np.asarray(dl_av)[np.argsort(z_temp)]
    z_temp = z_temp[np.argsort(z_temp)]
    # if for same element/Z multiple lines, join them.
    # K or Ka should be lowest, L above, M above that, ... (as typically K gives lowest DL, L higher, M higher still...)
    unique_z, unique_id = np.unique(z_temp, return_index=True)
    if unique_z.size != z_temp.size:
        new_z = unique_z
        new_labels = []
        for i in range(0, unique_z.size):
            z_indices = np.where(z_temp == unique_z[i])[0]
            dl_order = np.flip(np.argsort(dl_av[z_indices]))
            new_labels.append("\n".join(unique_el[z_indices][dl_order]))
        new_labels = np.asarray(new_labels)
    else:
        new_z = np.asarray(z_temp)
        new_labels = np.asarray(el_labels)
    new_labels = new_labels[np.argsort(new_z)]
    new_z = new_z[np.argsort(new_z)]

    # actual plotting
    if len(dl.shape) == 1 and type(el_names[0]) is type(np.str_()):
        # only single dl range is provided
        # plot curves and axes
        if bar is True:
            bar_x = np.zeros(el_z.size)
            for i in range(0,el_z.size):
                bar_x[i] = list(unique_el).index("-".join(el_names[i].split(" ")))
            plt.bar(bar_x, dl, yerr=dl_err, label=str(ref)+'_'+str(tm), capsize=3)
            ax = plt.gca()
            ax.set_xticks(np.linspace(0,unique_el.size-1,num=unique_el.size))
            ax.set_xticklabels(unique_el, fontsize=toptickfontsize)
        else:
            plt.errorbar(el_z, dl, yerr=dl_err, label=str(ref)+'_'+str(tm), linestyle='', fmt=next(marker), capsize=3)
            ax = plt.gca()
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            plt.xlabel("Atomic Number (Z)", fontsize=titlefontsize)
            secaxx = ax.secondary_xaxis('top')
            secaxx.set_xticks(new_z)
            secaxx.set_xticklabels(new_labels, fontsize=toptickfontsize)
            # fit curve through points and plot as dashed line in same color
            try:
                fit_par = np.polyfit(el_z, np.log(dl), 2)
                func = np.poly1d(fit_par)
                fit_x = np.linspace(np.min(el_z), np.max(el_z), num=(np.max(el_z)-np.min(el_z))*2)
                plt.plot(fit_x, np.exp(func(fit_x)), linestyle='--', color=plt.legend().legendHandles[0].get_colors()[0]) #Note: this also plots a legend, which is removed later on.
                ax.get_legend().remove()
            except Exception:
                pass
        plt.ylabel(ytitle, fontsize=titlefontsize)
        plt.yscale('log')
        plt.yticks(fontsize=tickfontsize)
        plt.xticks(fontsize=tickfontsize)
        # add legend
        handles, labels = ax.get_legend_handles_labels() # get handles
        handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles] # remove the errorbars
        plt.legend(handles, labels, loc='best', fontsize=titlefontsize)
        plt.show()
    elif len(dl.shape) == 2:
        # multiple dl ranges are provided. Loop over them, annotate differences between tm and ref comparissons
        #   Only 1 of the two (tm or ref) should be of size >1; loop over that one
        if (tm is None and ref is None) or (tm.size == 1 and ref.size == 1):
            ref = np.asarray(['DL '+str(int(n)) for n in np.linspace(0, dl.shape[0]-1, num=dl.shape[0])])
        if tm is not None and tm.size > 1:
            if ref is not None:
                label_prefix = str(ref[0])+"_"
            else:
                label_prefix = ''
            for i in range(0, tm.size):
                # plot curves and axes
                if bar is True:
                    el = np.asarray(["-".join(name.split(" ")) for name in el_names[i]])
                    bar_x = np.zeros(el.size)
                    for k in range(0,el.size):
                        bar_x[k] = list(unique_el).index(el[i]) + (0.9/tm.size)*(i-(tm.size-1)/2.)
                    plt.bar(bar_x, dl[i], yerr=dl_err[i], label=label_prefix+str(tm[i]), capsize=3, width=(0.9/tm.size))
                    ax = plt.gca()
                    if i == 0:
                        ax.set_xticks(np.linspace(0,unique_el.size-1,num=unique_el.size))
                        ax.set_xticklabels(unique_el, fontsize=toptickfontsize)
                else:
                    el_z = np.asarray([Elements.getz(name.split(" ")[0]) for name in el_names[i]])
                    plt.errorbar(el_z, dl[i], yerr=dl_err[i], label=label_prefix+str(tm[i]), linestyle='', fmt=next(marker), capsize=3)
                    ax = plt.gca()
                    if i == 0:
                        plt.xlabel("Atomic Number (Z)", fontsize=titlefontsize)
                        ax.xaxis.set_minor_locator(MultipleLocator(1))
                        secaxx = ax.secondary_xaxis('top')
                        secaxx.set_xticks(new_z)
                        secaxx.set_xticklabels(new_labels, fontsize=toptickfontsize)
                    # fit curve through points and plot as dashed line in same color
                    try:
                        fit_par = np.polyfit(el_z, np.log(np.asarray(dl[i], dtype='float64')), 2)
                        func = np.poly1d(fit_par)
                        fit_x = np.linspace(np.min(el_z), np.max(el_z), num=(np.max(el_z)-np.min(el_z))*2)
                        plt.plot(fit_x, np.exp(func(fit_x)), linestyle='--', color=plt.legend().legendHandles[-1].get_colors()[0]) #Note: this also plots a legend, which is removed later on.
                        ax.get_legend().remove()
                    except Exception:
                        pass
            plt.ylabel(ytitle, fontsize=titlefontsize)
            plt.yscale('log')
            plt.yticks(fontsize=tickfontsize)
            plt.xticks(fontsize=tickfontsize)
            # add legend
            handles, labels = ax.get_legend_handles_labels() # get handles
            handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles] # remove the errorbars
            plt.legend(handles, labels, loc='best', fontsize=titlefontsize)
            plt.show()
        elif ref is not None and ref.size > 1:
            if tm:
                label_suffix = "_"+str(tm)
            else:
                label_suffix = ''
            for i in range(0, ref.size):
                # plot curves and axes
                if bar is True:
                    el = np.asarray(["-".join(name.split(" ")) for name in el_names[i]])
                    bar_x = np.zeros(el.size) + (0.9/ref.size)*(i-(ref.size-1)/2.)
                    for k in range(0,el.size):
                        bar_x[k] = list(unique_el).index(el[k])
                    plt.bar(bar_x, dl[i], yerr=dl_err[i], label=str(ref[i])+label_suffix, capsize=3, width=(0.9/ref.size))
                    ax = plt.gca()
                    if i == 0:
                        ax.set_xticks(np.linspace(0,unique_el.size-1,num=unique_el.size))
                        ax.set_xticklabels(unique_el, fontsize=toptickfontsize)
                else:
                    el_z = np.asarray([Elements.getz(name.split(" ")[0]) for name in el_names[i]])
                    plt.errorbar(el_z, dl[i], yerr=dl_err[i], label=str(ref[i])+label_suffix, linestyle='', fmt=next(marker), capsize=3)
                    ax = plt.gca()
                    if i == 0:
                        plt.xlabel("Atomic Number (Z)", fontsize=titlefontsize)
                        ax.xaxis.set_minor_locator(MultipleLocator(1))
                        secaxx = ax.secondary_xaxis('top')
                        secaxx.set_xticks(new_z)
                        secaxx.set_xticklabels(new_labels, fontsize=toptickfontsize)
                    # fit curve through points and plot as dashed line in same color
                    try:
                        fit_par = np.polyfit(el_z, np.log(dl[i]), 2)
                        func = np.poly1d(fit_par)
                        fit_x = np.linspace(np.min(el_z), np.max(el_z), num=(np.max(el_z)-np.min(el_z))*2)
                        plt.plot(fit_x, np.exp(func(fit_x)), linestyle='--', color=plt.legend().legendHandles[-1].get_colors()[0]) #Note: this also plots a legend, which is removed later on.
                        ax.get_legend().remove()
                    except Exception:
                        pass
            plt.ylabel(ytitle, fontsize=titlefontsize)
            plt.yscale('log')
            plt.yticks(fontsize=tickfontsize)
            plt.xticks(fontsize=tickfontsize)
            # add legend
            handles, labels = ax.get_legend_handles_labels() # get handles
            handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles] # remove the errorbars
            plt.legend(handles, labels, loc='best', fontsize=titlefontsize)
            plt.show()
        else:
            print("Error: ref and/or tm dimensions do not fit dl dimensions.")
            return False
            
    elif len(dl.shape) == 3:
        # multiple dl ranges, loop over both tm and ref
        if tm is None:
            tm = np.asarray(['tm'+str(int(n)) for n in np.linspace(0, dl.shape[0]-1, num=dl.shape[0])])
        if ref is None:
            ref = np.asarray(['ref'+str(int(n)) for n in np.linspace(0, dl.shape[1]-1, num=dl.shape[1])])
        for i in range(0, ref.size):
            for j in range(0, tm.size):
                # plot curves and axes
                if bar is True:
                    el = np.asarray(["-".join(name.split(" ")) for name in el_names[i,j]])
                    bar_x = np.zeros(el.size)
                    for k in range(0,el.size):
                        bar_x[k] = list(unique_el.index(el[k]) + (0.9/(tm.size*ref.size))*(i*tm.size+j-(tm.size*ref.size-1)/2.))
                    plt.bar(bar_x, dl[i,j], yerr=dl_err[i,j], label=str(ref[i])+'_'+str(tm[j]), capsize=3, width=(0.9/(tm.size*ref.size)))
                    ax = plt.gca()
                    if i == 0 and j == 0:
                        ax.set_xticks(np.linspace(0, unique_el.size-1, num=unique_el.size))
                        ax.set_xticklabels(unique_el, fontsize=toptickfontsize)
                else:
                    el_z = np.asarray([Elements.getz(name.split(" ")[0]) for name in el_names[i,j]])
                    plt.errorbar(el_z, dl[i,j], yerr=dl_err[i,j], label=str(ref[i])+'_'+str(tm[j]), linestyle='', fmt=next(marker), capsize=3)
                    ax = plt.gca()
                    if i == 0 and j == 0:
                        plt.xlabel("Atomic Number (Z)", fontsize=titlefontsize)
                        ax.xaxis.set_minor_locator(MultipleLocator(1))
                        secaxx = ax.secondary_xaxis('top')
                        secaxx.set_xticks(new_z)
                        secaxx.set_xticklabels(new_labels, fontsize=toptickfontsize)
                    # fit curve through points and plot as dashed line in same color
                    try:
                        fit_par = np.polyfit(el_z, np.log(dl[i,j]), 2)
                        func = np.poly1d(fit_par)
                        fit_x = np.linspace(np.min(el_z), np.max(el_z), num=(np.max(el_z)-np.min(el_z))*2)
                        plt.plot(fit_x, np.exp(func(fit_x)), linestyle='--', color=plt.legend().legendHandles[-1].get_colors()[0]) #Note: this also plots a legend, which is removed later on.
                        ax.get_legend().remove()
                    except Exception:
                        pass
                plt.ylabel(ytitle, fontsize=titlefontsize)
                plt.yscale('log')
                plt.yticks(fontsize=tickfontsize)
                plt.xticks(fontsize=tickfontsize)
        # add legend
        handles, labels = ax.get_legend_handles_labels() # get handles
        handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles] # remove the errorbars
        plt.legend(handles, labels, loc='best', fontsize=titlefontsize)
        plt.show()  
    elif (len(dl.shape) == 1 and type(el_names[0]) is not type(np.str_())):
        # multiple dl ranges with different length, loop over both tm and ref
        if tm is None:
            tm = np.asarray(['tm'+str(int(n)) for n in np.linspace(0, dl.shape[0]-1, num=dl.shape[0])])
        if tm.size == 1:
            tm_tmp = [tm]
        if ref is None:
            ref = np.asarray(['ref'+str(int(n)) for n in np.linspace(0, dl.shape[1]-1, num=dl.shape[1])])
        for i in range(0, ref.size):
            el_names_tmp = el_names[i]
            dl_tmp = dl[i]
            dl_err_tmp = dl_err[i]
            for j in range(0, tm.size):
                # plot curves and axes
                if bar is True:
                    el = np.asarray(["-".join(name.split(" ")) for name in el_names_tmp])
                    bar_x = np.zeros(el.size)
                    for k in range(0,el.size):
                        bar_x[k] = list(unique_el).index(el[k]) + (0.9/(tm.size*ref.size))*(i*tm.size+j-(tm.size*ref.size-1)/2.)
                    plt.bar(bar_x, dl_tmp, yerr=dl_err_tmp, label=str(ref[i])+'_'+str(tm_tmp[j]), capsize=3, width=(0.9/(tm.size*ref.size)))
                    ax = plt.gca()
                    if i == 0 and j == 0:
                        ax.set_xticks(np.linspace(0, unique_el.size-1, num=unique_el.size))
                        ax.set_xticklabels(unique_el, fontsize=toptickfontsize)
                else:
                    el_z = np.asarray([Elements.getz(name.split(" ")[0]) for name in el_names_tmp])
                    plt.errorbar(el_z, dl_tmp, yerr=dl_err_tmp, label=str(ref[i])+'_'+str(tm_tmp[j]), linestyle='', fmt=next(marker), capsize=3)
                    ax = plt.gca()
                    if i == 0 and j == 0:
                        plt.xlabel("Atomic Number (Z)", fontsize=titlefontsize)
                        ax.xaxis.set_minor_locator(MultipleLocator(1))
                        secaxx = ax.secondary_xaxis('top')
                        secaxx.set_xticks(new_z)
                        secaxx.set_xticklabels(new_labels, fontsize=toptickfontsize)
                    # fit curve through points and plot as dashed line in same color
                    try:
                        fit_par = np.polyfit(el_z, np.log(dl_tmp), 2)
                        func = np.poly1d(fit_par)
                        fit_x = np.linspace(np.min(el_z), np.max(el_z), num=(np.max(el_z)-np.min(el_z))*2)
                        plt.plot(fit_x, np.exp(func(fit_x)), linestyle='--', color=plt.legend().legendHandles[-1].get_colors()[0]) #Note: this also plots a legend, which is removed later on.
                        ax.get_legend().remove()
                    except Exception:
                        pass
                plt.ylabel(ytitle, fontsize=titlefontsize)
                plt.yscale('log')
                plt.yticks(fontsize=tickfontsize)
                plt.xticks(fontsize=tickfontsize)
        # add legend
        handles, labels = ax.get_legend_handles_labels() # get handles
        handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles] # remove the errorbars
        plt.legend(handles, labels, loc='best', fontsize=titlefontsize)
        plt.show()                
              
    else:
        print("Error: input argument: dl dimension is >= 4. dl should be of shape (n_elements[, n_tm][, n_ref])")
        return False
    
    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
        plt.close()

##############################################################################
def calc_detlim(h5file, cncfile, plotytitle="Detection Limit (ppm)", sampletilt=90):
    """
    Calculate detection limits following the equation DL = 3*sqrt(Ib)/Ip * Conc
      Calculates 1s and 1000s DL
      Also calculates elemental yields (Ip/conc [(ct/s)/(ug/cm²)]) 

    Parameters
    ----------
    h5file : string
        File directory path to the H5 file containing the data.
    cncfile : string
        File directory path to the CNC file containing the reference material composition information.
    ytitle : String, optional
        Label to be used for the y-axis title during detection limit plotting. The default is "Detection Limit (ppm)".
    sampletilt: float, optional
        Angle in degrees between sample surface and incidence beam. Currently this value is only used during calculation of 
        areal concentrations, where the sample thickness registered in the cncfile is corrected by sin(sampletilt). 
        The default is 90 degrees (incidence beam is perpendicular to sample surface).

    Yields
    ------
    None.

    """
    # read in cnc file data
    cnc = read_cnc(cncfile)
    
    # read h5 file
    with h5py.File(h5file, 'r', locking=True) as file:
        if 'norm' not in file.keys():
            print("ERROR: calc_detlim: cannot open normalised data in "+h5file)
            return
        keys = [key for key in file['norm'].keys() if 'channel' in key]
        tm = np.asarray(file['raw/acquisition_time']) # Note this is pre-normalised tm! Correct for I0 value difference between raw and I0norm
        I0 = np.asarray(file['raw/I0'])
        I0norm = np.asarray(file['norm/I0'])
        tmnorm = str(file['norm'].attrs["TmNorm"])

    if tmnorm == "True":
        tmnorm = True
    else:
        tmnorm = False
        
    # correct tm for appropriate normalisation factor
    #   tm is time for which DL would be calculated using values as reported, taking into account the previous normalisation factor
    if tmnorm is True:
        normfactor = I0norm/(np.sum(I0*tm))
    else:
        normfactor = I0norm/np.sum(I0)
    tm = np.sum(tm)

    for index, chnl in enumerate(keys):
        with h5py.File(h5file, 'r', locking=True) as file:
            sum_fit0 = np.asarray(file['norm/'+chnl+'/sum/int'])
            sum_bkg0 = np.asarray(file['norm/'+chnl+'/sum/bkg'])
            sum_fit0_err = np.asarray(file['norm/'+chnl+'/sum/int_stddev'])/sum_fit0
            sum_bkg0_err = np.asarray(file['norm/'+chnl+'/sum/bkg_stddev'])/sum_bkg0
            names0 = [n for n in file['norm/'+chnl+'/names']]
                    
        # undo normalisation on intensities as performed during norm_xrf_batch
        #   in order to get intensities matching the current tm value (i.e. equal to raw fit values)
        names0 = np.asarray([n.decode('utf8') for n in names0[:]])
        if 'Compt' in list(names0):
            sum_fit0 = sum_fit0[np.arange(len(names0))!=list(names0).index('Compt')]
            sum_bkg0 = sum_bkg0[np.arange(len(names0))!=list(names0).index('Compt')]
            sum_fit0_err = sum_fit0_err[np.arange(len(names0))!=list(names0).index('Compt')]
            sum_bkg0_err = sum_bkg0_err[np.arange(len(names0))!=list(names0).index('Compt')]
            names0 = names0[np.arange(len(names0))!=list(names0).index('Compt')]
        if 'Rayl' in list(names0):
            sum_fit0 = sum_fit0[np.arange(len(names0))!=list(names0).index('Rayl')]
            sum_bkg0 = sum_bkg0[np.arange(len(names0))!=list(names0).index('Rayl')]
            sum_fit0_err = sum_fit0_err[np.arange(len(names0))!=list(names0).index('Rayl')]
            sum_bkg0_err = sum_bkg0_err[np.arange(len(names0))!=list(names0).index('Rayl')]
            names0 = names0[np.arange(len(names0))!=list(names0).index('Rayl')]
        sum_bkg0 = sum_bkg0/normfactor
        sum_fit0 = sum_fit0/normfactor
        # prune cnc.conc array to appropriate elements according to names0
        #   creates arrays of size names0 , where 0 values in conc0 represent elements not stated in cnc_files.
        conc0 = np.zeros(names0.size)
        conc0_err = np.zeros(names0.size)
        conc0_areal = np.zeros(names0.size)
        conc0_areal_err = np.zeros(names0.size)
        for j in range(0, names0.size):
            el_name = names0[j].split(" ")[0]
            # Check if the perceived sample thickness is bigger than the information depth for said element.
            #   If K line, consider KL3 information depth, if L line the L3M5 ...
            line = names0[j].split(' ')[1]
            if line[0] == 'K':
                line = 'KL3' #Ka1
            elif line[0] == 'L':
                line = 'L3M5' #La1
            elif line[0] == 'M':
                line = 'M5N7' #Ma1
            mu = 0
            for i in range(0, cnc.z.size):
                mu += Elements.getmassattcoef(Elements.getsymbol(cnc.z[i]), Elements.getxrayenergy(el_name, line))['total'][0] * cnc.conc[i]/1E6 #in cm²/g
            escape_depth = (np.log(100)/(cnc.density*1E-3*mu)) #in cm
            virtual_sample_thickness = (cnc.thickness/np.sin(sampletilt/180.*np.pi))*1E-4 #in cm
            for i in range(0, cnc.z.size):
                if el_name == Elements.getsymbol(cnc.z[i]):
                    if escape_depth > virtual_sample_thickness:
                        conc0_areal[j] = cnc.conc[i]*cnc.density*virtual_sample_thickness*1E-3 # unit: [ug/cm²]
                    else:
                        conc0_areal[j] = cnc.conc[i]*cnc.density*escape_depth*1E-3 # unit: [ug/cm²]
                    conc0_areal_err[j] = (cnc.err[i]/cnc.conc[i])*conc0_areal[j] # unit: [ug/cm²]
                    conc0[j] = cnc.conc[i] # unit: [ppm]
                    conc0_err[j] = (cnc.err[i]/cnc.conc[i])*conc0[j] # unit: [ppm]    
        
        # some values will be 0 (due to conc0 or conc1 being 0). Ignore these in further calculations.
        names0_mod = []
        dl_1s_0 = []
        dl_1000s_0 = []
        dl_1s_err_0 = []
        dl_1000s_err_0 = []
        el_yield_0 = []
        el_yield_err_0 = []
        for i in range(0, conc0.size):
            if conc0[i] > 0:
                # detection limit corresponding to tm=1s
                dl_1s_0.append( (3.*np.sqrt(sum_bkg0[i]/tm)/(sum_fit0[i]/tm)) * conc0[i])
                j = len(dl_1s_0)-1
                dl_1000s_0.append(dl_1s_0[j] / np.sqrt(1000.))
                el_yield_0.append((sum_fit0[i]*normfactor/I0norm) / conc0_areal[i]) # element yield expressed as cps/conc
                # calculate DL errors (based on standard error propagation)
                dl_1s_err_0.append(np.sqrt(sum_fit0_err[i]**2 + sum_bkg0_err[i]**2 +
                                         (conc0_err[i]/conc0[i])*(conc0_err[i]/conc0[i])) * dl_1s_0[j])
                dl_1000s_err_0.append(dl_1s_err_0[j] / dl_1s_0[j] * dl_1000s_0[j])
                el_yield_err_0.append(np.sqrt((conc0_areal_err[i]/conc0_areal[i])*(conc0_areal_err[i]/conc0_areal[i]) + sum_fit0_err[i]**2)*el_yield_0[j])
                names0_mod.append(names0[i])

        # save DL data to file
        cncfile = cncfile.split("/")[-1]
        with h5py.File(h5file, 'r+', locking=True) as file:
            # remove old keys as these are now redundant, we should use a single cncfile for either detector channel
            if 'detlim' in file.keys():
                for key in [k for k in file['detlim/'].keys()]:
                    if key != cncfile:
                        del file['detlim/'+key]
                        del file['elyield/'+key]
                try:
                    del file['detlim/'+cncfile+'/unit']
                except Exception:
                    pass
            file.create_dataset('detlim/'+cncfile+'/unit', data='ppm')
            try:
                del file['detlim/'+cncfile+'/'+chnl+'/names']
                del file['detlim/'+cncfile+'/'+chnl+'/1s/data']
                del file['detlim/'+cncfile+'/'+chnl+'/1s/stddev']
                del file['detlim/'+cncfile+'/'+chnl+'/1000s/data']
                del file['detlim/'+cncfile+'/'+chnl+'/1000s/stddev']
                del file['elyield/'+cncfile+'/'+chnl+'/yield']
                del file['elyield/'+cncfile+'/'+chnl+'/stddev']
                del file['elyield/'+cncfile+'/'+chnl+'/names']
            except Exception:
                pass
            file.create_dataset('detlim/'+cncfile+'/'+chnl+'/names', data=[n.encode('utf8') for n in names0_mod[:]])
            file.create_dataset('detlim/'+cncfile+'/'+chnl+'/1s/data', data=dl_1s_0, compression='gzip', compression_opts=4)
            file.create_dataset('detlim/'+cncfile+'/'+chnl+'/1s/stddev', data=dl_1s_err_0, compression='gzip', compression_opts=4)
            file.create_dataset('detlim/'+cncfile+'/'+chnl+'/1000s/data', data=dl_1000s_0, compression='gzip', compression_opts=4)
            file.create_dataset('detlim/'+cncfile+'/'+chnl+'/1000s/stddev', data=dl_1000s_err_0, compression='gzip', compression_opts=4)    
            dset = file.create_dataset('elyield/'+cncfile+'/'+chnl+'/yield', data=el_yield_0, compression='gzip', compression_opts=4)
            dset.attrs["Unit"] = "(ct/s)/(ug/cm²)"
            dset = file.create_dataset('elyield/'+cncfile+'/'+chnl+'/stddev', data=el_yield_err_0, compression='gzip', compression_opts=4)
            dset.attrs["Unit"] = "(ct/s)/(ug/cm²)"
            file.create_dataset('elyield/'+cncfile+'/'+chnl+'/names', data=[n.encode('utf8') for n in names0_mod[:]])
            dset = file['detlim']
            dset.attrs["LastUpdated"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            dset = file['elyield']
            dset.attrs["LastUpdated"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    
        # plot the DLs
        plot_detlim([dl_1s_0, dl_1000s_0],
                    [names0_mod, names0_mod],
                    tm=['1s','1000s'], ref=['DL'], 
                    dl_err=[dl_1s_err_0, dl_1000s_err_0], bar=False, save=str(os.path.splitext(h5file)[0])+'_ch'+str(index)+'_DL.png', ytitle=plotytitle)

##############################################################################
def hdf_overview_images(h5file, datadir, ncols, pix_size, scl_size, clrmap='viridis', log=False, sqrt=False, rotate=0, fliph=False, cb_opts=None, clim=None, dpi=420):
    """
    Generate publishing quality overview images of all fitted elements in H% file (including scale bars, colorbar, ...)

    Parameters
    ----------
    h5file : string
        File directory path to the H5 file containing the data.
    h5dir : string
        Data directory within the H5 file containing the data to be analysed.
    ncols : integer
        Amount of columns in which the overview image will be displayed.
    pix_size : float
        Image pixel size in µm.
    scl_size : float
        Image scale bar size to be displayed in µm.
    clrmap: string, optional
        Image color map. The default is 'viridis'.
    log : Boolean, optional
        If True, the Briggs logarithm of the data is displayed. The default is False.
    sqrt : Boolean, optional
        If True, the square root of the data is displayed. The default is False.
    rotate : integer, optional
        Amount of degrees, rounded to nearest 90, over which images should be rotated. The default is 0.
    fliph : Boolean, optional
        If True, data is flipped over the horizontal image axes (after optional rotation). The default is False.
    cb_opts : Xims.Colorbar_opt class, optional
        User supplied intensity scale colorbar options for imaging. The default is None.
    clim : list, optional
        List containing the lower and upper intensity limit to apply to the plots. The default is None, indicating no specific limits.

    Returns
    -------
    None.

    """
    import Xims
    filename = os.path.splitext(h5file)[0]

    imsdata0 = Xims.read_h5(h5file, datadir+'/channel00/ims')
    imsdata0.data[imsdata0.data < 0] = 0.
    if log:
        filename += '_log'
    if sqrt:
        filename += '_sqrt'
    try:
        imsdata1 = Xims.read_h5(h5file, datadir+'/channel01/ims')
        if imsdata1 is None:
            chan01_flag = False
        else:
            imsdata1.data[imsdata1.data < 0] = 0.
            chan01_flag = True
    except Exception:
        chan01_flag = False

    # rotate where appropriate
    if rotate != 0:
        imsdata0.data = np.rot90(imsdata0.data, k=np.rint(rotate/90.), axes=(0,1))
        if chan01_flag == True:
            imsdata1.data = np.rot90(imsdata1.data, k=np.rint(rotate/90.), axes=(0,1))
    # flip image horizontally
    if fliph is True:
        imsdata0.data = np.flip(imsdata0.data, axis=0)
        if chan01_flag == True:
            imsdata1.data = np.flip(imsdata1.data, axis=0)
            
    sb_opts = Xims.Scale_opts(xscale=True, x_pix_size=pix_size, x_scl_size=scl_size, x_scl_text=str(scl_size)+' µm')
    if cb_opts is None:
        if log:
            cb_opts = Xims.Colorbar_opt(title='log. Int.;[cts]')
        elif sqrt:
            cb_opts = Xims.Colorbar_opt(title='sqrt. Int.;[cts]')
        else:
            cb_opts = Xims.Colorbar_opt(title='Int.;[cts]')
    nrows = int(np.ceil(len(imsdata0.names)/ncols)) # define nrows based on ncols
    colim_opts = Xims.Collated_image_opts(ncol=ncols, nrow=nrows, cb=True, clim=clim)

    if log:
        imsdata0.data = np.log10(imsdata0.data)
    if sqrt:
        imsdata0.data = np.sqrt(imsdata0.data)

    
    Xims.plot_colim(imsdata0, np.arange(len(imsdata0.names)), clrmap, sb_opts=sb_opts, cb_opts=cb_opts, colim_opts=colim_opts, save=filename+'_ch0_'+datadir+'_overview.png', dpi=dpi)
    
    if chan01_flag == True:
        nrows = int(np.ceil(len(imsdata1.names)/ncols)) # define nrows based on ncols
        colim_opts = Xims.Collated_image_opts(ncol=ncols, nrow=nrows, cb=True, clim=clim)

        if log:
            imsdata1.data = np.log10(imsdata1.data)
        if sqrt:
            imsdata1.data = np.sqrt(imsdata1.data)
        
        Xims.plot_colim(imsdata1, np.arange(len(imsdata0.names)), clrmap, sb_opts=sb_opts, cb_opts=cb_opts, colim_opts=colim_opts, save=filename+'_ch1_'+datadir+'_overview.png', dpi=dpi)


##############################################################################
def norm_xrf_batch(h5file, I0norm=None, snake=False, sort=False, timetriggered=False, tmnorm=False, 
                   halfpixshift=True, mot2nosort=False, omitspectra=False, interpol_method='nearest',
                   lowmemory=False):
    """
    Function to normalise IMS images to detector deadtime and I0 values.

    Parameters
    ----------
    h5file : string
        File directory path to the H5 file containing the data.
    I0norm : integer, optional
        When I0norm is supplied, a (long) int should be provided to which I0 value one should normalise. Otherwise the max of the I0 map is used. The default is None.
    snake : Boolean, optional
        If the scan was performed following a snake-like pattern, set to True to allow for appropriate image reconstruction. The default is False.
    sort : Boolean, optional
        Sorts the data using the motor positions. Typically this is already done during initial raw data conversion, but in some occassions it may be opportune
        to omit it there and perform it during the normalisation step. The default is False.
    timetriggered : Boolean, optional
        If data is collected following a time triggered scheme rather than position triggered, then motor position interpolation is done differently.
        The default is False.
    tmnorm : Boolean, optional
        When I0 values are not scaled for acquisition time, then additional normalisation for acquisition time can be performed by setting tmnorm to True. The default is False.
    halfpixshift : Boolean, optional
        This value is only used when snake is True. Implements a half pixel shift in the motor encoder positions to account of the bidirectional event triggering. The default is True.
    mot2nosort : Boolean, optional
        When sort is True, mot2nosort can be set to True to omit sorting using the mot2 encoder values. The default is False.
    omitspectra : Boolean, optional
        When omitspectra is True, sorting and interpolation will not be performed on the /raw/channelXX/spectra, icr and ocr datasets.The default is False.
        This can be beneficial in case of handling particularly large datasets, but be warned that this may cause unwanted effects during further processing
        as the raw spectra data will no longer be able to be directly correlated to the element image data. For instance during PCA or Kmeans clustering, the 
        resulting cluster sum spectra may not be indicative of the intended region.
    interpol_method : string, optional
        The scipy.griddata interpolation method to be used, as supplied to the griddata function. The default is 'nearest'.
    lowmemory : Boolean, optional
        Set True to interpolate spectra for motor positions in a more memory efficient way. Note that this significantly increases the required processing time. The default is False.

    Returns
    -------
    None.

    """
    print("Initiating data normalisation of <"+h5file+">...", end=" ")
    # read h5file
    with h5py.File(h5file, 'r', locking=True) as file:
        keys = [key for key in file['fit'].keys() if 'channel' in key]
        I0_raw =  np.asarray(file['raw/I0'])
        tm_raw = np.asarray(file['raw/acquisition_time'])
        mot1_raw = np.asarray(file['mot1'])
        mot1_name = str(file['mot1'].attrs["Name"])
        mot2_raw = np.asarray(file['mot2'])
        mot2_name = str(file['mot2'].attrs["Name"])
        cmd = str(np.asarray(file['cmd'])).split(' ')
        if 'raw/I1' in file.keys():
            I1_raw = np.asarray(file['raw/I1'])
            i1flag = True

    # need to make a copy of mot1 and mot2 in case of sorting, which will also be used in snake etc.
    #   Not the most memory efficient way to do things, but these sizes will usually not be limiting and at least it's an easier fix
    for index, chnl in enumerate(keys):
        mot1 = mot1_raw.copy()
        mot2 = mot2_raw.copy()
        I0 = I0_raw.copy()
        tm = tm_raw.copy()
        if i1flag:
            I1 = I1_raw.copy()
        with h5py.File(h5file, 'r', locking=True) as file:
            ims0 = np.squeeze(np.asarray(file['fit/'+chnl+'/ims']))
        if len(ims0.shape) == 2 or len(ims0.shape) == 1:
            if len(ims0.shape) == 2:
                ims0 = ims0.reshape((ims0.shape[0], ims0.shape[1], 1))
                I0 = I0.reshape((np.squeeze(I0).shape[0], 1))
                if i1flag is True:
                    I1 = I1.reshape((np.squeeze(I1).shape[0], 1))
                tm = tm.reshape((np.squeeze(tm).shape[0], 1))
                mot1 = mot1.reshape((np.squeeze(mot1).shape[0], 1))
                mot2 = mot2.reshape((np.squeeze(mot2).shape[0], 1))
            else:
                ims0 = ims0.reshape((ims0.shape[0],1, 1))
                I0 = I0.reshape((I0.shape[0], 1))
                if i1flag is True:
                    I1 = I1.reshape((I0.shape[0], 1))
                tm = tm.reshape((tm.shape[0], 1))
                mot1 = mot1.reshape((mot1.shape[0], 1))
                mot2 = mot2.reshape((mot2.shape[0], 1))
            if I0.shape[0] > ims0.shape[1]:
                I0 = I0[0:ims0.shape[1],:]
                if i1flag is True:
                    I1 = I1[0:ims0.shape[1],:]
            if tm.shape[0] > ims0.shape[1]:
                tm = tm[0:ims0.shape[1],:]
            if mot1.shape[0] > ims0.shape[1]:
                mot1 = mot1[0:ims0.shape[1],:]            
            if mot2.shape[0] > ims0.shape[1]:
                mot2 = mot2[0:ims0.shape[1],:]
            if ims0.shape[1] > mot1.shape[0]:
                ims0 = ims0[:,0:mot1.shape[0],:]      
                I0 = I0[0:mot1.shape[0],:]
                if i1flag is True:
                    I1 = I1[0:mot1.shape[0],:]
                tm = tm[0:mot1.shape[0],:]
            except_list = ["b'timescanc", "b'dscan", "b'ascan", "timescanc", "dscan", "ascan", "c", "b'c"]
            if cmd[0] not in except_list:
                snake = True
                timetriggered=True  #if timetriggered is true one likely has more datapoints than fit on the regular grid, so have to interpolate in different way
    
        # Set snake etc to false when concerning hxrf data, as here the mot positions are measurement IDs rather than motor coordinates
        if mot1_name == "hxrf":
            snake = False
            timetriggered = False
    
        # set I0 value to normalise to
        if I0norm is None:
            normto = np.max(I0)
        else:
            normto = I0norm
        # set I0 indices that are 0, equal to normto (i.e. these points will not be normalised, as there was technically no beam)
        if np.nonzero(I0==0)[1].size != 0:
            for row, col in zip(np.nonzero(I0==0)[0], np.nonzero(I0==0)[1]):
                I0[row, col] = normto
    
        # for continuous scans, the mot1 position runs in snake-type fashion
        #   so we need to sort the positions line per line and adjust all other data accordingly
        # Usually sorting will have happened in xrf_fit_batch, but in some cases it is better to omit there and do it here
        #   for instance when certain scan lines need to be deleted
        if sort is True:
            for i in range(mot1[:,0].size):
                sort_id = np.argsort(mot1[i,:])
                ims0[:,i,:] = ims0[:,i,sort_id]
                mot1[i,:] = mot1[i,sort_id]
                mot2[i,:] = mot2[i,sort_id]
                I0[i,:] = I0[i,sort_id]
                tm[i,:] = tm[i,sort_id]
                if i1flag is True:
                    I1[i,:] = I1[i,sort_id]
            # To make sure (especially when merging scans) sort mot2 as well
            if mot2nosort is not True:
                for i in range(mot2[0,:].size):
                    sort_id = np.argsort(mot2[:,i])
                    ims0[:,:,i] = ims0[:,sort_id,i]
                    mot1[:,i] = mot1[sort_id,i]
                    mot2[:,i] = mot2[sort_id,i]
                    I0[:,i] = I0[sort_id,i]
                    if i1flag is True:
                        I1[:,i] = I1[sort_id,i]
                    tm[:,i] = tm[sort_id,i]
        # store data in any case so we can free some memory before working on the spectra
        with h5py.File(h5file, 'r+', locking=True) as file:
            if index == 0:
                try:
                    del file['raw/I0']
                    if i1flag is True:
                        del file['raw/I1']
                    del file['mot1']
                    del file['mot2']
                    del file['raw/acquisition_time']
                except Exception:
                    pass
                file.create_dataset('raw/I0', data=I0, compression='gzip', compression_opts=4)
                if i1flag is True:
                    file.create_dataset('raw/I1', data=I1, compression='gzip', compression_opts=4)
                dset = file.create_dataset('mot1', data=mot1, compression='gzip', compression_opts=4)
                dset.attrs['Name'] = mot1_name
                dset = file.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
                dset.attrs['Name'] = mot2_name
                file.create_dataset('raw/acquisition_time', data=tm, compression='gzip', compression_opts=4)
            try:
                del file['fit/'+chnl+'/ims']
            except Exception:
                pass
            file.create_dataset('fit/'+chnl+'/ims', data=ims0, compression='gzip', compression_opts=4)
        # make sure to free up some memory, we'll read in the relevant data again later
        del ims0, I0, I1, mot1, mot2, tm

        # Redo the procedure for the spectra, icr and ocr if requested
        if omitspectra is False:
            mot1 = mot1_raw.copy()
            mot2 = mot2_raw.copy()
            with h5py.File(h5file, 'r', locking=True) as file:
                spectra0 = np.squeeze(np.asarray(file['raw/'+chnl+'/spectra']))
                icr0 = np.squeeze(np.asarray(file['raw/'+chnl+'/icr']))
                ocr0 = np.squeeze(np.asarray(file['raw/'+chnl+'/ocr']))
            if len(spectra0.shape) == 2 or len(spectra0.shape) == 1:
                if len(spectra0.shape) == 2:
                    spectra0 = spectra0.reshape((spectra0.shape[0], 1, spectra0.shape[1]))
                    icr0 = icr0.reshape((np.squeeze(icr0).shape[0], 1))
                    ocr0 = ocr0.reshape((np.squeeze(ocr0).shape[0], 1))
                    mot1 = mot1.reshape((np.squeeze(mot1).shape[0], 1))
                    mot2 = mot2.reshape((np.squeeze(mot2).shape[0], 1))
                else:
                    spectra0 = spectra0.reshape((1, 1, spectra0.shape[0]))
                    icr0 = icr0.reshape((icr0.shape[0], 1))
                    ocr0 = ocr0.reshape((ocr0.shape[0], 1))
                    mot1 = mot1.reshape((mot1.shape[0], 1))
                    mot2 = mot2.reshape((mot2.shape[0], 1))
                if mot1.shape[0] > spectra0.shape[0]:
                    mot1 = mot1[0:spectra0.shape[0],:]            
                if mot2.shape[0] > spectra0.shape[0]:
                    mot2 = mot2[0:spectra0.shape[0],:]
                if spectra0.shape[0] > mot1.shape[0]:
                    spectra0 = spectra0[0:mot1.shape[0],:,:]      
                    icr0 = icr0[0:mot1.shape[0],:]
                    ocr0 = ocr0[0:mot1.shape[0],:]
        
            # for continuous scans, the mot1 position runs in snake-type fashion
            #   so we need to sort the positions line per line and adjust all other data accordingly
            # Usually sorting will have happened in xrf_fit_batch, but in some cases it is better to omit there and do it here
            #   for instance when certain scan lines need to be deleted
            if sort is True:
                for i in range(mot1[:,0].size):
                    sort_id = np.argsort(mot1[i,:])
                    spectra0[i,:,:] = spectra0[i,sort_id,:]
                    icr0[i,:] = icr0[i,sort_id]
                    ocr0[i,:] = ocr0[i,sort_id]
                    mot1[i,:] = mot1[i,sort_id]
                    mot2[i,:] = mot2[i,sort_id]
                # To make sure (especially when merging scans) sort mot2 as well
                if mot2nosort is not True:
                    for i in range(mot2[0,:].size):
                        sort_id = np.argsort(mot2[:,i])
                        icr0[:,i] = icr0[sort_id,i]
                        ocr0[:,i] = ocr0[sort_id,i]
                        spectra0[:,i,:] = spectra0[sort_id,i,:]
                        mot1[:,i] = mot1[sort_id,i]
                        mot2[:,i] = mot2[sort_id,i]
            # store data in any case so we can free up memory before starting the interpolation
            with h5py.File(h5file, 'r+', locking=True) as file:
                if index == 0:
                    try:
                        del file['raw/'+chnl+'/spectra']
                        del file['raw/'+chnl+'/icr']
                        del file['raw/'+chnl+'/ocr']
                    except Exception:
                        pass
                    file.create_dataset('raw/'+chnl+'/icr', data=icr0, compression='gzip', compression_opts=4)
                    file.create_dataset('raw/'+chnl+'/ocr', data=ocr0, compression='gzip', compression_opts=4)
                    file.create_dataset('raw/'+chnl+'/spectra', data=spectra0, compression='gzip', compression_opts=4)
            del icr0, ocr0, spectra0, mot1, mot2

        # read in data again for snake correction and motor position interpolation
        with h5py.File(h5file, 'r', locking=True) as file:
            mot1_raw = np.asarray(file['mot1'])
            mot1_name = str(file['mot1'].attrs["Name"])
            mot2_raw = np.asarray(file['mot2'])
            mot2_name = str(file['mot2'].attrs["Name"])
            ims0 = np.asarray(file['fit/'+chnl+'/ims'])
            names0 = np.asarray([n for n in file['fit/'+chnl+'/names']])
            sum_fit0 = np.asarray(file['fit/'+chnl+'/sum/int'])
            sum_bkg0 = np.asarray(file['fit/'+chnl+'/sum/bkg'])
            I0 =  np.asarray(file['raw/I0'])
            tm = np.asarray(file['raw/acquisition_time'])
            if i1flag is True:
                I1 = np.asarray(file['raw/I1'])
        mot1 = mot1_raw.copy()
        mot2 = mot2_raw.copy()
        if len(ims0.shape) == 2 or len(ims0.shape) == 1:
            if len(ims0.shape) == 2:
                ims0 = ims0.reshape((ims0.shape[0], ims0.shape[1], 1))
                I0 = I0.reshape((np.squeeze(I0).shape[0], 1))
                if i1flag is True:
                    I1 = I1.reshape((np.squeeze(I1).shape[0], 1))
                tm = tm.reshape((np.squeeze(tm).shape[0], 1))
                mot1 = mot1.reshape((np.squeeze(mot1).shape[0], 1))
                mot2 = mot2.reshape((np.squeeze(mot2).shape[0], 1))
            else:
                ims0 = ims0.reshape((ims0.shape[0],1, 1))
                I0 = I0.reshape((I0.shape[0], 1))
                if i1flag is True:
                    I1 = I1.reshape((I0.shape[0], 1))
                tm = tm.reshape((tm.shape[0], 1))
                mot1 = mot1.reshape((mot1.shape[0], 1))
                mot2 = mot2.reshape((mot2.shape[0], 1))

        # correct I0
        ims0[ims0 < 0] = 0.
        sum_fit0[sum_fit0 < 0] = 0.
        sum_bkg0[sum_bkg0 < 0] = 0.
        ims0_err = np.nan_to_num(np.sqrt(ims0)/ims0)
        sum_fit0_err = np.nan_to_num(np.sqrt(sum_fit0[:]+2*sum_bkg0[:])/sum_fit0[:])
        sum_bkg0_err = np.nan_to_num(np.sqrt(sum_bkg0[:])/sum_bkg0[:])
        for i in range(0, ims0.shape[0]):
            if tmnorm is True:
                ims0[i,:,:] = ims0[i,:,:]/(I0*tm) * normto
            else:
                ims0[i,:,:] = ims0[i,:,:]/(I0) * normto
        if tmnorm is True:
            sum_fit0 = sum_fit0/(np.sum(I0*tm)) * normto
            sum_bkg0 = sum_bkg0/(np.sum(I0*tm)) * normto
        else:
            sum_fit0 = (sum_fit0/np.sum(I0)) * normto
            sum_bkg0 = (sum_bkg0/np.sum(I0)) * normto
        ims0[np.isnan(ims0)] = 0.           
    
        # if this is snakescan, interpolate ims array for motor positions so images look nice
        #   this assumes that mot1 was the continuously moving motor
        if snake is True:
            print("Interpolating image for motor positions...", end=" ")
            if timetriggered is False:
                if halfpixshift is True:
                    pos_low = np.min(mot1[:,0])
                    pos_high = np.max(mot1[:,0])
                    for i in range(0, mot1[:,0].size): #correct for half a pixel shift
                        if mot1[i,0] <= np.average((pos_high,pos_low)):
                            mot1[i,:] += abs(mot1[i,1]-mot1[i,0])/2.
                        else:
                            mot1[i,:] -= abs(mot1[i,1]-mot1[i,0])/2.
                mot1_pos = np.average(mot1, axis=0) #mot1[0,:]
                mot2_pos = np.average(mot2, axis=1) #mot2[:,0]
                ims0_tmp = np.zeros((ims0.shape[0], ims0.shape[1], ims0.shape[2]))
                ims0_err_tmp = np.zeros((ims0.shape[0], ims0.shape[1], ims0.shape[2]))
            if timetriggered is True:
                if halfpixshift is True:
                    # correct positions for half pixel shift
                    mot1[0:mot1.size-1, 0] = mot1[0:mot1.size-1, 0] + np.diff(mot1[:,0])/2.
                # based on cmd determine regular grid positions
                if cmd[0]=="b'cdmeshs":
                    xdim = np.floor(np.abs(float(cmd[2])-float(cmd[3]))/float(cmd[4]))
                    ydim = np.floor(np.abs(float(cmd[6])-float(cmd[7]))/float(cmd[8]))+1
                    mot1_pos = np.linspace(float(cmd[2]), float(cmd[3]), num=int(xdim))
                    mot2_pos = np.linspace(float(cmd[6]), float(cmd[7]), num=int(ydim)) 
                else:
                    mot1_pos = np.linspace(float(cmd[2]), float(cmd[3]), num=int(cmd[4]))
                    mot2_pos = np.linspace(float(cmd[6]), float(cmd[7]), num=int(cmd[8])) 
                if cmd[0] == "b'cdmesh" or cmd[0] == "b'cdmeshs":
                    mot1_pos = mot1_pos - (mot1_pos[0] - mot1[0,0])
                    mot2_pos = mot2_pos - (mot2_pos[0] - mot2[0,0])
                ims0_tmp = np.zeros((ims0.shape[0], mot2_pos.shape[0], mot1_pos.shape[0]))
                ims0_err_tmp = np.zeros((ims0.shape[0], mot2_pos.shape[0], mot1_pos.shape[0]))
            # interpolate to the regular grid motor positions
            mot1_tmp, mot2_tmp = np.mgrid[mot1_pos[0]:mot1_pos[-1]:complex(mot1_pos.size),
                    mot2_pos[0]:mot2_pos[-1]:complex(mot2_pos.size)]
            x = mot1.ravel()
            y = mot2.ravel()
    
    
            for i in range(names0.size):
                values = ims0[i,:,:].ravel()
                ims0_tmp[i,:,:] = griddata((x, y), values, (mot1_tmp, mot2_tmp), method=interpol_method).T
                ims0_err_tmp[i,:,:] = griddata((x, y), ims0_err[i,:,:].ravel(), (mot1_tmp, mot2_tmp), method=interpol_method).T
            ims0 = np.nan_to_num(ims0_tmp)
            ims0_err = np.nan_to_num(ims0_err_tmp)*ims0
            print("Done")
            with h5py.File(h5file, 'r+', locking=True) as file:
                if index == 0:
                    I0 = griddata((x, y), I0.ravel(), (mot1_tmp, mot2_tmp), method=interpol_method).T
                    if i1flag is True:
                        I1 = griddata((x, y), I1.ravel(), (mot1_tmp, mot2_tmp), method=interpol_method).T
                    tm = griddata((x, y), tm.ravel(), (mot1_tmp, mot2_tmp), method=interpol_method).T
                    try:
                        del file['mot1']
                        del file['mot2']
                        del file['raw/I0']
                        del file['raw/acquisition_time']
                        if i1flag is True:
                            del file['raw/I1']
                    except Exception:
                        pass
                    file.create_dataset('raw/I0', data=I0, compression='gzip', compression_opts=4)
                    if i1flag is True:
                        file.create_dataset('raw/I1', data=I1, compression='gzip', compression_opts=4)
                    file.create_dataset('raw/acquisition_time', data=tm, compression='gzip', compression_opts=4)
                    dset = file.create_dataset('mot1', data=mot1_tmp.T, compression='gzip', compression_opts=4)
                    dset.attrs['Name'] = mot1_name
                    dset = file.create_dataset('mot2', data=mot2_tmp.T, compression='gzip', compression_opts=4)
                    dset.attrs['Name'] = mot2_name
            del ims0_tmp, ims0_err_tmp
            
        # save normalised data
        print("     Writing...", end=" ")
        with h5py.File(h5file, 'r+', locking=True) as file:
            try:
                del file['norm/I0']
                del file['norm/'+chnl]
            except KeyError:
                pass
            file.create_dataset('norm/I0', data=normto)
            file.create_dataset('norm/'+chnl+'/ims', data=ims0, compression='gzip', compression_opts=4)
            file.create_dataset('norm/'+chnl+'/ims_stddev', data=ims0_err, compression='gzip', compression_opts=4)
            file.create_dataset('norm/'+chnl+'/names', data=names0)
            file.create_dataset('norm/'+chnl+'/sum/int', data=sum_fit0, compression='gzip', compression_opts=4)
            file.create_dataset('norm/'+chnl+'/sum/bkg', data=sum_bkg0, compression='gzip', compression_opts=4)
            file.create_dataset('norm/'+chnl+'/sum/int_stddev', data=sum_fit0*sum_fit0_err, compression='gzip', compression_opts=4)
            file.create_dataset('norm/'+chnl+'/sum/bkg_stddev', data=sum_bkg0*sum_bkg0_err, compression='gzip', compression_opts=4)
            dset = file['norm']
            dset.attrs["LastUpdated"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if tmnorm is True:
                dset.attrs["TmNorm"] = "True"
            else:
                dset.attrs["TmNorm"] = "False"
                
        # Reprocess some stuff to also interpolate the spectra, icr and ocr datasets
        #   also free up some memory that we shouldn't need anymore
        del ims0, ims0_err, names0, sum_fit0, sum_bkg0, sum_fit0_err, sum_bkg0_err

        if omitspectra is False:
            # if this is snakescan, interpolate ims array for motor positions so images look nice
            #   this assumes that mot1 was the continuously moving motor
            if snake is True:
                print("Interpolating spectra for motor positions...", end=" ")
                with h5py.File(h5file, 'a', locking=True) as file:
                    icr0 = np.squeeze(np.asarray(file['raw/'+chnl+'/icr']))
                    ocr0 = np.squeeze(np.asarray(file['raw/'+chnl+'/ocr']))
                    icr0 = griddata((x, y), icr0.ravel(), (mot1_tmp, mot2_tmp), method=interpol_method).T
                    ocr0 = griddata((x, y), ocr0.ravel(), (mot1_tmp, mot2_tmp), method=interpol_method).T
                    try:
                        del file['raw/'+chnl+'/icr']
                        del file['raw/'+chnl+'/ocr']
                    except Exception:
                        pass
                    file.create_dataset('raw/'+chnl+'/icr', data=icr0, compression='gzip', compression_opts=4)
                    file.create_dataset('raw/'+chnl+'/ocr', data=ocr0, compression='gzip', compression_opts=4)
                    if lowmemory is True:
                        for i in range(file['raw/'+chnl+'/spectra'].shape[2]):
                            file['raw/'+chnl+'/spectra'][:,:,i] = griddata((x, y), file['raw/'+chnl+'/spectra'][:,:,i].ravel(), (mot1_tmp, mot2_tmp), method=interpol_method).T
                    else:
                        spectra0 = np.squeeze(np.asarray(file['raw/'+chnl+'/spectra']))
                        for i in range(file['raw/'+chnl+'/spectra'].shape[2]):
                            spectra0[:,:,i] = griddata((x, y), spectra0[:,:,i].ravel(), (mot1_tmp, mot2_tmp), method=interpol_method).T
                        try:
                            del file['raw/'+chnl+'/spectra']
                        except Exception:
                            pass
                        file.create_dataset('raw/'+chnl+'/spectra', data=spectra0, compression='gzip', compression_opts=4)
                    print("Done")

        print("Done")
    
##############################################################################
def  fit_xrf_batch(h5file, cfgfile, channel=None, standard=None, ncores=None, verbose=None, scatroi=False):
    """
    Fit a batch of xrf spectra using the PyMca fitting routines. A PyMca config file should be supplied.
    The cfg file should use the SNIP background subtraction method. Others will fail as considered 'too slow' by the PyMca fast linear fit routine itself.
    Additionally, using the standard option also integrates the individual spectra, not only the sum spectrum, without fast linear fit. This can take a long time!!
   

    Parameters
    ----------
    h5file : string
        File directory path to the H5 file containing the data.
    cfgfile : string
        File path to the PyMca-type CFG configuration file containing the fitting parameters.
    channel: NoneType, optional
        Set channel to a list of channel names to which the fitting should be limited. e.g. channel=['channel01'] 
    standard : NoneType, optional
        If not a NoneType (e.g. string) then all spectra are integrated separately without using the fast linear fit procedure. The default is None.
    ncores : Integer, optional
        The amount of cores over which the multiprocessing package should split the task. Values of -1, 0 and None allow the system to use all available cores, minus 1. The default is None.
    verbose : Boolean, optional
        If not None, the PyMca fit returns errors encountered during the procedure. The default is None.
    scatroi: Boolean, optional
        If True, the Compton and Rayleigh intensity will be determined as integrated region of interest values instead of the fitted result as obtained through the PyMca fit.
        The peak width is determined from 

    Returns
    -------
    None.

    """
    # perhaps channel00 and channel01 need different cfg files. Allow for tuple or array in this case.
    cfgfile = np.asarray(cfgfile)
        
    # let's read the h5file structure and launch our fit.
    file = h5py.File(h5file, 'r', locking=True)
    if channel is None:
        keys = [key for key in file['raw'].keys() if 'channel' in key]
    else:
        keys = channel
    for index, chnl in enumerate(keys):
        if cfgfile.size == 1:
            cfg = str(cfgfile)
        else:
            cfg = str(cfgfile[index])
        with h5py.File(h5file, 'r', locking=True) as file:
            spectra0 = np.asarray(file['raw/'+chnl+'/spectra'])
            sumspec0 = np.asarray(file['raw/'+chnl+'/sumspec'])
            icr0 = np.asarray(file['raw/'+chnl+'/icr'])
            ocr0 = np.asarray(file['raw/'+chnl+'/ocr'])
        spec0_shape = spectra0.shape
        if len(spec0_shape) == 2:
            spectra0 = np.asarray(spectra0).reshape((spec0_shape[0], 1, spec0_shape[1]))
            icr0 = np.asarray(icr0).reshape((icr0.shape[0], 1))
            ocr0 = np.asarray(ocr0).reshape((ocr0.shape[0], 1))
        nchannels0 = spectra0.shape[2]
    
        # find indices where icr=0 and put those values = icr
        if np.nonzero(ocr0==0)[0].size != 0:
            ocr0[np.nonzero(ocr0==0)[0]] = icr0[np.nonzero(ocr0==0)[0]]
            if np.nonzero(icr0==0)[0].size != 0:
                icr0[np.nonzero(icr0==0)[0]] = 1 #if icr0 equal to 0, let's set to 1
                ocr0[np.nonzero(ocr0==0)[0]] = icr0[np.nonzero(ocr0==0)[0]] # let's run this again. Seems redundant, but icr could have changed.
    
        # work PyMca's magic!
        t0 = time.time()
        print("Initiating fit of <"+h5file+"> "+chnl+" using model(s) <"+cfg+">...", end=" ")
        n_spectra = round(spectra0.size/nchannels0, 0)
        if standard is None:
            # read and set PyMca configuration for channel00
            fastfit = FastXRFLinearFit.FastXRFLinearFit()
            try: 
                fastfit.setFitConfigurationFile(cfg)
            except Exception:
                print("-----------------------------------------------------------------------------")
                print("Error: %s is not a valid PyMca configuration file." % cfg)
                print("-----------------------------------------------------------------------------")
            fitresults0 = fastfit.fitMultipleSpectra(x=range(0,nchannels0), y=np.asarray(spectra0[:,:,:]), ysum=sumspec0)
            #fit sumspec
            config = ConfigDict.ConfigDict()
            config.read(cfg)
            config['fit']['use_limit'] = 1 # make sure the limits of the configuration will be taken
            mcafit = ClassMcaTheory.ClassMcaTheory()
            mcafit.configure(config)
            mcafit.setData(range(0,nchannels0), sumspec0)
            mcafit.estimate()
            fitresult0_sum, result0_sum = mcafit.startfit(digest=1)
            sum_fit0 = [result0_sum[peak]["fitarea"] for peak in result0_sum["groups"]]
            sum_bkg0 = [result0_sum[peak]["statistics"]-result0_sum[peak]["fitarea"] for peak in result0_sum["groups"]]
    
            print("Done")
            # actual fit results are contained in fitresults['parameters']
            #   labels of the elements are fitresults.labels("parameters"), first # are 'A0, A1, A2' of polynomial background, which we don't need
            peak_int0 = np.asarray(fitresults0['parameters'])
            names0 = fitresults0.labels("parameters")
            names0 = [n.replace('Scatter Peak000', 'Rayl') for n in names0]
            names0 = np.asarray([n.replace('Scatter Compton000', 'Compt') for n in names0])
            cutid0 = 0
            for i in range(names0.size):
                if names0[i] == 'A'+str(i):
                    cutid0 = i+1
            del fitresults0
        else: #standard is not None; this is a srm spectrum and as such we would like to obtain the background values.
            # channel00
            config = ConfigDict.ConfigDict()
            config.read(cfg)
            config['fit']['use_limit'] = 1 # make sure the limits of the configuration will be taken
            mcafit = ClassMcaTheory.ClassMcaTheory()
            mcafit.configure(config)
            if ncores is None or ncores == -1 or ncores == 0:
                ncores = multiprocessing.cpu_count()-1
            spec_chansum = np.sum(spectra0, axis=2)
            spec2fit_id = np.asarray(np.where(spec_chansum.ravel() > 0.)).squeeze()
            spec2fit = np.asarray(spectra0).reshape((spectra0.shape[0]*spectra0.shape[1], spectra0.shape[2]))[spec2fit_id,:]
            if spectra0.shape[0]*spectra0.shape[1] > 1:
                print("Using "+str(ncores)+" cores...", end=" ")
                pool = multiprocessing.Pool(processes=ncores)
                results, groups = zip(*pool.map(partial(Pymca_fit, mcafit=mcafit, verbose=verbose), spec2fit))
                results = list(results)
                groups = list(groups)
                if groups[0] is None: #first element could be None, so let's search for first not-None item.
                    for i in range(0, np.asarray(groups, dtype='object').shape[0]):
                        if groups[i] is not None:
                            groups[0] = groups[i]
                            break
                none_id = [i for i, x in enumerate(results) if x is None]
                if none_id != []:
                    for i in range(0, np.asarray(none_id).size):
                        results[none_id[i]] = [0]*np.asarray(groups[0]).shape[0] # set None to 0 values
                peak_int0 = np.zeros((spectra0.shape[0]*spectra0.shape[1], np.asarray(groups[0]).shape[0]))
                peak_int0[spec2fit_id,:] = np.asarray(results).reshape((spec2fit_id.size, np.asarray(groups[0]).shape[0]))
                peak_int0 = np.moveaxis(peak_int0.reshape((spectra0.shape[0], spectra0.shape[1], np.asarray(groups[0]).shape[0])),-1,0)
                peak_int0[np.isnan(peak_int0)] = 0.
                pool.close()
                pool.join()
            else:
                mcafit.setData(range(0,nchannels0), spectra0[0,0,:])
                mcafit.estimate()
                fitresult0, result0 = mcafit.startfit(digest=1)
                peak_int0 = np.asarray([result0[peak]["fitarea"] for peak in result0["groups"]])
                peak_int0 = peak_int0.reshape((peak_int0.shape[0],1,1))
                
            
            #fit sumspec
            mcafit.setData(range(0,nchannels0), sumspec0)
            mcafit.estimate()
            fitresult0_sum, result0_sum = mcafit.startfit(digest=1)
            names0 = result0_sum["groups"]
            names0 = [n.replace('Scatter Peak000', 'Rayl') for n in result0_sum["groups"]]
            names0 = np.asarray([n.replace('Scatter Compton000', 'Compt') for n in names0])
            cutid0 = 0
            for i in range(names0.size):
                if names0[i] == 'A'+str(i):
                    cutid0 = i+1
            sum_fit0 = [result0_sum[peak]["fitarea"] for peak in result0_sum["groups"]]
            sum_bkg0 = [result0_sum[peak]["statistics"]-result0_sum[peak]["fitarea"] for peak in result0_sum["groups"]]

        ims0 = peak_int0[cutid0:,:,:]
    
        if scatroi is True:
            # new cte and gain obtained from previous fit
            cte = result0_sum["fittedpar"][result0_sum["parameters"].index("Zero")]
            gain = result0_sum["fittedpar"][result0_sum["parameters"].index("Gain")]
            # determine Rayleigh and Compton min and max channel nrs
            #   in principle this integrated roi is also in result0_sum['statistics'] for each peak, but calculating it ourselves feels more clean
            #       and is more convenient as we don't have to change Pymca_fit() then
            if type(config['fit']['energy']) is type(float):
                raylE = config['fit']['energy']
            else:
                raylE = config['fit']['energy'][0] #if list of energies is provided, only take first element
            comptE = raylE / (1.+(raylE/511.)*(1.-np.cos(config['attenuators']['Matrix'][-1]/180.*np.pi)))
            rayl_min = np.round(((raylE - result0_sum['Scatter Peak000']['Scatter 000']['fwhm']/2)-cte)/gain).astype(int)
            rayl_max = np.round(((raylE + result0_sum['Scatter Peak000']['Scatter 000']['fwhm']/2)-cte)/gain).astype(int)
            compt_min = np.round(((comptE - result0_sum['Scatter Compton000']['Scatter 000']['fwhm']/2)-cte)/gain).astype(int)
            compt_max = np.round(((comptE + result0_sum['Scatter Compton000']['Scatter 000']['fwhm']/2)-cte)/gain).astype(int)
            # now integrate the results
            raylid = np.where(names0 =='Rayl')[0][0]
            comptid = np.where(names0 =='Compt')[0][0]
            sum_bkg0[raylid] = 0. #background intensities are now0 as the full roi intensity is contained within the fitted peak...
            sum_bkg0[comptid] = 0.
            sum_fit0[raylid] = np.sum(sumspec0[rayl_min:rayl_max])
            sum_fit0[comptid] = np.sum(sumspec0[compt_min:compt_max])
            ims0[raylid,:,:] = np.sum(spectra0[:,:,rayl_min:rayl_max], axis=2)
            ims0[comptid,:,:] = np.sum(spectra0[:,:,compt_min:compt_max], axis=2)
            
        del spectra0, result0_sum, peak_int0
        print("Fit finished after "+str(time.time()-t0)+" seconds for "+str(n_spectra)+" spectra.")
    
        # correct for deadtime  
        # check if icr/ocr values are appropriate!
        if np.average(ocr0/icr0) > 1.:
            print("ERROR: "+chnl+" ocr/icr is larger than 1!")
        if icr0.shape[0] > ims0.shape[1]:
            icr0 = icr0[0:ims0.shape[1],:]
            ocr0 = ocr0[0:ims0.shape[1],:]
        for i in range(names0.size):
            ims0[i,:,:] = ims0[i,:,:] * icr0/ocr0
        sum_fit0 = np.asarray(sum_fit0)*np.sum(icr0)/np.sum(ocr0)
        sum_bkg0 = np.asarray(sum_bkg0)*np.sum(icr0)/np.sum(ocr0)
        if len(spec0_shape) == 2:
            ims0 = np.squeeze(ims0)
    
        # save the fitted data
        print("Writing fit data to "+h5file+"...", end=" ")
        with h5py.File(h5file, 'r+', locking=True) as file:
            try:
                del file['fit/'+chnl+'/ims']
                del file['fit/'+chnl+'/names']
                del file['fit/'+chnl+'/cfg']
                del file['fit/'+chnl+'/sum/int']
                del file['fit/'+chnl+'/sum/bkg']
            except Exception:
                pass
            file.create_dataset('fit/'+chnl+'/ims', data=ims0, compression='gzip', compression_opts=4)
            file.create_dataset('fit/'+chnl+'/names', data=[n.encode('utf8') for n in names0[cutid0:]])
            file.create_dataset('fit/'+chnl+'/cfg', data=cfg)
            file.create_dataset('fit/'+chnl+'/sum/int', data=sum_fit0, compression='gzip', compression_opts=4)
            file.create_dataset('fit/'+chnl+'/sum/bkg', data=sum_bkg0, compression='gzip', compression_opts=4)
            dset = file['fit']
            dset.attrs["LastUpdated"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            dset.attrs["ScatROI"] = scatroi
    print('Done')

##############################################################################
def Pymca_fit(spectra, mcafit, verbose=None):
    """
    Function to perform XRF fitting using the PyMca algorithm over an array of spectra.

    Parameters
    ----------
    spectra : float array
        Array of spectra of shape [N×M] where N is the number of spectra and M the number of channels in each spectrum.
    mcafit : ClassMcaTheory
        PyMca class containing the fit configuration parameters.
    verbose : NoneType, optional
        If not None, errors encountered during the fit are printed in the python console window. The default is None.

    Returns
    -------
    result : float array
        Array containing the spectral integrated intensities for each line group.
    groups : string array
        Array containing the line group labels.

    """

    mcafit.setData(range(0,spectra.shape[0]), spectra)
    try:
        mcafit.estimate()
        fitresult, result = mcafit.startfit(digest=1)
        groups = result["groups"]
        result = [result[peak]["fitarea"] for peak in result["groups"]]
    except Exception as ex:
        if verbose is not None:
            print('Error in mcafit.estimate()', ex)
        result = None
        groups = None
        
    return result, groups

##############################################################################
def Xproc_rspe(spefile):
    # extract the spectrum from the SPE file
    with open(spefile, 'r') as f:
        lines = f.readlines()
    
    data_id = lines.index("$DATA:\n")
    
    header = lines[0:data_id]
    spectrum = np.asarray([item for sublist in [np.fromstring(line, sep=' ') for line in lines[data_id+2:]] for item in sublist])
    return spectrum, header

##############################################################################
def ConvMxrfSpe(speprefix, outfile, mot1_name='X', mot2_name='Y'):
    """
    Read SPE files generated by the XMI MicroXRF instrument and restructure as a H5 file for further processing
    Example: ConvMxrfSpe('/data/eagle/folder/a_', 'a_merge.h5')

    Parameters
    ----------
    speprefix : String
        File path prefix to the SPE files that should be converted to a H5 file.
        The function uses this prefix to identify all SPE files, appending it with '*.spe'
    outfile : String
        File path name for the converted H5 file.
    mot1_name : string, optional
        Motor 1 identifier within the SPE file. The default is 'X'.
    mot2_name : String, optional
        Motor 2 identifier within the SPE file. The default is 'Y'.

    Returns
    -------
    None.

    """
    import glob
    
    spefiles = glob.glob(speprefix+'*.spe')

    xid = []
    yid = []
    for file in spefiles:
        temp = file.replace('\\','/').split(speprefix)[-1].split('.')[0].split('_')
        xid.append(int(temp[1]))
        yid.append(int(temp[0]))
    xmax = np.max(np.asarray(xid))
    ymax = np.max(np.asarray(yid))

    spectra = []
    ocr = [] 
    icr = [] 
    i0 = []
    tm = []
    mot1 = []
    mot2 = []

    motname_ids = ['X', 'Y', 'Z', 'T']
    if mot1_name not in motname_ids or mot2_name not in motname_ids:
        raise ValueError("ValueError: mot1_name or mot2_name not known to exist in Mxrf SPE files: {} {}".format(mot1_name, mot2_name))

    for index, file in enumerate(sorted(spefiles)):
        spe, head = Xproc_rspe(file)
        temp = file.replace('\\','/').split(speprefix)[-1].split('.')[0].split('_')
        xid = int(temp[1])-1
        yid = int(temp[0])-1
        if index == 0:
            spectra = np.zeros((xmax, ymax, spe.shape[0]))
            ocr = np.zeros((xmax, ymax)) 
            icr = np.zeros((xmax, ymax)) 
            i0 = np.zeros((xmax, ymax))+1. #unfortunately, no i0 signal is registered
            tm = np.zeros((xmax, ymax))
            mot1 = np.zeros((xmax, ymax))
            mot2 = np.zeros((xmax, ymax))

        spectra[xid,yid,:] = spe

        tm[xid,yid] = float(np.fromstring(head[head.index('$MEAS_TIM:\n')+1], sep=' ')[1]) #store realtime as icr and ocr are known, so deadtime correction is done at later stage
        ocr[xid,yid] = float(np.fromstring(head[head.index('$ICR_&_OCR (normalised to 1s)\n')+1], sep=' ')[1])
        icr[xid,yid] = float(np.fromstring(head[head.index('$ICR_&_OCR (normalised to 1s)\n')+1], sep=' ')[0])

        mot1[xid,yid] = float(np.fromstring(head[head.index('XYZT\n')+1], sep=' ')[motname_ids.index(mot1_name)])
        mot2[xid,yid] = float(np.fromstring(head[head.index('XYZT\n')+1], sep=' ')[motname_ids.index(mot2_name)])


    sumspec = np.sum(spectra[:], axis=(0,1))
    maxspec = np.zeros(sumspec.shape[0])
    for i in range(sumspec.shape[0]):
        maxspec[i] = spectra[:,:,i].max()
    i1 = np.zeros(i0.shape)

    outfile = '/'.join(speprefix.split('/')[:-1])+'/'+outfile
    print("Writing converted file: "+outfile+"...", end=" ")
    with h5py.File(outfile, 'w', locking=True) as f:
        f.create_dataset('cmd', data='scan XMI MicroXRF')
        f.create_dataset('raw/channel00/spectra', data=spectra, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/icr', data=ocr, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/ocr', data=ocr, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/sumspec', data=np.squeeze(sumspec), compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/maxspec', data=np.squeeze(maxspec), compression='gzip', compression_opts=4)
        f.create_dataset('raw/I0', data=i0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/I1', data=i1, compression='gzip', compression_opts=4)
        f.create_dataset('raw/acquisition_time', data=tm, compression='gzip', compression_opts=4)
        dset = f.create_dataset('mot1', data=mot1, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot1_name
        dset = f.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot2_name
    print("Done")


##############################################################################
def ConvEdaxSpc(spcprefix, outfile, scandim, coords=[0,0,1,1]):
    """
    Read the EDAX EagleIII SPC files and restructure as H5 for further processing
    Example: ConvEdaxSpc('/data/eagle/folder/a', 'a_merge.h5', (30,1), coords=[22.3, 17, 0.05, 0.])

    Parameters
    ----------
    spcprefix : String
        File path prefix to the SPC files that should be converted to a H5 file.
        The function uses this prefix to identify all SPC files, appending it with '*.spc'
    outfile : String
        File path name for the converted H5 file.
    scandim : list of length 2
        Dimensions of the scan [X, Y].
    coords : list of length 4, optional
        If coords is provided, motor positions are calculated. coords=[Xstart, Ystart, Xincr, Yincr]
        Where Xstart and Ystart are the position coordinates of the first measurement (a11.SPC)
        and Xincr and Yincr are the step sizes of X and Y motors, in mm. The software assumes that
        X is the 'fast moving' motor, i.e. makes most steps during the scan and that no snake-pattern scans are performed. 
        The default is [0,0,1,1].

    Raises
    ------
    ValueError
        A ValueError is raised when the number of found SPC files does not match the provided scandim dimensions.

    Returns
    -------
    None.

    """
    import glob
    
    spcfiles = glob.glob(spcprefix+'*.spc')
    # check if found files number matches provided scan dimension
    if len(spcfiles) != scandim[0]*scandim[1]:
        raise ValueError("ValueError: scan dimension ", scandim, " does not match number of SPC files: ", len(spcfiles))
    
    spectra = []
    ocr = [] #icr and ocr are the same for edax eagle data as we acquire for certain set livetime
    i0 = []
    tm = []
    mot1 = []
    mot2 = []
    
    x,y = 0, 0
    for file in sorted(spcfiles, key=len): #key=len is needed to assure appropriate sorting due to difference in numerical and alphabetical sorting
        s = Spc(file)      
        i0.append(float(s.rv["Current"]))
        tm.append(float(s.rv["LiveTime"]))
        spectra.append(s.rv["Data"].astype(float))
        ocr.append(float(s.rv["OCR"]))

        mot1.append(coords[0]+x*coords[2])
        mot2.append(coords[1]+y*coords[3])
        x += 1
        if x >= scandim[0]:
            x = 0
            y += 1

    # reshape the arrays to appropriate scan dimensions
    spectra = np.asarray(spectra).reshape((scandim[1], scandim[0], len(spectra[0])))
    ocr = np.asarray(ocr).reshape((scandim[1], scandim[0]))
    i0 = np.asarray(i0).reshape((scandim[1], scandim[0]))
    tm = np.asarray(tm).reshape((scandim[1], scandim[0]))
    mot1 = np.asarray(mot1).reshape((scandim[1], scandim[0]))
    mot2 = np.asarray(mot2).reshape((scandim[1], scandim[0]))
    sumspec = np.sum(spectra[:], axis=(0,1))
    maxspec = np.zeros(sumspec.shape[0])
    for i in range(sumspec.shape[0]):
        maxspec[i] = spectra[:,:,i].max()
    i1 = np.zeros(i0.shape)

    outfile = '/'.join(spcprefix.split('/')[:-1])+'/'+outfile
    if not outfile.endswith('.h5'):
        outfile += '.h5'
    print("Writing converted file: "+outfile+"...", end=" ")
    with h5py.File(outfile, 'w', locking=True) as f:
        f.create_dataset('cmd', data='ascan EDAX EagleIII')
        f.create_dataset('raw/channel00/spectra', data=spectra, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/icr', data=ocr, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/ocr', data=ocr, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/sumspec', data=np.squeeze(sumspec), compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/maxspec', data=np.squeeze(maxspec), compression='gzip', compression_opts=4)
        f.create_dataset('raw/I0', data=i0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/I1', data=i1, compression='gzip', compression_opts=4)
        f.create_dataset('raw/acquisition_time', data=tm, compression='gzip', compression_opts=4)
        dset = f.create_dataset('mot1', data=mot1, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = 'X'
        dset = f.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = 'Y'
    print("Done")

##############################################################################
def ConvMalPanMPS(mpsfile):
    """
    Read the Malvern Panalytical EPSILON 3XL MPS files and restructure as H5 for further processing
    

    Parameters
    ----------
    mpsfile : String
        File path to the MPS file to be converted. If a list is provided, all files are concatenated in a single H5 file.

    Returns
    -------
    None.

    """
    
    if type(mpsfile) != type(list()):
        mpsfile = [mpsfile]
        
 
    spectra0 = []
    tm0 = []
    ocr0 = []
    i0 = []
    i1 = []
    mot1 = []
    mot2 = []
    ydim = len(mpsfile)
    xdim = 1
 
    for mps in mpsfile:
        with open(mps, 'r') as file:
            data = file.readlines()    
        
        measurement_id = os.path.splitext(mps)[0]
        nchannels = int(data[[i for i,n in enumerate(data) if 'NrOfChannels' in n][0]].split(':')[1])
        
        spectra0.append(np.asarray(data[-1*nchannels:], dtype=float))
        tm0.append(float('.'.join(data[[i for i,n in enumerate(data) if 'LiveTime' in n][0]].split(':')[1].split(','))))
        ocr0.append(np.sum(spectra0[-1])) #icr and ocr are the same for edax eagle data as we acquire for certain set livetime
        i0.append(float('.'.join(data[[i for i,n in enumerate(data) if 'uA' in n][0]].split(':')[1].split(','))))
        i1.append(float('.'.join(data[[i for i,n in enumerate(data) if 'NormCurCounts' in n][0]].split(':')[1].split(','))))
        mot1.append(''.join(data[[i for i,n in enumerate(data) if 'SampleIdent' in n][0]].split(':')[1:]))
        mot2.append(mps)

    spectra0 = np.asarray(spectra0)
    spectra0 = spectra0.reshape((ydim, xdim, spectra0.shape[1]))
    ocr0 = np.asarray(ocr0).reshape((ydim, xdim))
    i0 = np.asarray(i0).reshape((ydim, xdim))
    i1 = np.asarray(i1).reshape((ydim, xdim))
    tm0 = np.asarray(tm0).reshape((ydim, xdim))
    mot1 = np.asarray([n.encode('utf8') for n in mot1]).reshape((ydim, xdim))
    mot2 = np.asarray([n.encode('utf8') for n in mot2]).reshape((ydim, xdim))

    sumspec0 = np.sum(spectra0[:], axis=(0,1))
    maxspec0 = np.zeros(sumspec0.shape[0])
    for i in range(sumspec0.shape[0]):
        maxspec0[i] = spectra0[:,:,i].max()


    # Hooray! We read all the information! Let's write it to a separate file
    print("Writing merged file: "+measurement_id+"_merge.h5...", end=" ")
    with h5py.File(measurement_id+"_merge.h5", 'w', locking=True) as f:
        f.create_dataset('cmd', data='Measurement PANalaytical Epsilon 3XL')
        f.create_dataset('raw/channel00/spectra', data=spectra0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/icr', data=ocr0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/ocr', data=ocr0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/sumspec', data=sumspec0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/maxspec', data=maxspec0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/I0', data=i0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/I1', data=i1, compression='gzip', compression_opts=4)
        f.create_dataset('raw/acquisition_time', data=tm0, compression='gzip', compression_opts=4)
        dset = f.create_dataset('mot1', data=mot1, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = 'hxrf'
        dset = f.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = 'hxrf'
    print("Done")
        
##############################################################################
def ConvBrukerPdz(pdzfiles, outfile):
    """
    Read the Bruker handheld PDZ files and restructure as H5 for further processing
    

    Parameters
    ----------
    pdzfiles : String or list
        File path(s) to the PDZ file(s) to be converted. Note that all files should be of the same measurement mode, 
        and thus should contain the same amount of spectra in case of a multi-spectral mode.
    outfile : String
        H5 file name to store the converted data in.

    Returns
    -------
    None.

    """
    # pdzfile = "C:/work/OneDrive - UGent/Research/Raman/20240613_Peter_SpanjeMozaic/JAEN - Mosaics/Jaen EDXRF/Backup-14-06-2024-151938/01468-GeoExploration.pdz"
    import pdz_read as PR
    
    if type(pdzfiles) != type(list()):
        pdzfiles = [pdzfiles]
                
    for index, pdzfile in enumerate(pdzfiles):
        pdzdata = PR.read_pdz(pdzfile) 
        # will return a list of structures if multiple spectra were found in a single file.
        #   The max amount appears to be three
        if type(pdzdata) != type(list()):
            pdzdata = [pdzdata]
            n_spectra = 1
        else:
            n_spectra = len(pdzdata)

        #define the arrays to store all data based on amount of pdzfiles and n_spectra within the file
        # Note that all iterated files should have the same amount of specrta inside
        if index == 0:            
            i1 = np.zeros((n_spectra,len(pdzfiles)))
            mot2 = np.array((n_spectra,len(pdzfiles))).fill(np.nan)
            mot1 = np.array((n_spectra,len(pdzfiles)))
            i0 = np.array((n_spectra,len(pdzfiles)))
            tm = np.array((n_spectra,len(pdzfiles)))
            icr = np.array((n_spectra,len(pdzfiles)))
            ocr = np.array((n_spectra,len(pdzfiles)))
            spectra = np.array((n_spectra,len(pdzfiles),len(pdzdata['spectrum'])))
        
        for i in range(n_spectra):
            mot1[i, index] = pdzdata['ID']
            i0[i, index] = pdzdata['current']
            tm[i, index] = pdzdata['realtime']
            spectra[i, index, :] = pdzdata['spectrum']
            icr[i, index] = pdzdata['icr']
            ocr[i, index] = pdzdata['ocr']

    nspe = len(pdzfiles)
    nchnls = spectra.shape[2]
    for i in range(n_spectra):
        mot1_tmp = np.asarray([n.encode('utf8') for n in mot1[i,:]]).reshape((nspe,1))
        mot2_tmp = np.asarray(mot2[i,:]).reshape((nspe,1))
        ocr_tmp = np.asarray(ocr[i,:]).reshape((nspe,1))
        icr_tmp = np.asarray(icr[i,:]).reshape((nspe,1))
        tm_tmp = np.asarray(tm[i,:]).reshape((nspe,1))
        i1_tmp = np.asarray(i1[i,:]).reshape((nspe,1))
        i0_tmp = np.asarray(i0[i,:]).reshape((nspe,1))
        spectra_tmp = np.asarray(spectra).reshape((nspe,1,nchnls))
        sumspec = np.sum(spectra_tmp[:], axis=(0,1))
        maxspec = np.zeros(sumspec.shape[0])
        for i in range(sumspec.shape[0]):
            maxspec[i] = spectra_tmp[:,:,i].max()
    
        # Hooray! We read all the information! Let's write it to a separate file
        measurement_id = os.path.splitext(outfile)[0]
        print("Writing merged file: "+measurement_id+f'_{i}'+"_merge.h5...", end=" ")
        f = h5py.File(measurement_id+"_merge.h5", 'w', locking=True)
        f.create_dataset('cmd', data='dscan Handheld Bruker')
        f.create_dataset('raw/channel00/spectra', data=spectra_tmp, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/icr', data=icr_tmp, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/ocr', data=ocr_tmp, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/sumspec', data=np.squeeze(sumspec), compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/maxspec', data=np.squeeze(maxspec), compression='gzip', compression_opts=4)
        f.create_dataset('raw/I0', data=i0_tmp, compression='gzip', compression_opts=4)
        f.create_dataset('raw/I1', data=i1_tmp, compression='gzip', compression_opts=4)
        f.create_dataset('raw/acquisition_time', data=tm_tmp, compression='gzip', compression_opts=4)
        dset = f.create_dataset('mot1', data=mot1_tmp)
        dset.attrs['Name'] = 'hxrf'
        dset = f.create_dataset('mot2', data=mot2_tmp, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = 'hxrf'
        f.close()
        print("Done")
    
    
##############################################################################
def ConvDeltaCsv(csvfile):
    """
    Read the Delta Premium handheld CSV files and restructure as H5 for further processing
    

    Parameters
    ----------
    csvfile : String
        File path to the CSV file to be converted.

    Returns
    -------
    None.

    """
    import pandas as pd

    try:
        file = pd.read_csv(csvfile, header=None, sep=None)
    except UnicodeError:
        file = pd.read_csv(csvfile, header=None, sep=None, encoding='utf_16')
    
    rowheads = [n for n in file[0] if n is not np.nan]
    # loop through the different columns and assign them to spectra0, spectra2 etc.
    spectra0 = []
    ocr0 = []
    spectra1 = []
    ocr1 = []
    i0_0 = []
    i0_1 = []
    i1 = []
    tm0 = []
    tm1 = []
    mot1 = []
    mot2 = []
    for key in file.keys()[2:-1]:
        # in principle the instrument always measures in two modes consecutively, so if this column 
        # does not have a subsequent matching partner, something went wrong
        if file[key][rowheads.index('ExposureNum')] == '1':
            if file[key][rowheads.index('TestID')] != file[key+1][rowheads.index('TestID')]:
                print("WARNING: measurement %s does not have a matching 10keV mode measurement. Measurement exempt from H5 file." % file[key][rowheads.index('TestID')])
            else: #the measurement in 'key' has an accompanying 10keV mode measurement in 'key+1'
                i1.append(0) #no transmission counter registered
                mot1.append(file[key][rowheads.index('TestID')])
                mot2.append(np.nan)
                
                i0_0.append(float(file[key][rowheads.index('TubeCurrentMon')]))
                tm0.append(float(file[key][rowheads.index('Livetime')]))
                spectra0.append([file[key][rowheads.index('TimeStamp')+1:].astype(float)])
                ocr0.append(np.sum(spectra0[-1]))
                i0_1.append(float(file[key+1][rowheads.index('TubeCurrentMon')])) 
                tm1.append(float(file[key+1][rowheads.index('Livetime')])) 
                spectra1.append([file[key+1][rowheads.index('TimeStamp')+1:].astype(float)])
                ocr1.append(np.sum(spectra1[-1]))
    nspe = len(mot1)
    nchnls0 = np.asarray(spectra0[0]).shape
    nchnls1 = np.asarray(spectra1[0]).shape
    mot1 = np.asarray([n.encode('utf8') for n in mot1]).reshape((nspe,1))
    mot2 = np.asarray(mot2).reshape((nspe,1))
    ocr0 = np.asarray(ocr0).reshape((nspe,1))
    ocr1 = np.asarray(ocr1).reshape((nspe,1))
    tm0 = np.asarray(tm0).reshape((nspe,1))
    tm1 = np.asarray(tm1).reshape((nspe,1))
    i1 = np.asarray(i1).reshape((nspe,1))
    i0_0 = np.asarray(i0_0).reshape((nspe,1))
    i0_1 = np.asarray(i0_1).reshape((nspe,1))
    spectra0 = np.asarray(spectra0).reshape((nspe,1,nchnls0[1]))
    spectra1 = np.asarray(spectra1).reshape((nspe,1,nchnls1[1]))
    sumspec0 = np.sum(spectra0[:], axis=(0,1))
    maxspec0 = np.zeros(sumspec0.shape[0])
    for i in range(sumspec0.shape[0]):
        maxspec0[i] = spectra0[:,:,i].max()

    sumspec1 = np.sum(spectra1[:], axis=(0,1))
    maxspec1 = np.zeros(sumspec1.shape[0])
    for i in range(sumspec1.shape[0]):
        maxspec1[i] = spectra1[:,:,i].max()


    # Hooray! We read all the information! Let's write it to a separate file
    measurement_id = os.path.splitext(csvfile)[0]
    print("Writing merged file: "+measurement_id+"_merge_40keV.h5...", end=" ")
    f = h5py.File(measurement_id+"_merge_40keV.h5", 'w', locking=True)
    f.create_dataset('cmd', data='dscan Handheld Delta Premium 40keV mode')
    f.create_dataset('raw/channel00/spectra', data=spectra0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/icr', data=ocr0, compression='gzip', compression_opts=4) #as time is expressed as livetime no icr/ocr correction should be applied
    f.create_dataset('raw/channel00/ocr', data=ocr0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/sumspec', data=np.squeeze(sumspec0), compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/maxspec', data=np.squeeze(maxspec0), compression='gzip', compression_opts=4)
    f.create_dataset('raw/I0', data=i0_0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/I1', data=i1, compression='gzip', compression_opts=4)
    f.create_dataset('raw/acquisition_time', data=tm0, compression='gzip', compression_opts=4)
    dset = f.create_dataset('mot1', data=mot1)
    dset.attrs['Name'] = 'hxrf'
    dset = f.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
    dset.attrs['Name'] = 'hxrf'
    f.close()
    print("Done")
    
    print("Writing merged file: "+measurement_id+"_merge_10keV.h5...", end=" ")
    f = h5py.File(measurement_id+"_merge_10keV.h5", 'w', locking=True)
    f.create_dataset('cmd', data='dscan Handheld Delta Premium 10keV mode')
    f.create_dataset('raw/channel00/spectra', data=spectra1, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/icr', data=ocr1, compression='gzip', compression_opts=4) #as time is expressed as livetime no icr/ocr correction should be applied
    f.create_dataset('raw/channel00/ocr', data=ocr1, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/sumspec', data=np.squeeze(sumspec1), compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/maxspec', data=np.squeeze(maxspec1), compression='gzip', compression_opts=4)
    f.create_dataset('raw/I0', data=i0_1, compression='gzip', compression_opts=4)
    f.create_dataset('raw/I1', data=i1, compression='gzip', compression_opts=4)
    f.create_dataset('raw/acquisition_time', data=tm1, compression='gzip', compression_opts=4)
    dset = f.create_dataset('mot1', data=mot1)
    dset.attrs['Name'] = 'hxrf'
    dset = f.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
    dset.attrs['Name'] = 'hxrf'
    f.close()
    print("Done")

    
##############################################################################
def read_P06_spectra(file, sc_id, ch):
    """
    Read the spectra from a P06 nexus file.
    

    Parameters
    ----------
    file : string
        File name identifier.
    sc_id : string
        Scan name identifier.
    ch : (list of) string(s)
        Detector channel identifier. If a list of strings is provided, the detector channels are summed.

    Returns
    -------
    spe0_arr : float array
        The resulting spectrum contained within the nexus file.
    icr0_arr : float
        the ICR counter value for this measurement.
    ocr0_arr : TYPE
        the OCR counter value for this measurement.

    """
    if type(ch[0]) is str:
        # Reading the spectra files, icr and ocr
        print("Reading " +sc_id+"/"+ch[0]+"/"+file +"...", end=" ")
        f = h5py.File(sc_id+"/"+ch[0]+"/"+file, 'r', locking=True)
        if type(ch[1]) is str:
            spe0_arr = f['entry/instrument/xspress3/'+ch[1]+'/histogram'][:]
            try:
                icr0_arr = f['entry/instrument/xspress3/'+ch[1]+'/scaler/allEvent'][:]
                ocr0_arr = f['entry/instrument/xspress3/'+ch[1]+'/scaler/allGood'][:]
            except KeyError:
                icr0_arr = f['entry/instrument/xspress3/'+ch[1]+'/scaler/allevent'][:]
                ocr0_arr = f['entry/instrument/xspress3/'+ch[1]+'/scaler/allgood'][:]
        elif type(ch[1]) is list: #if multiple channels provided we want to add them 
            spe0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0]+'/histogram'][:])
            try:
                icr0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allEvent'][:])
                ocr0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allGood'][:])
            except KeyError:
                icr0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allevent'][:])
                ocr0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allgood'][:])
            for chnl in ch[1][1:]:
                spe0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/histogram'][:])
                try:
                    icr0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/scaler/allEvent'][:])
                    ocr0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/scaler/allGood'][:])
                except KeyError:
                    icr0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/scaler/allevent'][:])
                    ocr0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/scaler/allgood'][:])
        f.close()
        print("read")
    elif type(ch[0]) is list:
        # Reading the spectra files, icr and ocr
        dev = ch[0][0]
        print("Reading " +sc_id+"/"+dev+"/"+file +"...", end=" ")
        f = h5py.File(sc_id+"/"+dev+"/"+file, 'r', locking=True)
        # read spectra and icr/ocr from first device (ch[0][0]) so we can later add the rest
        #   as ch[0] is a list, ch[1] is also expected to be a list!
        if type(ch[1][0]) is str:
            spe0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0]+'/histogram'][:])
            try:
                icr0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allEvent'][:])
                ocr0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allGood'][:])
            except KeyError:
                icr0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allevent'][:])
                ocr0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allgood'][:])
        elif type(ch[1][0]) is list: #if multiple channels provided for this device we want to add them 
            spe0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0][0]+'/histogram'][:])
            try:
                icr0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0][0]+'/scaler/allEvent'][:])
                ocr0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0][0]+'/scaler/allGood'][:])
            except KeyError:
                icr0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0][0]+'/scaler/allevent'][:])
                ocr0_arr = np.asarray(f['entry/instrument/xspress3/'+ch[1][0][0]+'/scaler/allgood'][:])
            for chnl in ch[1][0][1:]:
                spe0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/histogram'][:])
                try:
                    icr0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/scaler/allEvent'][:])
                    ocr0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/scaler/allGood'][:])
                except KeyError:
                    icr0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/scaler/allevent'][:])
                    ocr0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/scaler/allgood'][:])
        f.close()
        print("read")
        # Now let's read the following spectra/devices
        for i in range(len(ch[0])-1):
            dev = ch[0][i+1]
            # Reading the spectra files, icr and ocr
            print("Reading " +sc_id+"/"+dev+"/"+file +"...", end=" ")
            f = h5py.File(sc_id+"/"+dev+"/"+file, 'r', locking=True)
            if type(ch[1][i+1]) is str:
                spe0_arr += np.asarray(f['entry/instrument/xspress3/'+ch[1][i+1]+'/histogram'][:])
                try:
                    icr0_arr += np.asarray(f['entry/instrument/xspress3/'+ch[1][i+1]+'/scaler/allEvent'][:])
                    ocr0_arr += np.asarray(f['entry/instrument/xspress3/'+ch[1][i+1]+'/scaler/allGood'][:])
                except KeyError:
                    icr0_arr += np.asarray(f['entry/instrument/xspress3/'+ch[1][i+1]+'/scaler/allevent'][:])
                    ocr0_arr += np.asarray(f['entry/instrument/xspress3/'+ch[1][i+1]+'/scaler/allgood'][:])
            elif type(ch[1][i+1]) is list: #if multiple channels provided for this device we want to add them 
                for chnl in ch[1][i+1][:]:
                    spe0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/histogram'][:])
                    try:
                        icr0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/scaler/allEvent'][:])
                        ocr0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/scaler/allGood'][:])
                    except KeyError:
                        icr0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/scaler/allevent'][:])
                        ocr0_arr += np.asarray(f['entry/instrument/xspress3/'+chnl+'/scaler/allgood'][:])
            f.close()    
            print("read")
    return spe0_arr, icr0_arr, ocr0_arr
    
##############################################################################
def ConvP06Nxs(scanid, sort=True, ch0=['xspress3_01','channel00'], ch1=None, readas1d=False, outdir=None):
    """
    Merges separate P06 nxs files to 1 handy H5 file containing 2D array of spectra, relevant motor positions, 
    I0 counter, ICR, OCR and mesaurement time.


    Parameters
    ----------
    scanid : string
        Scan identifier.
    sort : Boolean, optional
        If True the data is sorted following the motor encoder positions. The default is True.
    ch0 : (list of) String(s), optional
        First detector channel identifiers. This can be a list of strings, or a list of lists where each combination 
        of ch0[0] and ch0[1] strings are combined to generate unique detector identifiers, which are then summed to a single spectrum.
        The default is ['xspress3_01','channel00'].
    ch1 : (list of) String(s), optional
        Similar as ch0, but then for a second detector. The default is None.
    readas1d : Boolean, optional
        If True, data is read as a single 1 dimensional array and no reshaping of the scan to a 2D image is attempted. The default is False.
    outdir : string, optional
        If None (default) the converted data is stored in the same directory as the raw data. If not None the data is stored in the directory provided by the user.

    Returns
    -------
    None.

    """
    scanid = np.asarray(scanid)
    if outdir is None:
        if scanid.size == 1:
            scan_suffix = '/'.join(str(scanid).split('/')[0:-2])+'/scan'+str(scanid).split("_")[-1]
        else:
            scan_suffix = '/'.join(scanid[0].split('/')[0:-2])+'/scan'+str(scanid[0]).split("_")[-1]+'-'+str(scanid[-1]).split("_")[-1]
    else:
        if not outdir.endswith('/'):
            outdir += '/'
        if scanid.size == 1:
            scan_suffix = outdir+'scan'+str(scanid).split("_")[-1]
        else:
            scan_suffix = outdir+'scan'+str(scanid[0]).split("_")[-1]+'-'+str(scanid[-1]).split("_")[-1]

    # Make the (empty) dataset arrays in the merged H5 file
    files = list("")
    for k in range(scanid.size):
        if scanid.size == 1:
            sc_id = str(scanid)
        else:
            sc_id = str(scanid[k])
        # file with name scanid contains info on scan command
        f = h5py.File(sc_id+'.nxs', 'r', locking=True)
        scan_cmd = str(f['scan/program_name'].attrs["scan_command"][:])
        scan_cmd = np.asarray(scan_cmd.strip("[]'").split(" "))
        print(' '.join(scan_cmd))
        f.close()

        # actual spectrum scan files are in dir scanid/scan_0XXX/xspress3_01
        if type(ch0[0]) == type(str()):
            for file in sorted(os.listdir(sc_id+"/"+ch0[0])):
                if file.endswith(".nxs"):
                    files.append(file)
        else: #ch0 is a list
            for file in sorted(os.listdir(sc_id+"/"+ch0[0][0])):
                if file.endswith(".nxs"):
                    files.append(file)
        firstgo = True #set firstgo parameter to creat datasets at start
        for file in files:
            spe0_arr, icr0_arr, ocr0_arr = read_P06_spectra(file, sc_id, ch0)
            if ch1 is not None:
                spe1_arr, icr1_arr, ocr1_arr = read_P06_spectra(file, sc_id, ch1)
            if firstgo:
                with h5py.File(scan_suffix+"_merge.h5", 'w', locking=True) as hf:
                    hf.create_dataset('raw/channel00/spectra', data=spe0_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,spe0_arr.shape[1]))
                    hf.create_dataset('raw/channel00/icr', data=icr0_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))
                    hf.create_dataset('raw/channel00/ocr', data=ocr0_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))
                    if ch1 is not None:
                        hf.create_dataset('raw/channel01/spectra', data=spe1_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,spe1_arr.shape[1]))
                        hf.create_dataset('raw/channel01/icr', data=icr1_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))
                        hf.create_dataset('raw/channel01/ocr', data=ocr1_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))
                    firstgo = False
            else:
                with h5py.File(scan_suffix+"_merge.h5", 'a', locking=True) as hf:
                    hf['raw/channel00/spectra'].resize((hf['raw/channel00/spectra'].shape[0]+spe0_arr.shape[0], hf['raw/channel00/spectra'].shape[1]))
                    hf['raw/channel00/spectra'][-spe0_arr.shape[0]:,:] = spe0_arr
                    hf['raw/channel00/icr'].resize((hf['raw/channel00/icr'].shape[0]+icr0_arr.shape[0]), axis=0)
                    hf['raw/channel00/icr'][-icr0_arr.shape[0]:] = icr0_arr
                    hf['raw/channel00/ocr'].resize((hf['raw/channel00/ocr'].shape[0]+ocr0_arr.shape[0]), axis=0)
                    hf['raw/channel00/ocr'][-ocr0_arr.shape[0]:] = ocr0_arr
                    if ch1 is not None:
                        hf['raw/channel01/spectra'].resize((hf['raw/channel01/spectra'].shape[0]+spe1_arr.shape[0], hf['raw/channel01/spectra'].shape[1]))
                        hf['raw/channel01/spectra'][-spe1_arr.shape[0]:,:] = spe1_arr
                        hf['raw/channel01/icr'].resize((hf['raw/channel01/icr'].shape[0]+icr1_arr.shape[0]), axis=0)
                        hf['raw/channel01/icr'][-icr1_arr.shape[0]:] = icr1_arr
                        hf['raw/channel01/ocr'].resize((hf['raw/channel01/ocr'].shape[0]+ocr1_arr.shape[0]), axis=0)
                        hf['raw/channel01/ocr'][-ocr1_arr.shape[0]:] = ocr1_arr
        if os.path.isfile(sc_id+"/adc_01/"+files[-1]) is True:
            adc = '/adc_01/'
        elif os.path.isfile(sc_id+"/adc01/"+files[-1]) is True:
            adc = '/adc01/'
        else:
            adc = None
        if adc != None:
            firstgo = True #set firstgo parameter to creat datasets at start
            for file in files:
                # Reading I0 and measurement time data
                print("Reading " +sc_id+adc+file +"...", end=" ")
                f = h5py.File(sc_id+adc+file, 'r', locking=True)
                i0_arr = f['entry/data/value1'][:]
                i1_arr = f['entry/data/value2'][:]
                tm_arr = f['entry/data/exposuretime'][:]
                f.close()
                print("read")
                with h5py.File(scan_suffix+"_merge.h5", 'a', locking=True) as hf:
                    if firstgo:
                        hf.create_dataset('raw/I0', data=i0_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))
                        hf.create_dataset('raw/I1', data=i1_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))
                        hf.create_dataset('raw/acquisition_time', data=tm_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))
                        firstgo = False
                    else:
                        hf['raw/I0'].resize((hf['raw/I0'].shape[0]+i0_arr.shape[0]), axis=0)
                        hf['raw/I0'][-i0_arr.shape[0]:] = i0_arr
                        hf['raw/I1'].resize((hf['raw/I1'].shape[0]+i1_arr.shape[0]), axis=0)
                        hf['raw/I1'][-i1_arr.shape[0]:] = i1_arr
                        hf['raw/acquisition_time'].resize((hf['raw/acquisition_time'].shape[0]+tm_arr.shape[0]), axis=0)
                        hf['raw/acquisition_time'][-tm_arr.shape[0]:] = tm_arr
        else: #the adc_01 does not contain full list of nxs files as xpress etc, but only consists single main nxs file with all scan data
            if os.path.isdir(sc_id+"/adc_01") is True:
                adc = '/adc_01/'
            elif os.path.isdir(sc_id+"/adc01") is True:
                adc = '/adc01/'
            file = os.listdir(sc_id+adc[:-1])
            print("Reading " +sc_id+adc+file[0] +"...", end=" ") #os.listdir returns a list, so we pick first element as only 1 should be there right now
            f = h5py.File(sc_id+adc+file[0], 'r', locking=True)
            i0_arr = f['entry/data/Value1'][:]
            i1_arr = f['entry/data/Value2'][:]
            tm_arr = f['entry/data/ExposureTime'][:]
            f.close()
            print("read")  
            with h5py.File(scan_suffix+"_merge.h5", 'a', locking=True) as hf:
                hf.create_dataset('raw/I0', data=i0_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))
                hf.create_dataset('raw/I1', data=i1_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))
                hf.create_dataset('raw/acquisition_time', data=tm_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))

        # actual pilcgenerator files can be different structure depending on type of scan
        files = list("")
        for i in range(5,0,-1):
            if os.path.isdir(sc_id+"/pilctriggergenerator_0"+str(i)) is True:
                for file in sorted(os.listdir(sc_id+"/pilctriggergenerator_0"+str(i))):
                    if file.endswith(".nxs"):
                        files.append(file)
                pilcid = "/pilctriggergenerator_0"+str(i)+"/"
        try:
            md_dict = {}
            with open("/".join(sc_id.split("/")[0:-2])+"/scan_logbook.txt", "r") as file_handle:
                raw_data = file_handle.readlines()
                for scan_entry in raw_data:
                    tmp_dict = eval(scan_entry)
                    md_dict[tmp_dict['scan']['scan_prefix']] = tmp_dict
                dictionary = md_dict[sc_id.split('/')[-1]]
        except Exception as ex:
            print("Warning: ", ex)
            dictionary = md_dict
        if files == []:
            # Reading motor positions from main nxs file. Assumes only 2D scans are performed (stores encoder1 and 2 values)
            print("Reading " +sc_id +".nxs...", end=" ")
            f = h5py.File(sc_id+'.nxs', 'r', locking=True)
            if scan_cmd[0] == 'timescanc' or scan_cmd[0] == 'timescan':
                mot1_arr = np.asarray(f["scan/data/timestamp"][:])
                mot1_name = "timestamp"
                mot2_arr = np.asarray(f["scan/data/timestamp"][:])
                mot2_name = "timestamp"                                        
            elif scan_cmd[0] == 'dscan' or scan_cmd[0] == 'ascan':
                mot1_arr = np.asarray(f["scan/data/"+str(scan_cmd[1])][:])
                mot1_name = str(scan_cmd[1])
                mot2_arr = np.asarray(f["scan/data/"+str(scan_cmd[1])][:])
                mot2_name = str(scan_cmd[1])                            
            else:
                mot1_arr = np.asarray(f["scan/data/"+str(scan_cmd[1])][:])
                mot1_name = str(scan_cmd[1])
                mot2_arr = np.asarray(f["scan/data/"+str(scan_cmd[5])][:])
                mot2_name = str(scan_cmd[5])            
            with h5py.File(scan_suffix+"_merge.h5", 'a', locking=True) as hf:
                hf.create_dataset('mot1', data=mot1_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))
                hf.create_dataset('mot2', data=mot2_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))
            f.close()
            print("read")
        else:
            firstgo = True #set firstgo parameter to creat datasets at start
            for file in files:
                # Reading motor positions. Assumes only 2D scans are performed (stores encoder1 and 2 values)
                print("Reading " +sc_id+pilcid+file +"...", end=" ")
                f = h5py.File(sc_id+pilcid+file, 'r', locking=True)
                counter_id = f['entry/data/counter']
                enc_vals = []
                for i in range(10):
                    if 'encoder_'+str(i) in list(f['entry/data'].keys()):
                        enc_vals.append(f['entry/data/encoder_'+str(i)])
                enc_names = [str(enc.attrs["Name"]).strip("'") for enc in enc_vals]
                if scan_cmd[1] in enc_names:
                    mot1_arr = enc_vals[enc_names.index(scan_cmd[1])]
                    mot1_name = enc_names[enc_names.index(scan_cmd[1])]
                else: # in this case the motor in not in the encoder list, so could be a virtual motor... let's look in the accompanying python logbook
                    try:
                        pivot = dictionary["axes"]["axis0"]["virtual_motor_config"]["pivot_points"]
                        mot_list = list(dictionary["axes"]["axis0"]["virtual_motor_config"]["real_members"].keys())
                        mot1a = enc_vals[enc_names.index(mot_list[0])]
                        mot1a_contrib = dictionary["axes"]["axis0"]["virtual_motor_config"]["real_members"][mot_list[0]]["contribution"]
                        mot1b = enc_vals[enc_names.index(mot_list[1])]
                        mot1b_contrib = dictionary["axes"]["axis0"]["virtual_motor_config"]["real_members"][mot_list[1]]["contribution"]
                        mot1_arr = mot1a_contrib*(np.asarray(mot1a)-pivot[0])+mot1b_contrib*(np.asarray(mot1b)-pivot[1]) + pivot[0] #just took first as in this case it's twice the same i.e. [250,250]
                        mot1_name = str(scan_cmd[1])
                    except Exception:
                        try:
                            if scan_cmd[0] == 'timescanc' or scan_cmd[0] == 'timescan':
                                print("Warning: timescan(c) command; using "+str(enc_names[0])+" encoder value...", end=" ")
                                mot1_arr = enc_vals[0]
                                mot1_name = enc_names[0]
                            else:
                                f2 = h5py.File(sc_id+'.nxs','r', locking=True)
                                mot1_arr = np.asarray(f2["scan/data/"+str(scan_cmd[1])][:])
                                mot1_name = str(scan_cmd[1])
                                f2.close()
                                counter_id = np.zeros(mot1_arr.shape)+1
                        except KeyError:
                            if scan_cmd[0] == 'timescanc' or scan_cmd[0] == 'timescan':
                                print("Warning: timescan(c) command; using "+str(enc_names[0])+" encoder value...", end=" ")
                                mot1_arr = enc_vals[0]
                                mot1_name = enc_names[0]
                            else:
                                if scan_cmd[1] == 'scanz':
                                    mot1_arr = enc_vals[enc_names.index('scanw')]
                                    mot1_name = enc_names[enc_names.index('scanw')]
                                elif scan_cmd[1] == 'scany':
                                    mot1_arr = enc_vals[enc_names.index('scanu')]
                                    mot1_name = enc_names[enc_names.index('scanu')]
                                elif scan_cmd[1] == 'scanx':
                                    mot1_arr = enc_vals[enc_names.index('scanv')]
                                    mot1_name = enc_names[enc_names.index('scanv')]
                                else:
                                    print("Warning: "+str(scan_cmd[1])+" not found in encoder list dictionary; using "+str(enc_names[0])+" instead...", end=" ")
                                    mot1_arr = enc_vals[0]
                                    mot1_name = enc_names[0]
                if scan_cmd.shape[0] > 6 and scan_cmd[5] in enc_names and len(enc_vals[enc_names.index(scan_cmd[5])]) == len(mot1_arr):
                    mot2_arr = enc_vals[enc_names.index(scan_cmd[5])]
                    mot2_name = enc_names[enc_names.index(scan_cmd[5])]
                else:
                    try:
                        pivot = dictionary["axes"]["axis1"]["virtual_motor_config"]["pivot_points"]
                        mot_list = list(dictionary["axes"]["axis1"]["virtual_motor_config"]["real_members"].keys())
                        mot2a = enc_vals[enc_names.index(mot_list[0])]
                        mot2a_contrib = dictionary["axes"]["axis1"]["virtual_motor_config"]["real_members"][mot_list[0]]["contribution"]
                        mot2b = enc_vals[enc_names.index(mot_list[1])]
                        mot2b_contrib = dictionary["axes"]["axis1"]["virtual_motor_config"]["real_members"][mot_list[1]]["contribution"]
                        mot2_arr = mot2a_contrib*(np.asarray(mot2a)-pivot[0])+mot2b_contrib*(np.asarray(mot2b)-pivot[1]) + pivot[0] #just took first as in this case it's twice the same i.e. [250,250]
                        mot2_name = str(scan_cmd[5])
                    except Exception:
                        try:
                            if scan_cmd[0] == 'timescanc' or scan_cmd[0] == 'timescan':
                                print("Warning: timescan(c) command; using "+str(enc_names[1])+" encoder value...", end=" ")
                                mot2_arr = enc_vals[1]
                                mot2_name = enc_names[1]
                            else:
                                f2 = h5py.File(sc_id+'.nxs','r', locking=True)
                                if scan_cmd[0] == 'ascan' or scan_cmd[0] == 'dscan':
                                    mot2_arr = np.asarray(f2["scan/data/"+str(scan_cmd[1])][:])
                                    mot2_name = str(scan_cmd[1])
                                else:
                                    mot2_arr = np.asarray(f2["scan/data/"+str(scan_cmd[5])][:])
                                    mot2_name = str(scan_cmd[5])
                                f2.close()
                        except KeyError:
                            if scan_cmd[0] == 'timescanc' or scan_cmd[0] == 'timescan':
                                print("Warning: timescan(c) command; using "+str(enc_names[1])+" encoder value...", end=" ")
                                mot2_arr = enc_vals[1]
                                mot2_name = enc_names[1]
                            else:
                                if scan_cmd[5] == 'scanz':
                                    mot2_arr = enc_vals[enc_names.index('scanw')]
                                    mot2_name = enc_names[enc_names.index('scanw')]
                                elif scan_cmd[5] == 'scany':
                                    mot2_arr = enc_vals[enc_names.index('scanu')]
                                    mot2_name = enc_names[enc_names.index('scanu')]
                                elif scan_cmd[5] == 'scanx':
                                    mot2_arr = enc_vals[enc_names.index('scanv')]
                                    mot2_name = enc_names[enc_names.index('scanv')]
                                else:
                                    print("Warning: "+str(scan_cmd[5])+" not found in encoder list dictionary; using "+str(enc_names[1])+" instead...", end=" ")
                                    mot2_arr = enc_vals[1]
                                    mot2_name = enc_names[1]
                if np.sum(counter_id) != counter_id.shape[0]:
                    mot1_arr = np.delete(mot1_arr, np.where(counter_id != 1))
                    mot2_arr = np.delete(mot2_arr, np.where(counter_id != 1))
                print("read")
                with h5py.File(scan_suffix+"_merge.h5", 'a', locking=True) as hf:
                    if firstgo:
                        hf.create_dataset('mot1', data=mot1_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))
                        hf.create_dataset('mot2', data=mot2_arr, compression='gzip', compression_opts=4, chunks=True, maxshape=(None,))
                        firstgo = False
                    else:
                        hf['mot1'].resize((hf['mot1'].shape[0]+mot1_arr.shape[0]), axis=0)
                        hf['mot1'][-mot1_arr.shape[0]:] = mot1_arr
                        hf['mot2'].resize((hf['mot2'].shape[0]+mot2_arr.shape[0]), axis=0)
                        hf['mot2'][-mot2_arr.shape[0]:] = mot2_arr
                f.close()

            
    # try to reshape if possible (for given scan_cmd and extracted data points), else just convert to np.asarray
    # let's translate scan command to figure out array dimensions we want to fill
    #   1D scan (ascan, dscan, timescan, ...) contain 7 parts, i.e. dscan samx 0 1 10 1 False
    #       sometimes False at end appears to be missing
    if scan_cmd[0][0] == 'c' and  scan_cmd[0] != 'cnt':
        if scan_cmd[0] == 'cdmeshs':
            xdim = np.floor(np.abs(float(scan_cmd[2])-float(scan_cmd[3]))/float(scan_cmd[4]))
        else:
            xdim = int(scan_cmd[4])
    elif scan_cmd[0] == 'cnt':
        xdim = 1
    elif scan_cmd[0] == 'timescanc':
        xdim = int(scan_cmd[1])        
    elif scan_cmd[0] == 'timescan': # to check...
        xdim = int(scan_cmd[1])+1
    else:
        xdim = int(scan_cmd[4])+1
    ydim = 1
    if scan_cmd.shape[0] > 7:
        if scan_cmd[0] == 'cdmeshs':
            ydim = np.floor(np.abs(float(scan_cmd[6])-float(scan_cmd[7]))/float(scan_cmd[8]))+1
        else:
            ydim = int(scan_cmd[8])+1
    with h5py.File(scan_suffix+"_merge.h5", 'r+', locking=True) as hf:
        spectra0 = np.asarray(hf['raw/channel00/spectra'])
        if spectra0.shape[0] == xdim*ydim and readas1d is not True:
            spectra0 = spectra0.reshape((ydim, xdim, spectra0.shape[1]))
            del hf['raw/channel00/spectra']
            hf.create_dataset('raw/channel00/spectra', data=spectra0, compression='gzip', compression_opts=4)
            del spectra0 #clear it from memory, probably not really needed but no harm either
            icr0 = np.asarray(hf['raw/channel00/icr'])
            ocr0 = np.asarray(hf['raw/channel00/ocr'])
            i0 = np.asarray(hf['raw/I0'])
            i1 = np.asarray(hf['raw/I1'])
            tm = np.asarray(hf['raw/acquisition_time'])
            icr0 = np.asarray(icr0).reshape((ydim, xdim))
            ocr0 = np.asarray(ocr0).reshape((ydim, xdim))
            i0 = np.asarray(i0).reshape((ydim, xdim))
            i1 = np.asarray(i1).reshape((ydim, xdim))
            tm = np.asarray(tm).reshape((ydim, xdim))
            del hf['raw/channel00/icr']
            del hf['raw/channel00/ocr']
            del hf['raw/I0']
            del hf['raw/I1']
            del hf['raw/acquisition_time']
            hf.create_dataset('raw/channel00/icr', data=icr0, compression='gzip', compression_opts=4)
            hf.create_dataset('raw/channel00/ocr', data=ocr0, compression='gzip', compression_opts=4)
            hf.create_dataset('raw/I0', data=i0, compression='gzip', compression_opts=4)
            hf.create_dataset('raw/I1', data=i1, compression='gzip', compression_opts=4)
            hf.create_dataset('raw/acquisition_time', data=tm, compression='gzip', compression_opts=4)
            if ch1 is not None:
                spectra1 = np.asarray(hf['raw/channel01/spectra'])
                spectra1 = spectra1.reshape((ydim, xdim, spectra1.shape[1]))
                del hf['raw/channel01/spectra']
                hf.create_dataset('raw/channel01/spectra', data=spectra1, compression='gzip', compression_opts=4)
                del spectra1 #clear it from memory, probably not really needed but no harm either
                icr1 = np.asarray(hf['raw/channel01/icr'])
                ocr1 = np.asarray(hf['raw/channel01/ocr'])
                icr1 = np.asarray(icr1).reshape((ydim, xdim))
                ocr1 = np.asarray(ocr1).reshape((ydim, xdim))
                del hf['raw/channel01/icr']
                del hf['raw/channel01/ocr']
                hf.create_dataset('raw/channel01/icr', data=icr1, compression='gzip', compression_opts=4)
                hf.create_dataset('raw/channel01/ocr', data=ocr1, compression='gzip', compression_opts=4)
            mot1 = np.asarray(hf['mot1'])
            mot2 = np.asarray(hf['mot2'])
            if np.asarray(mot1).shape[0] < xdim*ydim:
                print("Warning: mot1 has less than "+str(xdim*ydim)+" elements; padding with zero.")
                mot1 = np.concatenate((np.asarray(mot1), np.zeros(xdim*ydim-np.asarray(mot1).shape[0])))
            if np.asarray(mot2).shape[0] < xdim*ydim:
                print("Warning: mot2 has less than "+str(xdim*ydim)+" elements; padding with zero.")
                mot2 = np.concatenate((np.asarray(mot2), np.zeros(xdim*ydim-np.asarray(mot2).shape[0])))
            mot1 = np.asarray(mot1[0:xdim*ydim]).reshape((ydim, xdim))
            mot2 = np.asarray(mot2[0:xdim*ydim]).reshape((ydim, xdim))
            del hf['mot1']
            del hf['mot2']
            dset = hf.create_dataset('mot1', data=mot1, compression='gzip', compression_opts=4)
            dset.attrs['Name'] = mot1_name
            dset = hf.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
            dset.attrs['Name'] = mot2_name
            timetrig = False
        elif spectra0.shape[0] < xdim*ydim and readas1d is not True:
            zerosize = xdim*ydim-spectra0.shape[0]
            zeros = np.zeros((zerosize, spectra0.shape[1]))
            spectra0 = np.concatenate((spectra0, zeros)).reshape((ydim, xdim, spectra0.shape[1]))
            del hf['raw/channel00/spectra']
            hf.create_dataset('raw/channel00/spectra', data=spectra0, compression='gzip', compression_opts=4)
            del spectra0 #clear it from memory, probably not really needed but no harm either
            icr0 = np.asarray(hf['raw/channel00/icr'])
            ocr0 = np.asarray(hf['raw/channel00/ocr'])
            i0 = np.asarray(hf['raw/I0'])
            i1 = np.asarray(hf['raw/I1'])
            tm = np.asarray(hf['raw/acquisition_time'])
            zeros = np.zeros((zerosize))
            icr0 = np.concatenate((np.asarray(icr0), zeros)).reshape((ydim, xdim))
            ocr0 = np.concatenate((np.asarray(ocr0), zeros)).reshape((ydim, xdim))
            i0 = np.concatenate((np.asarray(i0), zeros)).reshape((ydim, xdim))
            i1 = np.concatenate((np.asarray(i1), zeros)).reshape((ydim, xdim))
            tm = np.concatenate((np.asarray(tm), zeros)).reshape((ydim, xdim))
            del hf['raw/channel00/icr']
            del hf['raw/channel00/ocr']
            del hf['raw/I0']
            del hf['raw/I1']
            del hf['raw/acquisition_time']
            hf.create_dataset('raw/channel00/icr', data=icr0, compression='gzip', compression_opts=4)
            hf.create_dataset('raw/channel00/ocr', data=ocr0, compression='gzip', compression_opts=4)
            hf.create_dataset('raw/I0', data=i0, compression='gzip', compression_opts=4)
            hf.create_dataset('raw/I1', data=i1, compression='gzip', compression_opts=4)
            hf.create_dataset('raw/acquisition_time', data=tm, compression='gzip', compression_opts=4)
            mot1 = np.asarray(hf['mot1'])
            mot2 = np.asarray(hf['mot2'])
            if np.asarray(mot1).shape[0] < xdim*ydim:
                print("Warning: mot1 has less than "+str(xdim*ydim)+" elements; padding with zero.")
                mot1 = np.concatenate((np.asarray(mot1), np.zeros(xdim*ydim-np.asarray(mot1).shape[0])))
            if np.asarray(mot2).shape[0] < xdim*ydim:
                print("Warning: mot2 has less than "+str(xdim*ydim)+" elements; padding with zero.")
                mot2 = np.concatenate((np.asarray(mot2), np.zeros(xdim*ydim-np.asarray(mot2).shape[0])))
            mot1 = np.concatenate((np.asarray(mot1), zeros))[0:xdim*ydim].reshape((ydim, xdim))
            mot2 = np.concatenate((np.asarray(mot2), zeros))[0:xdim*ydim].reshape((ydim, xdim))
            del hf['mot1']
            del hf['mot2']
            dset = hf.create_dataset('mot1', data=mot1, compression='gzip', compression_opts=4)
            dset.attrs['Name'] = mot1_name
            dset = hf.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
            dset.attrs['Name'] = mot2_name
            if ch1 is not None:
                spectra1 = np.asarray(hf['raw/channel01/spectra'])
                zerosize = xdim*ydim-spectra1.shape[0]
                zeros = np.zeros((zerosize, spectra1.shape[1]))
                spectra1 = np.asarray(spectra1)
                spectra1 = np.concatenate((spectra1, zeros)).reshape((ydim, xdim, spectra1.shape[1]))
                del hf['raw/channel01/spectra']
                hf.create_dataset('raw/channel01/spectra', data=spectra1, compression='gzip', compression_opts=4)
                del spectra1 #clear it from memory, probably not really needed but no harm either
                icr1 = np.asarray(hf['raw/channel01/icr'])
                ocr1 = np.asarray(hf['raw/channel01/ocr'])
                zeros = np.zeros((zerosize))
                icr1 = np.concatenate((np.asarray(icr1), zeros)).reshape((ydim, xdim))
                ocr1 = np.concatenate((np.asarray(ocr1), zeros)).reshape((ydim, xdim))
                del hf['raw/channel01/icr']
                del hf['raw/channel01/ocr']
                hf.create_dataset('raw/channel01/icr', data=icr1, compression='gzip', compression_opts=4)
                hf.create_dataset('raw/channel01/ocr', data=ocr1, compression='gzip', compression_opts=4)
            timetrig = False
        else:            
            # in this case we should never sort or flip data
            sort = False
            timetrig = True
  
    # redefine as original arrays for further processing
    with h5py.File(scan_suffix+"_merge.h5", 'r', locking=True) as hf:
        spectra0 = np.asarray(hf['raw/channel00/spectra'])
        icr0 = np.asarray(hf['raw/channel00/icr'])
        ocr0 = np.asarray(hf['raw/channel00/ocr']) 
        i0 = np.asarray(hf['raw/I0'])
        i1 = np.asarray(hf['raw/I1'])
        tm = np.asarray(hf['raw/acquisition_time']) 
        mot1 = np.asarray(hf['mot1'])
        mot2 = np.asarray(hf['mot2'])

    # for continuous scans, the mot1 position runs in snake-type fashion
    #   so we need to sort the positions line per line and adjust all other data accordingly
    if sort is True:
        for i in range(mot1[:,0].size):
            sort_id = np.argsort(mot1[i,:])
            spectra0[i,:,:] = spectra0[i,sort_id,:]
            icr0[i,:] = icr0[i,sort_id]
            ocr0[i,:] = ocr0[i,sort_id]
            i0[i,:] = i0[i,sort_id]
            i1[i,:] = i1[i,sort_id]
            tm[i,:] = tm[i,sort_id]
    
        # To make sure (especially when merging scans) sort mot2 as well
        for i in range(mot2[0,:].size):
            sort_id = np.argsort(mot2[:,i])
            spectra0[:,i,:] = spectra0[sort_id,i,:]
            icr0[:,i] = icr0[sort_id,i]
            ocr0[:,i] = ocr0[sort_id,i]
            i0[:,i] = i0[sort_id,i]
            i1[:,i] = i1[sort_id,i]
            tm[:,i] = tm[sort_id,i]

    # calculate sumspec and maxspec spectra
    if timetrig is False:
        sumspec0 = np.sum(spectra0[:], axis=(0,1))
        maxspec0 = np.zeros(sumspec0.shape[0])
        for i in range(sumspec0.shape[0]):
            maxspec0[i] = spectra0[:,:,i].max()
    else:
        sumspec0 = np.sum(spectra0[:], axis=(0))
        maxspec0 = np.zeros(sumspec0.shape[0])
        for i in range(sumspec0.shape[0]):
            maxspec0[i] = spectra0[:,i].max()

    # Hooray! We read all the information! Let's write it to a separate file
    print("Writing merged file: "+scan_suffix+"_merge.h5...", end=" ")
    with h5py.File(scan_suffix+"_merge.h5", 'r+', locking=True) as hf:
        hf.create_dataset('cmd', data=' '.join(scan_cmd))
        del hf['raw/channel00/spectra']
        hf.create_dataset('raw/channel00/spectra', data=np.squeeze(spectra0), compression='gzip', compression_opts=4)
        del spectra0
        del hf['raw/channel00/icr']
        del hf['raw/channel00/ocr']
        del hf['raw/I0']
        del hf['raw/I1']
        del hf['raw/acquisition_time']
        hf.create_dataset('raw/channel00/icr', data=np.squeeze(icr0), compression='gzip', compression_opts=4)
        hf.create_dataset('raw/channel00/ocr', data=np.squeeze(ocr0), compression='gzip', compression_opts=4)
        hf.create_dataset('raw/channel00/sumspec', data=np.squeeze(sumspec0), compression='gzip', compression_opts=4)
        hf.create_dataset('raw/channel00/maxspec', data=np.squeeze(maxspec0), compression='gzip', compression_opts=4)
        hf.create_dataset('raw/I0', data=np.squeeze(i0), compression='gzip', compression_opts=4)
        hf.create_dataset('raw/I1', data=np.squeeze(i1), compression='gzip', compression_opts=4)
        hf.create_dataset('raw/acquisition_time', data=np.squeeze(tm), compression='gzip', compression_opts=4)

    # redefine as original arrays for further processing
    if ch1 is not None:
        with h5py.File(scan_suffix+"_merge.h5", 'r', locking=True) as hf:
            spectra1 = np.asarray(hf['raw/channel01/spectra'])
            icr1 = np.asarray(hf['raw/channel01/icr'])
            ocr1 = np.asarray(hf['raw/channel01/ocr']) 

    # for continuous scans, the mot1 position runs in snake-type fashion
    #   so we need to sort the positions line per line and adjust all other data accordingly
    if sort is True:
        for i in range(mot1[:,0].size):
            sort_id = np.argsort(mot1[i,:])
            if ch1 is not None:
                spectra1[i,:,:] = spectra1[i,sort_id,:]
                icr1[i,:] = icr1[i,sort_id]
                ocr1[i,:] = ocr1[i,sort_id]
            mot1[i,:] = mot1[i,sort_id]
            mot2[i,:] = mot2[i,sort_id]
    
        # To make sure (especially when merging scans) sort mot2 as well
        for i in range(mot2[0,:].size):
            sort_id = np.argsort(mot2[:,i])
            if ch1 is not None:
                spectra1[:,i,:] = spectra1[sort_id,i,:]
                icr1[:,i] = icr1[sort_id,i]
                ocr1[:,i] = ocr1[sort_id,i]
            mot1[:,i] = mot1[sort_id,i]
            mot2[:,i] = mot2[sort_id,i]

    # calculate sumspec and maxspec spectra
    if ch1 is not None:
        if timetrig is False:
            sumspec1 = np.sum(spectra1[:], axis=(0,1))
            maxspec1 = np.zeros(sumspec1.shape[0])
            for i in range(sumspec1.shape[0]):
                maxspec1[i] = spectra1[:,:,i].max()
        else:
            sumspec1 = np.sum(spectra1[:], axis=(0))
            maxspec1 = np.zeros(sumspec1.shape[0])
            for i in range(sumspec1.shape[0]):
                maxspec1[i] = spectra1[:,i].max()

    # Hooray! We read all the information! Let's write it to a separate file
    with h5py.File(scan_suffix+"_merge.h5", 'r+', locking=True) as hf:
        if ch1 is not None:
            del hf['raw/channel01/spectra']
            hf.create_dataset('raw/channel01/spectra', data=np.squeeze(spectra1), compression='gzip', compression_opts=4)
            del spectra1
            del hf['raw/channel01/icr']
            del hf['raw/channel01/ocr']
            hf.create_dataset('raw/channel01/icr', data=np.squeeze(icr1), compression='gzip', compression_opts=4)
            hf.create_dataset('raw/channel01/ocr', data=np.squeeze(ocr1), compression='gzip', compression_opts=4)
            hf.create_dataset('raw/channel01/sumspec', data=np.squeeze(sumspec1), compression='gzip', compression_opts=4)
            hf.create_dataset('raw/channel01/maxspec', data=np.squeeze(maxspec1), compression='gzip', compression_opts=4)
        del hf['mot1']
        del hf['mot2']
        dset = hf.create_dataset('mot1', data=np.squeeze(mot1), compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot1_name
        dset = hf.create_dataset('mot2', data=np.squeeze(mot2), compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot2_name
    print("ok")

##############################################################################
def ConvSoleilNxs(solnxs, mot1_name="COD_GONIO_Tz1", mot2_name="COD_GONIO_Ts2", ch0id=["channel00", "channel01"], ch1id=None, i0id="", i1id=None, icrid="icr", ocrid="ocr", tmid="realtime00", sort=True, outdir=None):
    '''
    Convert PUMA or NANOSCOPIUM Nexus format to our H5 structure file

    Parameters
    ----------
    solnxs : String or list of strings
        File path(s) of the PUMA Nexus file(s) to convert. When multiple are provided, the data is concatenated.
    mot1_name : string, optional
        Motor 1 identifier within the PUMA Nexus file. The default is 'COD_GONIO_Tz1'.
    mot2_name : String, optional
        Motor 2 identifier within the PUMA Nexus file. The default is 'COD_GONIO_Ts2'.
    ch0id : string, optional
        detector channel 0 identifier within the PUMA Nexus file. The default is ["channel00", "channel01"].
    ch1id : string, optional
        detector channel 1 identifier within the PUMA Nexus file. The default is None.
    i0id : string, optional
        I0 (incident beam flux) identifier within the PUMA Nexus file. The default is ''.
    i1id : string, optional
        I1 (transmitted beam flux) identifier within the PUMA Nexus file. The default is None.
    icrid : string, optional
        ICR identifier within the PUMA Nexus file. The same identifier label is used for multiple detectors. The default is 'icr'.
    ocrid : string, optional
        OCR identifier within the PUMA Nexus file. The same identifier label is used for multiple detectors. The default is 'ocr'.
    tmid : string, optional
        Measurement time identifier within the PUMA Nexus file. The default is "realtime00".
    sort : Boolean, optional
        If True the data is sorted following the motor encoder positions. The default is True.
    outdir : string, optional
        If None (default) the converted data is stored in the same directory as the raw data. If not None the data is stored in the directory provided by the user.

    Returns
    -------
    None.

    '''
    
    if type(solnxs) is not type([]):
        solnxs = [solnxs]
    if type(ch0id) is not type([]):
        ch0id = [ch0id]
    if ch1id is not None:
        if type(ch1id) is not type([]):
            ch1id = [ch1id]
    
        
    # puma filetype does not appear to have command line, so we'll just refer to the scan name itself
    scan_cmd = ' '.join([os.path.splitext(os.path.basename(nxs))[0] for nxs in solnxs]) 
    
    for index, nxs in enumerate(solnxs):
        print('Processing PUMA file '+nxs+'...', end='')
        if index == 0:
            with h5py.File(nxs,'r', locking=True) as f:
                if 'acq' in [key for key in f.keys()]:
                    basedir = "acq/scan_data/"
                    expflag = False
                elif 'exp' in [key for key in f.keys()]:
                    basedir = "exp/scan_data/"
                    expflag = True
                elif os.path.splitext(os.path.basename(nxs))[0].split('-')[0] in [key for key in f.keys()]:
                    basedir = os.path.splitext(os.path.basename(nxs))[0].split('-')[0]+"/scan_data/"
                    expflag = False
                mot1 = np.asarray(f[basedir+mot1_name])
                if mot1.ndim > 1:
                    mot2 = np.asarray(f[basedir+mot2_name])
                tm = np.asarray(f[basedir+tmid]) #realtime; there is one for each detector channel, but they should be (approx) the same value
                if expflag:
                    if mot1.ndim == 1:
                        mot2 = mot1*0.
                    else:
                        mot2 = np.array([mot2]*mot1.shape[1]).T
                        tm = np.array([tm]*mot1.shape[0])
                for i, chnl in enumerate(ch0id): #TODO: doesn't appear to make sense to check if i=0 or not, as same thing is done... later in code as well.
                    if i == 0:
                        spectra0 = np.asarray(f[basedir+chnl])
                        # icr and ocr id must be completed with last 2 digits from channel id
                        if expflag:
                            icr0 = np.asarray(f[basedir+icrid])
                            ocr0 = np.asarray(f[basedir+ocrid])
                        else:
                            icr0 = np.asarray(f[basedir+icrid+chnl[-2:]])
                            ocr0 = np.asarray(f[basedir+ocrid+chnl[-2:]])
                    else:
                        spectra0 = np.asarray(f[basedir+chnl])
                        # icr and ocr id must be completed with last 2 digits from channel id
                        if expflag:
                            icr0 = np.asarray(f[basedir+icrid])
                            ocr0 = np.asarray(f[basedir+ocrid])
                        else:
                            icr0 = np.asarray(f[basedir+icrid+chnl[-2:]])
                            ocr0 = np.asarray(f[basedir+ocrid+chnl[-2:]])
                 
                # at this time not certain there will be i0 or i1 data available. If not, just use 1-filled matrix of same size as mot1.
                if i0id != "":
                    i0 = np.asarray(f[basedir+i0id])
                else:
                    i0 = mot1*0.+1
                if i1id is not None:
                    i1 = np.asarray(f[basedir+i1id])
    
                if ch1id is not None:
                    for i, chnl in enumerate(ch1id):
                        if i == 0:
                            spectra1 = np.asarray(f[basedir+chnl])
                            # icr and ocr id must be completed with last 2 digits from channel id
                            if expflag:
                                icr1 = np.asarray(f[basedir+icrid])
                                ocr1 = np.asarray(f[basedir+ocrid])
                            else:
                                icr1 = np.asarray(f[basedir+icrid+chnl[-2:]])
                                ocr1 = np.asarray(f[basedir+ocrid+chnl[-2:]])
                        else:
                            spectra1 = np.asarray(f[basedir+chnl])
                            # icr and ocr id must be completed with last 2 digits from channel id
                            if expflag:
                                icr1 = np.asarray(f[basedir+icrid])
                                ocr1 = np.asarray(f[basedir+ocrid])
                            else:
                                icr1 = np.asarray(f[basedir+icrid+chnl[-2:]])
                                ocr1 = np.asarray(f[basedir+ocrid+chnl[-2:]])
        else:
           # figure out which motor axis dimension is identical to last scans, so that one can concatenate them.
            with h5py.File(nxs,'r', locking=True) as f:
                olddim = mot1.shape
                newdim = np.asarray(f[basedir+mot1_name]).shape
                if olddim[0] == newdim[0] and olddim[1] != newdim[1]:
                    axis=0
                elif olddim[0] != newdim[0] and olddim[1] == newdim[1]:
                    axis=1
                elif olddim[0] == newdim[0] and olddim[1] == newdim[1]:
                    # here have to monitor actual motorpositions to figure out which axis was moving...
                    if np.allclose(np.asarray(f[basedir+mot1_name]), mot1, 1E-4):
                        axis=0
                    else:
                        axis=1
                else:
                    axis=0
                mot1 = np.concatenate((mot1, np.asarray(f[basedir+mot1_name])), axis=axis)
                if mot1.ndim > 1:
                    mot2 = np.concatenate((mot2, np.asarray(f[basedir+mot2_name])), axis=axis)
                tm = np.concatenate((tm, np.asarray(f[basedir+tmid])), axis=axis) #realtime; there is one for each detector channel, but they should be (approx) the same value
                if expflag:
                    if mot1.ndim == 1:
                        mot2 = mot1*0.
                    else:
                        mot2 = np.array([mot2]*mot1.shape[1]).T
                        tm = np.array([tm]*mot1.shape[0])
                for i, chnl in enumerate(ch0id):
                    if i == 0:
                        spectra0 = np.concatenate((spectra0, np.asarray(f[basedir+chnl])), axis=axis)
                        # icr and ocr id must be completed with last 2 digits from channel id
                        icr0 = np.concatenate((icr0, np.asarray(f[basedir+icrid+chnl[-2:]])), axis=axis)
                        ocr0 = np.concatenate((ocr0, np.asarray(f[basedir+ocrid+chnl[-2:]])), axis=axis)
                    else:
                        spectra0 = np.concatenate((spectra0, np.asarray(f[basedir+chnl])), axis=axis)
                        # icr and ocr id must be completed with last 2 digits from channel id
                        icr0 = np.concatenate((icr0, np.asarray(f[basedir+icrid+chnl[-2:]])), axis=axis)
                        ocr0 = np.concatenate((ocr0, np.asarray(f[basedir+ocrid+chnl[-2:]])), axis=axis)
                 
                # at this time not certain there will be i0 or i1 data available. If not, just use 1-filled matrix of same size as mot1.
                if i0id != "":
                    i0 = np.concatenate((i0, np.asarray(f[basedir+i0id])), axis=axis)
                else:
                    i0 = np.concatenate((i0, mot1*0.+1), axis=axis)
                if i1id is not None:
                    i1 = np.concatenate((i1, np.asarray(f[basedir+i1id])), axis=axis)
                
                if ch1id is not None:
                    for i, chnl in enumerate(ch1id):
                        if i == 0:
                            spectra1 = np.concatenate((spectra1, np.asarray(f[basedir+chnl])), axis=axis)
                            # icr and ocr id must be completed with last 2 digits from channel id
                            if expflag:
                                icr1 = np.concatenate((icr1, np.asarray(f[basedir+icrid])), axis=axis)
                                ocr1 = np.concatenate((ocr1, np.asarray(f[basedir+ocrid])), axis=axis)
                            else:
                                icr1 = np.concatenate((icr1, np.asarray(f[basedir+icrid+chnl[-2:]])), axis=axis)
                                ocr1 = np.concatenate((ocr1, np.asarray(f[basedir+ocrid+chnl[-2:]])), axis=axis)
                        else:
                            spectra1 = np.concatenate((spectra1, np.asarray(f[basedir+chnl])), axis=axis)
                            # icr and ocr id must be completed with last 2 digits from channel id
                            if expflag:
                                icr1 = np.concatenate((icr1, np.asarray(f[basedir+icrid])), axis=axis)
                                ocr1 = np.concatenate((ocr1, np.asarray(f[basedir+ocrid])), axis=axis)
                            else:
                                icr1 = np.concatenate((icr1, np.asarray(f[basedir+icrid+chnl[-2:]])), axis=axis)
                                ocr1 = np.concatenate((ocr1, np.asarray(f[basedir+ocrid+chnl[-2:]])), axis=axis)
                

    # sort the positions line per line and adjust all other data accordingly
    if sort is True:
        for i in range(mot2[:,0].size):
            sort_id = np.argsort(mot2[i,:])
            spectra0[i,:,:] = spectra0[i,sort_id,:]
            icr0[i,:] = icr0[i,sort_id]
            ocr0[i,:] = ocr0[i,sort_id]
            if ch1id is not None:
                	spectra1[i,:,:] = spectra1[i,sort_id,:]
                	icr1[i,:] = icr1[i,sort_id]
                	ocr1[i,:] = ocr1[i,sort_id]
            mot1[i,:] = mot1[i,sort_id]
            mot2[i,:] = mot2[i,sort_id]
            i0[i,:] = i0[i,sort_id]
            if i1id is not None:
                	i1[i,:] = i1[i,sort_id]
            tm[i,:] = tm[i,sort_id]
    
        # To make sure (especially when merging scans) sort mot2 as well
        for i in range(mot1[0,:].size):
            sort_id = np.argsort(mot1[:,i])
            spectra0[:,i,:] = spectra0[sort_id,i,:]
            icr0[:,i] = icr0[sort_id,i]
            ocr0[:,i] = ocr0[sort_id,i]
            if ch1id is not None:
                spectra1[:,i,:] = spectra1[sort_id,i,:]
                icr1[:,i] = icr1[sort_id,i]
                ocr1[:,i] = ocr1[sort_id,i]
            mot1[:,i] = mot1[sort_id,i]
            mot2[:,i] = mot2[sort_id,i]
            i0[:,i] = i0[sort_id,i]
            if i1id is not None:
                i1[:,i] = i1[sort_id,i]
            tm[:,i] = tm[sort_id,i]
        
    
    # calculate maxspec and sumspec
    if spectra0.ndim == 3:
        sumspec0 = np.sum(spectra0[:], axis=(0,1))
        maxspec0 = np.zeros(sumspec0.shape[0])
        for i in range(sumspec0.shape[0]):
            maxspec0[i] = spectra0[:,:,i].max()
        if ch1id is not None:
            sumspec1 = np.sum(spectra1[:], axis=(0,1))
            maxspec1 = np.zeros(sumspec1.shape[0])
            for i in range(sumspec1.shape[0]):
                maxspec1[i] = spectra1[:,:,i].max()
    elif spectra0.ndim == 2:
        scan_cmd = 'dscan '+scan_cmd
        sumspec0 = np.sum(spectra0[:], axis=(0))
        maxspec0 = np.zeros(sumspec0.shape[0])
        for i in range(sumspec0.shape[0]):
            maxspec0[i] = spectra0[:,i].max()
        if ch1id is not None:
            sumspec1 = np.sum(spectra1[:], axis=(0))
            maxspec1 = np.zeros(sumspec1.shape[0])
            for i in range(sumspec1.shape[0]):
                maxspec1[i] = spectra1[:,i].max()    
    
    # write h5 file in our structure
    if outdir is None:
        filename = solnxs[0].split(".")[0]+'_conv.h5'
    else:
        if not outdir.endswith('/'):
            outdir += '/'
        filename = outdir+os.path.splitext(os.path.basename(solnxs[0]))[0]+'_conv.h5'

    with h5py.File(filename, 'w', locking=True) as f:
        f.create_dataset('cmd', data=scan_cmd)
        f.create_dataset('raw/channel00/spectra', data=spectra0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/icr', data=icr0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/ocr', data=ocr0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/sumspec', data=sumspec0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/maxspec', data=maxspec0, compression='gzip', compression_opts=4)
        if ch1id is not None:
            f.create_dataset('raw/channel01/spectra', data=spectra1, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel01/icr', data=icr1, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel01/ocr', data=ocr1, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel01/sumspec', data=sumspec1, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel01/maxspec', data=maxspec1, compression='gzip', compression_opts=4)
        f.create_dataset('raw/I0', data=i0, compression='gzip', compression_opts=4)
        if i1id is not None:
            f.create_dataset('raw/I1', data=i1, compression='gzip', compression_opts=4)
        dset = f.create_dataset('mot1', data=mot1, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot1_name
        dset = f.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot2_name
        f.create_dataset('raw/acquisition_time', data=tm, compression='gzip', compression_opts=4)
    print("Done")
    
##############################################################################
def ConvID15H5(h5id15, scanid, scan_dim, mot1_name='hry', mot2_name='hrz', ch0id='falconx_det0', ch1id='falconx2_det0', i0id='fpico2', i0corid=None, i1id='fpico3', i1corid=None, icrid='trigger_count_rate', ocrid='event_count_rate', atol=None, sort=True, outdir=None):
    """
    Convert ID15A bliss H5 format to our H5 structure file
    

    Parameters
    ----------
    h5id15 : String
        File path of the bliss H5 file to convert.
    scanid : (list of) String(s)
        Scan identifier within the bliss H5 file, e.g. '1.1'. When scanid is an array or list of multiple elements, 
        the images will be stitched together to 1 file.
    scan_dim : tuple
        The scan dimensions of the data contained within the bliss H5 file.
    mot1_name : string, optional
        Motor 1 identifier within the bliss H5 file. The default is 'hry'.
    mot2_name : String, optional
        Motor 2 identifier within the bliss H5 file. The default is 'hrz'.
    ch0id : string, optional
        detector channel 0 identifier within the bliss H5 file. The default is 'falconx_det0'.
    ch1id : string, optional
        detector channel 1 identifier within the bliss H5 file. The default is 'falconx2_det0'.
    i0id : string, optional
        I0 (incident beam flux) identifier within the bliss H5 file. The default is 'fpico2'.
    i0corid : string, optional
        I0 signal correction signal identifier within the bliss H5 file. The default is None.
    i1id : string, optional
        I1 (transmitted beam flux) identifier within the bliss H5 file. The default is 'fpico3'.
    i1corid : string, optional
        I1 signal correction signal identifier within the bliss H5 file. The default is None.
    icrid : string, optional
        ICR identifier within the bliss H5 file. The same identifier label is used for multiple detectors. The default is 'trigger_count_rate'.
    ocrid : string, optional
        OCR identifier within the bliss H5 file. The same identifier label is used for multiple detectors. The default is 'event_count_rate'.
    atol : float, optional
        Absolute tolerance used by numpy.allclose() when comparing motor coordinate positions to determine the motor position array incremental direction.
        The default is None, indicating a value of 1e-4.
    sort : Boolean, optional
        If True the data is sorted following the motor encoder positions. The default is True.
    outdir : string, optional
        If None (default) the converted data is stored in the same directory as the raw data. If not None the data is stored in the directory provided by the user.

    Returns
    -------
    bool
        Returns False upon failure.

    """
    scan_dim = np.asarray(scan_dim)
    scanid = np.asarray(scanid)
    if scan_dim.size == 1:
        scan_dim = np.asarray((scan_dim, 1))
    if atol is None:
        atol = 1e-4

    if scanid.size == 1:
        scan_suffix = '_scan'+str(scanid).split(".")[0]
    else:
        scan_suffix = '_scan'+str(scanid[0]).split(".")[0]+'-'+str(scanid[-1]).split(".")[0]

    if np.asarray(h5id15).size == 1:
        h5id15 = [h5id15]*scanid.size
    else: # assumes we have same amount of scan nrs as h5id15 files!
        lasth5 = h5id15[-1].split('/')[-1].split('.')[0]
        scan_suffix = '_scan'+str(scanid[0]).split(".")[0]+'-'+lasth5+'_scan'+str(scanid[-1]).split(".")[0]

    print('Processing id15 file '+h5id15[0]+'...', end='')
    # read h5id15 file(s)
    for j in range(0, scanid.size):
        if scanid.size == 1:
            sc_id = str(scanid)
            sc_dim = scan_dim
            file = h5id15[0]
        else:
            sc_id = str(scanid[j])
            file = h5id15[j]
            if scan_dim.size > 2:
                sc_dim = scan_dim[j]
            else:
                sc_dim = scan_dim
        if j == 0:
            with h5py.File(file, 'r', locking=True) as f:
                try:
                    scan_cmd = f[sc_id+'/title'][()].decode('utf8')
                except Exception:
                    scan_cmd = f[sc_id+'/title'][()]
                spectra0 = np.asarray(f[sc_id+'/measurement/'+ch0id][:sc_dim[0]*sc_dim[1],:]).reshape((sc_dim[0], sc_dim[1], -1))
                icr0 = np.asarray(f[sc_id+'/measurement/'+ch0id+'_'+icrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                ocr0 = np.asarray(f[sc_id+'/measurement/'+ch0id+'_'+ocrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                if ch1id is not None:
                    spectra1 = np.asarray(f[sc_id+'/measurement/'+ch1id][:sc_dim[0]*sc_dim[1],:]).reshape((sc_dim[0], sc_dim[1], -1))
                    icr1 = np.asarray(f[sc_id+'/measurement/'+ch1id+'_'+icrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                    ocr1 = np.asarray(f[sc_id+'/measurement/'+ch1id+'_'+ocrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                i0 = np.asarray(f[sc_id+'/measurement/'+i0id][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                if 'current' in f[sc_id+'/instrument/machine/'].keys():
                    i0 = i0/np.asarray(f[sc_id+'/instrument/machine/current'])
                if i0corid is not None:
                    try:
                        i0cor = np.average(np.asarray(f[str(scanid).split('.')[0]+'.2/measurement/'+i0corid][:]))
                    except KeyError:
                        try:
                            i0cor = np.average(np.asarray(f[str(scanid)+'/instrument/'+i0corid+'/data'][:]))
                        except KeyError:
                            print("***ERROR: no viable i0cor value obtained. Set to 1.")
                            i0cor = 1.
                    i0 = i0/np.average(i0) * i0cor
                if i1id is not None:
                    i1 = np.asarray(f[sc_id+'/measurement/'+i1id][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                    if 'current' in f[sc_id+'/instrument/machine/'].keys():
                        i1 = i1/np.asarray(f[sc_id+'/instrument/machine/current'])
                    if i1corid is not None:
                        try:
                            i1cor = np.average(np.asarray(f[str(scanid).split('.')[0]+'.2/measurement/'+i1corid]))
                        except KeyError:
                            try:
                                i1cor = np.average(np.asarray(f[str(scanid)+'/instrument/'+i1corid+'/data']))
                            except KeyError:
                                print("***ERROR: no viable i1cor value obtained. Set to 1.")
                                i1cor = 1.
                        i1 = i1/np.average(i1) * i1cor
                try:
                    mot1 = np.asarray(f[sc_id+'/measurement/'+mot1_name][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                    mot2 = np.asarray(f[sc_id+'/measurement/'+mot2_name][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                except KeyError:
                    mot1 = np.asarray(f[sc_id+'/instrument/positioners/'+mot1_name][()]).reshape(sc_dim)
                    mot2 = np.asarray(f[sc_id+'/instrument/positioners/'+mot2_name][()]).reshape(sc_dim)
                tm = np.asarray(f[sc_id+'/measurement/'+ch0id+'_elapsed_time'][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)

            # find direction in which mot1 and mot2 increase
            if mot1.size > 1:
                if np.allclose(mot1[0,:], mot1[1,:], atol=atol):
                    mot1_id = 1
                else:
                    mot1_id = 0
            else:
                mot1_id = 0
            if mot2.size > 1:
                if np.allclose(mot2[0,:], mot2[1,:], atol=atol):
                    mot2_id = 1
                else:
                    mot2_id = 0
            else:
                mot2_id = 0

        # when reading second file, we should stitch it to the first file's image
        else:
            with h5py.File(file, 'r', locking=True) as f:
                try:
                    scan_cmd += ' '+f[sc_id+'/title'][()].decode('utf8')
                except Exception:
                    scan_cmd += ' '+f[sc_id+'/title'][()]
                    
                #the other arrays we can't simply append: have to figure out which side to stitch them to, and if there is overlap between motor positions
                spectra0_temp = np.asarray(f[sc_id+'/measurement/'+ch0id][:sc_dim[0]*sc_dim[1],:]).reshape((sc_dim[0], sc_dim[1], -1))
                icr0_temp = np.asarray(f[sc_id+'/measurement/'+ch0id+'_'+icrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                ocr0_temp = np.asarray(f[sc_id+'/measurement/'+ch0id+'_'+ocrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                if ch1id is not None:
                    spectra1_temp = np.asarray(f[sc_id+'/measurement/'+ch1id][:sc_dim[0]*sc_dim[1],:]).reshape((sc_dim[0], sc_dim[1], -1))
                    icr1_temp = np.asarray(f[sc_id+'/measurement/'+ch1id+'_'+icrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                    ocr1_temp = np.asarray(f[sc_id+'/measurement/'+ch1id+'_'+ocrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                i0_temp = np.asarray(f[sc_id+'/measurement/'+i0id][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                if 'current' in f[sc_id+'/instrument/machine/'].keys():
                    i0_temp = i0_temp/np.asarray(f[sc_id+'/instrument/machine/current'][:])
                if i0corid is not None:
                    try:
                        i0cor = np.average(np.asarray(f[str(scanid).split('.')[0]+'.2/measurement/'+i0corid][:]))
                    except KeyError:
                        try:
                            i0cor = np.average(np.asarray(f[str(scanid)+'/instrument/'+i0corid+'/data'][:]))
                        except KeyError:
                            print("***ERROR: no viable i0cor value obtained. Set to 1.")
                            i0cor = 1.
                    i0_temp = i0_temp/np.average(i0_temp) * i0cor
                if i1id is not None:
                    i1_temp = np.asarray(f[sc_id+'/measurement/'+i1id][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                    if 'current' in f[sc_id+'/instrument/machine/'].keys():
                        i1_temp = i1_temp/np.asarray(f[sc_id+'/instrument/machine/current'][:])
                    if i1corid is not None:
                        try:
                            i1cor = np.average(np.asarray(f[str(scanid).split('.')[0]+'.2/measurement/'+i1corid][:]))
                        except KeyError:
                            try:
                                i1cor = np.average(np.asarray(f[str(scanid)+'/instrument/'+i1corid+'/data'][:]))
                            except KeyError:
                                print("***ERROR: no viable i1cor value obtained. Set to 1.")
                                i1cor = 1.
                        i1_temp = i1_temp/np.average(i1_temp) * i1cor
                try:
                    mot1_temp = np.asarray(f[sc_id+'/measurement/'+mot1_name][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                    mot2_temp = np.asarray(f[sc_id+'/measurement/'+mot2_name][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                except KeyError:
                    mot1_temp = np.asarray(f[sc_id+'/instrument/positioners/'+mot1_name][()]).reshape(sc_dim)
                    mot2_temp = np.asarray(f[sc_id+'/instrument/positioners/'+mot2_name][()]).reshape(sc_dim)
                tm_temp = np.asarray(f[sc_id+'/measurement/'+ch0id+'_elapsed_time'][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)

            mot_flag = 0 #TODO: if mot1_temp.shape is larger than mot1 it crashes...
            if mot1_id == 0:
                if mot1_temp[:,0].shape[0] >= mot1[:,0].shape[0]:
                    mot1lim = mot1[:,0].shape[0]
                else:
                    mot1lim = mot1_temp[:,0].shape[0]
                if mot2_temp[:,0].shape[0] >= mot2[:,0].shape[0]:
                    mot2lim = mot2[:,0].shape[0]
                else:
                    mot2lim = mot2_temp[:,0].shape[0]
            if mot1_id == 1:
                if mot1_temp[0,:].shape[0] >= mot1[0,:].shape[0]:
                    mot1lim = mot1[0,:].shape[0]
                else:
                    mot1lim = mot1_temp[0,:].shape[0]
                if mot2_temp[0,:].shape[0] >= mot2[0,:].shape[0]:
                    mot2lim = mot2[0,:].shape[0]
                else:
                    mot2lim = mot2_temp[0,:].shape[0]
            if mot1_id == 1 and not np.allclose(mot1[0,(mot1[0,:].shape[0]-mot1lim):], mot1_temp[0,:mot1lim], atol=atol):
                    mot_flag = 1
                    # print('Here1',end=' ')
            elif mot1_id == 0 and not np.allclose(mot1[(mot1[:,0].shape[0]-mot1lim):,0], mot1_temp[:mot1lim,0], atol=atol):
                    mot_flag = 1
                    # print('Here2',end=' ')
            elif mot2_id == 1 and not np.allclose(mot2[0,(mot2[0,:].shape[0]-mot2lim):], mot2_temp[0,:mot2lim], atol=atol):
                    mot_flag = 2
                    # print('Here3',end=' ')
            elif mot2_id == 0 and not np.allclose(mot2[(mot2[:,0].shape[0]-mot2lim):,0], mot2_temp[:mot2lim,0], atol=atol):
                    mot_flag = 2
                    # print('Here4')
                    print(mot2[(mot2[:,0].shape[0]-mot2_temp[:,0].shape[0]):,0])
                    print(mot2_temp[:,0])
            
            # check if several regions have identical mot1 or mot2 positions
            if mot_flag == 2:
                # as mot1 and mot1_temp are identical, it must be that mot2 changes
                if mot2.max() < mot2_temp.min():
                    spectra0 = np.concatenate((spectra0, spectra0_temp),axis=mot1_id)
                    icr0 = np.concatenate((icr0, icr0_temp), axis=mot1_id)
                    ocr0 = np.concatenate((ocr0, ocr0_temp), axis=mot1_id)
                    if ch1id is not None:
                        spectra1 = np.concatenate((spectra1, spectra1_temp),axis=mot1_id)
                        icr1 = np.concatenate((icr1, icr1_temp), axis=mot1_id)
                        ocr1 = np.concatenate((ocr1, ocr1_temp), axis=mot1_id)
                    i0 = np.concatenate((i0, i0_temp), axis=mot1_id)
                    if i1id is not None:
                        i1 = np.concatenate((i1, i1_temp), axis=mot1_id)
                    mot1 = np.concatenate((mot1, mot1_temp), axis=mot1_id)
                    mot2 = np.concatenate((mot2, mot2_temp), axis=mot1_id)
                    tm = np.concatenate((tm, tm_temp), axis=mot1_id)
                elif mot2_temp.max() < mot2.min():
                    # mot2_temp should come before mot2
                    spectra0 = np.concatenate((spectra0_temp, spectra0),axis=mot1_id)
                    icr0 = np.concatenate((icr0_temp, icr0), axis=mot1_id)
                    ocr0 = np.concatenate((ocr0_temp, ocr0), axis=mot1_id)
                    if ch1id is not None:
                        spectra1 = np.concatenate((spectra1_temp, spectra1),axis=mot1_id)
                        icr1 = np.concatenate((icr1_temp, icr1), axis=mot1_id)
                        ocr1 = np.concatenate((ocr1_temp, ocr1), axis=mot1_id)
                    i0 = np.concatenate((i0_temp, i0), axis=mot1_id)
                    if i1id is not None:
                        i1 = np.concatenate((i1_temp, i1), axis=mot1_id)
                    mot1 = np.concatenate((mot1_temp, mot1), axis=mot1_id)
                    mot2 = np.concatenate((mot2_temp, mot2), axis=mot1_id)
                    tm = np.concatenate((tm_temp, tm), axis=mot1_id)
                else:
                    # there is some overlap between mot2 and mot2_temp; figure out where it overlaps and stitch like that
                    #TODO: there is the case where the new slice could fit entirely within the old one...
                    if mot2.min() < mot2_temp.min():
                        # mot2 should come first, followed by mot2_temp
                        if mot2_id == 0:
                            keep_id = np.asarray(np.where(mot2[:,0] < mot2_temp.min())).max()+1 #add one as we also need last element of id's
                            spectra0 = np.concatenate((spectra0[0:keep_id,:,:], spectra0_temp),axis=mot1_id)
                            icr0 = np.concatenate((icr0[0:keep_id,:], icr0_temp), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0[0:keep_id,:], ocr0_temp), axis=mot1_id)
                            if ch1id is not None:
                                spectra1 = np.concatenate((spectra1[0:keep_id,:,:], spectra1_temp),axis=mot1_id)
                                icr1 = np.concatenate((icr1[0:keep_id,:], icr1_temp), axis=mot1_id)
                                ocr1 = np.concatenate((ocr1[0:keep_id,:], ocr1_temp), axis=mot1_id)
                            i0 = np.concatenate((i0[0:keep_id,:], i0_temp), axis=mot1_id)
                            if i1id is not None:
                                i1 = np.concatenate((i1[0:keep_id,:], i1_temp), axis=mot1_id)
                            mot1 = np.concatenate((mot1[0:keep_id,:], mot1_temp), axis=mot1_id)
                            mot2 = np.concatenate((mot2[0:keep_id,:], mot2_temp), axis=mot1_id)
                            tm = np.concatenate((tm[0:keep_id,:], tm_temp), axis=mot1_id)
                        else:
                            keep_id = np.asarray(np.where(mot2[0,:] < mot2_temp.min())).max()+1 #add one as we also need last element of id's
                            spectra0 = np.concatenate((spectra0[:,0:keep_id,:], spectra0_temp),axis=mot1_id)
                            icr0 = np.concatenate((icr0[:,0:keep_id], icr0_temp), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0[:,0:keep_id], ocr0_temp), axis=mot1_id)
                            if ch1id is not None:
                                spectra1 = np.concatenate((spectra1[:,0:keep_id,:,:], spectra1_temp),axis=mot1_id)
                                icr1 = np.concatenate((icr1[:,0:keep_id], icr1_temp), axis=mot1_id)
                                ocr1 = np.concatenate((ocr1[:,0:keep_id], ocr1_temp), axis=mot1_id)
                            i0 = np.concatenate((i0[:,0:keep_id], i0_temp), axis=mot1_id)
                            if i1id is not None:
                                i1 = np.concatenate((i1[:,0:keep_id], i1_temp), axis=mot1_id)
                            mot1 = np.concatenate((mot1[:,0:keep_id], mot1_temp), axis=mot1_id)
                            mot2 = np.concatenate((mot2[:,0:keep_id], mot2_temp), axis=mot1_id)
                            tm = np.concatenate((tm[:,0:keep_id], tm_temp), axis=mot1_id)
                    else:
                        # first mot2_temp, followed by remainder of mot2 (where no more overlap)
                        keep_id = np.asarray(np.where(mot2_temp[mot2_id] < mot2.min())).max()+1 #add one as we also need last element of id's
                        if mot2_id == 0:
                            spectra0 = np.concatenate((spectra0_temp[:,0:keep_id,:], spectra0),axis=mot1_id)
                            icr0 = np.concatenate((icr0_temp[:,0:keep_id], icr0), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0_temp[:,0:keep_id], ocr0), axis=mot1_id)
                            if ch1id is not None:
                                spectra1 = np.concatenate((spectra1_temp[:,0:keep_id,:,:], spectra1),axis=mot1_id)
                                icr1 = np.concatenate((icr1_temp[:,0:keep_id], icr1), axis=mot1_id)
                                ocr1 = np.concatenate((ocr1_temp[:,0:keep_id], ocr1), axis=mot1_id)
                            i0 = np.concatenate((i0_temp[:,0:keep_id], i0), axis=mot1_id)
                            if i1id is not None:
                                i1 = np.concatenate((i1_temp[:,0:keep_id], i1), axis=mot1_id)
                            mot1 = np.concatenate((mot1_temp[:,0:keep_id], mot1), axis=mot1_id)
                            mot2 = np.concatenate((mot2_temp[:,0:keep_id], mot2), axis=mot1_id)
                            tm = np.concatenate((tm_temp[:,0:keep_id], tm), axis=mot1_id)
                        else:
                            spectra0 = np.concatenate((spectra0_temp[:,0:keep_id,:], spectra0),axis=mot1_id)
                            icr0 = np.concatenate((icr0_temp[:,0:keep_id], icr0), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0_temp[:,0:keep_id], ocr0), axis=mot1_id)
                            if ch1id is not None:
                                spectra1 = np.concatenate((spectra1_temp[:,0:keep_id,:,:], spectra1),axis=mot1_id)
                                icr1 = np.concatenate((icr1_temp[:,0:keep_id], icr1), axis=mot1_id)
                                ocr1 = np.concatenate((ocr1_temp[:,0:keep_id], ocr1), axis=mot1_id)
                            i0 = np.concatenate((i0_temp[:,0:keep_id], i0), axis=mot1_id)
                            if i1id is not None:
                                i1 = np.concatenate((i1_temp[:,0:keep_id], i1), axis=mot1_id)
                            mot1 = np.concatenate((mot1_temp[:,0:keep_id], mot1), axis=mot1_id)
                            mot2 = np.concatenate((mot2_temp[:,0:keep_id], mot2), axis=mot1_id)
                            tm = np.concatenate((tm_temp[:,0:keep_id], tm), axis=mot1_id)
            elif mot_flag == 1:
                # as mot2 and mot2_temp are identical, it must be that mot1 changes
                if mot1.max() < mot1_temp.min():
                    spectra0 = np.concatenate((spectra0, spectra0_temp),axis=mot1_id)
                    icr0 = np.concatenate((icr0, icr0_temp), axis=mot1_id)
                    ocr0 = np.concatenate((ocr0, ocr0_temp), axis=mot1_id)
                    if ch1id is not None:
                        spectra1 = np.concatenate((spectra1, spectra1_temp),axis=mot1_id)
                        icr1 = np.concatenate((icr1, icr1_temp), axis=mot1_id)
                        ocr1 = np.concatenate((ocr1, ocr1_temp), axis=mot1_id)
                    i0 = np.concatenate((i0, i0_temp), axis=mot1_id)
                    if i1id is not None:
                        i1 = np.concatenate((i1, i1_temp), axis=mot1_id)
                    mot1 = np.concatenate((mot1, mot1_temp), axis=mot1_id)
                    mot2 = np.concatenate((mot2, mot2_temp), axis=mot1_id)
                    tm = np.concatenate((tm, tm_temp), axis=mot1_id)
                elif mot1_temp.max() < mot1.min():
                    spectra0 = np.concatenate((spectra0_temp, spectra0),axis=mot1_id)
                    icr0 = np.concatenate((icr0_temp, icr0), axis=mot1_id)
                    ocr0 = np.concatenate((ocr0_temp, ocr0), axis=mot1_id)
                    if ch1id is not None:
                        spectra1 = np.concatenate((spectra1_temp, spectra1),axis=mot1_id)
                        icr1 = np.concatenate((icr1_temp, icr1), axis=mot1_id)
                        ocr1 = np.concatenate((ocr1_temp, ocr1), axis=mot1_id)
                    i0 = np.concatenate((i0_temp, i0), axis=mot1_id)
                    if i1id is not None:
                        i1 = np.concatenate((i1_temp, i1), axis=mot1_id)
                    mot1 = np.concatenate((mot1_temp, mot1), axis=mot1_id)
                    mot2 = np.concatenate((mot2_temp, mot2), axis=mot1_id)
                    tm = np.concatenate((tm_temp, tm), axis=mot1_id)
                else:
                    # there is some overlap between mot1 and mot1_temp; figure out where it overlaps and stitch like that
                    #TODO: there is the case where the new slice could fit entirely within the old one...
                    if mot1.min() < mot1_temp.min():
                        # mot1 should come first, followed by mot1_temp
                        if mot1_id == 0:
                            keep_id = np.asarray(np.where(mot1[:,0] < mot1_temp.min())).max()+1 #add one as we also need last element of id's
                            spectra0 = np.concatenate((spectra0[0:keep_id,:,:], spectra0_temp),axis=mot1_id)
                            icr0 = np.concatenate((icr0[0:keep_id,:], icr0_temp), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0[0:keep_id,:], ocr0_temp), axis=mot1_id)
                            if ch1id is not None:
                                spectra1 = np.concatenate((spectra1[0:keep_id,:,:], spectra1_temp),axis=mot1_id)
                                icr1 = np.concatenate((icr1[0:keep_id,:], icr1_temp), axis=mot1_id)
                                ocr1 = np.concatenate((ocr1[0:keep_id,:], ocr1_temp), axis=mot1_id)
                            i0 = np.concatenate((i0[0:keep_id,:], i0_temp), axis=mot1_id)
                            if i1id is not None:
                                i1 = np.concatenate((i1[0:keep_id,:], i1_temp), axis=mot1_id)
                            mot1 = np.concatenate((mot1[0:keep_id,:], mot1_temp), axis=mot1_id)
                            mot2 = np.concatenate((mot2[0:keep_id,:], mot2_temp), axis=mot1_id)
                            tm = np.concatenate((tm[0:keep_id,:], tm_temp), axis=mot1_id)
                        else:
                            keep_id = np.asarray(np.where(mot1[0,:] < mot1_temp.min())).max()+1 #add one as we also need last element of id's
                            spectra0 = np.concatenate((spectra0[:,0:keep_id,:], spectra0_temp),axis=mot1_id)
                            icr0 = np.concatenate((icr0[:,0:keep_id], icr0_temp), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0[:,0:keep_id], ocr0_temp), axis=mot1_id)
                            if ch1id is not None:
                                spectra1 = np.concatenate((spectra1[:,0:keep_id,:,:], spectra1_temp),axis=mot1_id)
                                icr1 = np.concatenate((icr1[:,0:keep_id], icr1_temp), axis=mot1_id)
                                ocr1 = np.concatenate((ocr1[:,0:keep_id], ocr1_temp), axis=mot1_id)
                            i0 = np.concatenate((i0[:,0:keep_id], i0_temp), axis=mot1_id)
                            if i1id is not None:
                                i1 = np.concatenate((i1[:,0:keep_id], i1_temp), axis=mot1_id)
                            mot1 = np.concatenate((mot1[:,0:keep_id], mot1_temp), axis=mot1_id)
                            mot2 = np.concatenate((mot2[:,0:keep_id], mot2_temp), axis=mot1_id)
                            tm = np.concatenate((tm[:,0:keep_id], tm_temp), axis=mot1_id)
                    else:
                        # first mot1_temp, followed by remainder of mot1 (where no more overlap)
                        if mot1_id == 0:
                            keep_id = np.asarray(np.where(mot1_temp[:,0] < mot1.min())).max()+1 #add one as we also need last element of id's
                            spectra0 = np.concatenate((spectra0_temp[0:keep_id,:,:], spectra0),axis=mot1_id)
                            icr0 = np.concatenate((icr0_temp[0:keep_id,:], icr0), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0_temp[0:keep_id,:], ocr0), axis=mot1_id)
                            if ch1id is not None:
                                spectra1 = np.concatenate((spectra1_temp[0:keep_id,:,:], spectra1),axis=mot1_id)
                                icr1 = np.concatenate((icr1_temp[0:keep_id,:], icr1), axis=mot1_id)
                                ocr1 = np.concatenate((ocr1_temp[0:keep_id,:], ocr1), axis=mot1_id)
                            i0 = np.concatenate((i0_temp[0:keep_id,:], i0), axis=mot1_id)
                            if i1id is not None:
                                i1 = np.concatenate((i1_temp[0:keep_id,:], i1), axis=mot1_id)
                            mot1 = np.concatenate((mot1_temp[0:keep_id,:], mot1), axis=mot1_id)
                            mot2 = np.concatenate((mot2_temp[0:keep_id,:], mot2), axis=mot1_id)
                            tm = np.concatenate((tm_temp[0:keep_id,:], tm), axis=mot1_id)
                        else:
                            keep_id = np.asarray(np.where(mot1_temp[0,:] < mot1.min())).max()+1 #add one as we also need last element of id's
                            spectra0 = np.concatenate((spectra0_temp[:,0:keep_id,:], spectra0),axis=mot1_id)
                            icr0 = np.concatenate((icr0_temp[:,0:keep_id], icr0), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0_temp[:,0:keep_id], ocr0), axis=mot1_id)
                            if ch1id is not None:
                                spectra1 = np.concatenate((spectra1_temp[:,0:keep_id,:,:], spectra1),axis=mot1_id)
                                icr1 = np.concatenate((icr1_temp[:,0:keep_id], icr1), axis=mot1_id)
                                ocr1 = np.concatenate((ocr1_temp[:,0:keep_id], ocr1), axis=mot1_id)
                            i0 = np.concatenate((i0_temp[:,0:keep_id], i0), axis=mot1_id)
                            if i1id is not None:
                                i1 = np.concatenate((i1_temp[:,0:keep_id], i1), axis=mot1_id)
                            mot1 = np.concatenate((mot1_temp[:,0:keep_id], mot1), axis=mot1_id)
                            mot2 = np.concatenate((mot2_temp[:,0:keep_id], mot2), axis=mot1_id)
                            tm = np.concatenate((tm_temp[:,0:keep_id], tm), axis=mot1_id)
            else:
                print("Error: all motor positions are identical within 1e-4.")
                return False
        
    # sort the positions line per line and adjust all other data accordingly
    if sort is True:
        for i in range(mot2[:,0].size):
            sort_id = np.argsort(mot2[i,:])
            spectra0[i,:,:] = spectra0[i,sort_id,:]
            icr0[i,:] = icr0[i,sort_id]
            ocr0[i,:] = ocr0[i,sort_id]
            if ch1id is not None:
                	spectra1[i,:,:] = spectra1[i,sort_id,:]
                	icr1[i,:] = icr1[i,sort_id]
                	ocr1[i,:] = ocr1[i,sort_id]
            mot1[i,:] = mot1[i,sort_id]
            mot2[i,:] = mot2[i,sort_id]
            i0[i,:] = i0[i,sort_id]
            if i1id is not None:
                	i1[i,:] = i1[i,sort_id]
            tm[i,:] = tm[i,sort_id]
    
        # To make sure (especially when merging scans) sort mot2 as well
        for i in range(mot1[0,:].size):
            sort_id = np.argsort(mot1[:,i])
            spectra0[:,i,:] = spectra0[sort_id,i,:]
            icr0[:,i] = icr0[sort_id,i]
            ocr0[:,i] = ocr0[sort_id,i]
            if ch1id is not None:
                spectra1[:,i,:] = spectra1[sort_id,i,:]
                icr1[:,i] = icr1[sort_id,i]
                ocr1[:,i] = ocr1[sort_id,i]
            mot1[:,i] = mot1[sort_id,i]
            mot2[:,i] = mot2[sort_id,i]
            i0[:,i] = i0[sort_id,i]
            if i1id is not None:
                i1[:,i] = i1[sort_id,i]
            tm[:,i] = tm[sort_id,i]

    # calculate maxspec and sumspec
    sumspec0 = np.sum(spectra0[:], axis=(0,1))
    maxspec0 = np.zeros(sumspec0.shape[0])
    for i in range(sumspec0.shape[0]):
        maxspec0[i] = spectra0[:,:,i].max()
    if ch1id is not None:
        sumspec1 = np.sum(spectra1[:], axis=(0,1))
        maxspec1 = np.zeros(sumspec1.shape[0])
        for i in range(sumspec1.shape[0]):
            maxspec1[i] = spectra1[:,:,i].max()
    
    # write h5 file in our structure
    if outdir is None:
        filename = h5id15[0].split(".")[0]+scan_suffix+'.h5' #scanid is of type 1.1,  2.1,  4.1
    else:
        if not outdir.endswith('/'):
            outdir += '/'
        filename = outdir+os.path.splitext(os.path.basename(h5id15[0]))[0]+scan_suffix+'.h5'
    with h5py.File(filename, 'w', locking=True) as f:
        f.create_dataset('cmd', data=scan_cmd)
        f.create_dataset('raw/channel00/spectra', data=spectra0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/icr', data=icr0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/ocr', data=ocr0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/sumspec', data=sumspec0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/maxspec', data=maxspec0, compression='gzip', compression_opts=4)
        if ch1id is not None:
            f.create_dataset('raw/channel01/spectra', data=spectra1, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel01/icr', data=icr1, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel01/ocr', data=ocr1, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel01/sumspec', data=sumspec1, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel01/maxspec', data=maxspec1, compression='gzip', compression_opts=4)
        f.create_dataset('raw/I0', data=i0, compression='gzip', compression_opts=4)
        if i1id is not None:
            f.create_dataset('raw/I1', data=i1, compression='gzip', compression_opts=4)
        dset = f.create_dataset('mot1', data=mot1, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot1_name
        dset = f.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot2_name
        f.create_dataset('raw/acquisition_time', data=tm, compression='gzip', compression_opts=4)
    print("Done")
