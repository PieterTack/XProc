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

##############################################################################
def read_cnc(cncfile):
    rv = Cnc()
    line = ""
    f = open(cncfile, "r")
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
    f.close()
    
    return rv

##############################################################################
def PCA(rawdata, nclusters=5, el_id=None):
    """
    returns: data transformed in 5 dims/columns + regenerated original data
    pass in: data as 2D NumPy array of dimension [M,N] with M the amount of observations and N the variables/elements
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
# perform PCA analysis on h5file dataset. 
#   before clustering, the routine performs a sqrt() normalisation on the data to reduce intensity differences between elements
#   a selection of elements can be given in el_id as their integer values corresponding to their array position in the dataset (first element id = 0)
#   kmeans can be set as an option, which will perform Kmeans clustering on the PCA score images and extract the respective sumspectra
def h5_pca(h5file, h5dir, nclusters=5, el_id=None, kmeans=False):
    # read in h5file data, with appropriate h5dir
    file = h5py.File(h5file, 'r+')
    data = np.array(file[h5dir])
    if el_id is not None:
        names = [n.decode('utf8') for n in file['/'.join(h5dir.split("/")[0:-1])+'/names']]
    if 'channel00' in h5dir:
        if kmeans is not None:
            spectra = np.array(file['raw/channel00/spectra'])
        channel = 'channel00'
    elif 'channel02' in h5dir:
        if kmeans is not None:
            spectra = np.array(file['raw/channel02/spectra'])
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
        if spectra.shape[0] == clusters.size:
            for i in range(nclusters):
                file.create_dataset('kmeans/'+channel+'/sumspec_'+str(i), data=sumspec[i,:], compression='gzip', compression_opts=4)    
    file.close()    

##############################################################################
def Kmeans(rawdata, nclusters=5, el_id=None, whiten=True):
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
# perform Kmeans clustering on a h5file dataset.
#   a selection of elements can be given in el_id as their integer values corresponding to their array position in the dataset (first element id = 0)
#   Before clustering data is whitened using scipy routines
def h5_kmeans(h5file, h5dir, nclusters=5, el_id=None, nosumspec=False):
    # read in h5file data, with appropriate h5dir
    file = h5py.File(h5file, 'r+')
    data = np.array(file[h5dir])
    if el_id is not None:
        names = [n.decode('utf8') for n in file['/'.join(h5dir.split("/")[0:-1])+'/names']]
    if 'channel00' in h5dir:
        spectra = np.array(file['raw/channel00/spectra'])
        channel = 'channel00'
    elif 'channel02' in h5dir:
        spectra = np.array(file['raw/channel02/spectra'])
        channel = 'channel02'
    spectra = spectra.reshape((spectra.shape[0]*spectra.shape[1], spectra.shape[2]))
    
    # perform Kmeans clustering
    clusters, centroids = Kmeans(data, nclusters=nclusters, el_id=el_id)
    
    # calculate cluster sumspectra
    #   first check if raw spectra shape is identical to clusters shape, as otherwise it's impossible to relate appropriate spectrum to pixel
    #TODO: in case of timetriggered scans cluster.ravel indices may not match the appropriate cluster point!
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
    if el_id is not None:
        file.create_dataset('kmeans/'+channel+'/el_id', data=[n.encode('utf8') for n in np.array(names)[el_id]])
    else:
        file.create_dataset('kmeans/'+channel+'/el_id', data='None')        
    if nosumspec is False:
        if spectra.shape[0] == clusters.size:
            for i in range(nclusters):
                dset = file.create_dataset('kmeans/'+channel+'/sumspec_'+str(i), data=np.array(sumspec)[i,:], compression='gzip', compression_opts=4)    
                dset.attrs["NPixels"] = np.asarray(np.where(clusters.ravel() == i)).size
    file.close()
    
##############################################################################
# divide quantified images by the corresponding concentration value of the same element in the cncfile to obtain relative difference images
#   If an element in the h5file is not present in the cncfile it is simply not calculated and ignored
def div_by_cnc(h5file, cncfile, channel=None):
    # read in h5file quant data
    #   normalise intensities to 1s acquisition time as this is the time for which we have el yields
    file = h5py.File(h5file, 'r+')
    if channel is None:
        channel = list(file['quant'].keys())[0]
    h5_ims = np.array(file['quant/'+channel+'/ims'])
    h5_names = np.array([n.decode('utf8') for n in file['quant/'+channel+'/names']])
    h5_z = [Elements.getz(n.split(" ")[0]) for n in h5_names]
    
    # read in cnc file
    cnc = read_cnc(cncfile)

    # loop over h5_z and count how many times there's a common z in h5_z and cnc.z
    cnt = 0
    for z in range(0,len(h5_z)):
        if h5_z[z] in cnc.z:
            cnt+=1
    
    # make array to store rel_diff data and calculate them
    rel_diff = np.zeros((cnt, h5_ims.shape[1], h5_ims.shape[2]))
    rel_names = []
    
    cnt = 0
    for z in range(0, len(h5_z)):
        if h5_z[z] in cnc.z:
            rel_diff[cnt, :, :] = h5_ims[z,:,:] / cnc.conc[list(cnc.z).index(h5_z[z])]
            rel_names.append(h5_names[z])
            cnt+=1

    # save rel_diff data
    try:
        del file['rel_dif/'+channel+'/names']
        del file['rel_dif/'+channel+'/ims']
        del file['rel_dif/'+channel+'/cnc']
    except Exception:
        pass
    file.create_dataset('rel_dif/'+channel+'/names', data=[n.encode('utf8') for n in rel_names])
    file.create_dataset('rel_dif/'+channel+'/ims', data=rel_diff, compression='gzip', compression_opts=4)
    file.create_dataset('rel_dif/'+channel+'/cnc', data=cncfile.split("/")[-1].encode('utf8'))
    file.close()

##############################################################################
# quantify XRF data, making use of elemental yields as determined from reference files
#   h5file and reffiles should all contain norm data as determined by norm_xrf_batch()
#   The listed ref files should have had their detection limits calculated by calc_detlim() before
#       as this function also calculates element yields.
#   If an element in the h5file is not present in the listed refs, its yield is estimated through linear interpolation of the closest neighbouring atoms with the same linetype.
#       if Z is at the start of end of the reference elements, the yield will be extrapolated from the first or last 2 elements in the reference
#       if only 1 element in the reference has the same linetype as the quantifiable element, but does not have the same Z, the same yield is used nevertheless as inter/extrapolation is impossible
#   If keyword norm is provided, the elemental yield is corrected for the intensity of this signal
#       the signal has to be present in both reference and XRF data fit
#   If keyword absorb is provided, the fluorescence signal in the XRF data is corrected for absorption through sample matrix
#       type: tuple (['element'], 'cnc file')
#       the element will be used to find Ka and Kb line intensities and correct for their respective ratio
#       using concentration values from the provided cnc files.
def quant_with_ref(h5file, reffiles, channel='channel00', norm=None, absorb=None, snake=False):
    import plotims


    # first let's go over the reffiles and calculate element yields
    #   distinguish between K and L lines while doing this
    reffiles = np.array(reffiles)
    if reffiles.size == 1:
        reff = h5py.File(str(reffiles), 'r')
        ref_yld = [yld for yld in reff['elyield/'+[keys for keys in reff['elyield'].keys()][0]+'/'+channel+'/yield']] # elemental yields in (ug/cm²)/(ct/s)
        ref_yld_err = [yld for yld in reff['elyield/'+[keys for keys in reff['elyield'].keys()][0]+'/'+channel+'/stddev']] # elemental yield errors in (ug/cm²)/(ct/s)
        ref_names = [n.decode('utf8') for n in reff['elyield/'+[keys for keys in reff['elyield'].keys()][0]+'/'+channel+'/names']]
        ref_z = [Elements.getz(n.split(" ")[0]) for n in ref_names]
        ref_yld_err = np.array(ref_yld_err) / np.array(ref_yld) #convert to relative error
        if norm is not None:
            names = [n.decode('utf8') for n in reff['norm/'+channel+'/names']]
            if norm in names:
                sum_fit = np.array(reff['norm/'+channel+'/sum/int'])
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
            reff = h5py.File(str(reffiles[i]), 'r')
            ref_yld_tmp = [yld for yld in reff['elyield/'+[keys for keys in reff['elyield'].keys()][0]+'/'+channel+'/yield']] # elemental yields in (ug/cm²)/(ct/s)
            ref_yld_err_tmp = [yld for yld in reff['elyield/'+[keys for keys in reff['elyield'].keys()][0]+'/'+channel+'/stddev']] # elemental yield errors in (ug/cm²)/(ct/s)
            ref_names_tmp = [n.decode('utf8') for n in reff['elyield/'+[keys for keys in reff['elyield'].keys()][0]+'/'+channel+'/names']]
            ref_yld_err_tmp = np.array(ref_yld_err_tmp) / np.array(ref_yld_tmp) #convert to relative error
            if norm is not None:
                names = [n.decode('utf8') for n in reff['norm/'+channel+'/names']]
                if norm in names:
                    ref_sum_fit = np.array(reff['norm/'+channel+'/sum/int'])
                    ref_sum_bkg = np.array(reff['norm/'+channel+'/sum/bkg'])
                    ref_rawI0 = np.sum(np.array(reff['raw/I0']))
                    ref_sum_fit[np.where(ref_sum_fit < 0)] = 0
                    ref_sum_bkg[np.where(ref_sum_bkg < 0)] = 0
                    ref_normto = np.array(reff['norm/I0'])
                    ref_sum_fit = ref_sum_fit / ref_normto
                    ref_sum_bkg = ref_sum_bkg / ref_normto
                    ref_yld_err_tmp = np.sqrt(ref_yld_err_tmp*ref_yld_err_tmp + np.sqrt((ref_sum_fit[names.index(norm)]+2.*ref_sum_bkg[names.index(norm)])*ref_rawI0)/(ref_sum_fit[names.index(norm)]*ref_rawI0))
                    ref_yld_tmp = [yld*(ref_sum_fit[names.index(norm)]) for yld in ref_yld_tmp]
                else:
                    print("ERROR: quant_with_ref: norm signal not present for reference material in "+reffiles[i])
                    return False
            for j in range(0, np.array(ref_yld_tmp).size):
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
    file = h5py.File(h5file, 'r')
    h5_ims = np.asarray(file['norm/'+channel+'/ims'])
    h5_ims_err = np.asarray(file['norm/'+channel+'/ims_stddev'])/h5_ims[:]
    h5_names = np.asarray([n.decode('utf8') for n in file['norm/'+channel+'/names']])
    h5_sum = np.asarray(file['norm/'+channel+'/sum/int'])
    h5_sum_err = np.asarray(file['norm/'+channel+'/sum/int_stddev'])/h5_sum[:]
    h5_normto = np.asarray(file['norm/I0'])
    if absorb is not None:
        h5_spectra = np.asarray(file['raw/'+channel+'/spectra'])
        h5_cfg = file['fit/'+channel+'/cfg'][()].decode('utf8')
        if len(h5_spectra.shape) == 2:
            h5_spectra = h5_spectra.reshape((h5_spectra.shape[0], 1, h5_spectra.shape[1]))
        elif len(h5_spectra.shape) == 1:
            h5_spectra = h5_spectra.reshape((1, 1, h5_spectra.shape[0]))
    if snake is True:
        mot1 = np.asarray(file['mot1'])
        mot2 = np.asarray(file['mot2'])
    file.close()

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
                ims[i,:,:] = ims[i,:,:] / h5_ims[list(h5_names).index(norm),:,:] #TODO: we can get some division by zero error here...
                sumint[i] = sumint[i] / h5_sum[list(h5_names).index(norm)]
                ims_err[i,:,:] = np.sqrt(ims_err[i,:,:]*ims_err[i,:,:] + h5_ims_err[list(h5_names).index(norm),:,:]*h5_ims_err[list(h5_names).index(norm),:,:])
                sumint_err[i] = np.sqrt(sumint_err[i]*sumint_err[i] + h5_sum_err[list(h5_names).index(norm)]*h5_sum_err[list(h5_names).index(norm)])
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
            ratio_ka1_kb1[np.where(ratio_ka1_kb1 <= 0.55*rate_ka1[j]/rate_kb1[j])] = rate_ka1[j]/rate_kb1[j] #TODO: this value may be inappropriate...
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

    # check which relative errors are largest: sumint_err or np.average(ims_err) or np.std(ims)/np.average(ims)
    #   then use this error as the sumint_err
    for i in range(sumint_err.size):
        sumint_err[i] = np.max(np.array([sumint_err[i], np.average(ims_err[i,:,:]), np.std(ims[i,:,:])/np.average(ims[i,:,:])]))

    # convert relative errors to absolute errors
    ims_err = ims_err*ims
    sumint_err = sumint_err*sumint
            
    # save quant data
    file = h5py.File(h5file, 'r+')
    try:
        del file['quant/'+channel]
    except Exception:
        pass
    conc_unit = "ug/cm²"
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
    if absorb is not None:
        file.create_dataset('quant/'+channel+'/ratio_exp', data=ratio_ka1_kb1, compression='gzip', compression_opts=4)
        file.create_dataset('quant/'+channel+'/ratio_th', data=rate_ka1/rate_kb1)
        file.create_dataset('quant/'+channel+'/rhot', data=rhot, compression='gzip', compression_opts=4)
    file.close()

    # plot images
    data = plotims.ims()
    data.data = np.zeros((ims.shape[1],ims.shape[2], ims.shape[0]+1))
    for i in range(0, ims.shape[0]):
        data.data[:, :, i] = ims[i, :, :]
    if absorb is not None:
        data.data[:,:,-1] = rhot[:,:]
    names = np.concatenate((names,[r'$\rho T$']))
    data.names = names
    cb_opts = plotims.Colorbar_opt(title=r'Conc.;[$\mu$g/cm²]')
    nrows = int(np.ceil(len(names)/4)) # define nrows based on ncols
    colim_opts = plotims.Collated_image_opts(ncol=4, nrow=nrows, cb=True)
    plotims.plot_colim(data, names, 'viridis', cb_opts=cb_opts, colim_opts=colim_opts, save=os.path.splitext(h5file)[0]+'_ch'+channel[-1]+'_quant.png')


##############################################################################
# create detection limit image that is of publish quality
#   dl is an array of dimensions ([n_ref, ][n_tm, ]n_elements)
#   includes 3sigma error bars, ...
#   tm and ref are 1D str arrays denoting name and measurement time (including unit!) of corresponding data
#TODO: something still wrong with element labels on top of scatter plots
def plot_detlim(dl, el_names, tm=None, ref=None, dl_err=None, bar=False, save=None, ytitle="Detection Limit (ppm)"):
    tickfontsize = 22
    titlefontsize = 26

    # check shape of dl. If 1D, then only single curve selected. 2D array means several DLs
    dl = np.array(dl, dtype='object')
    el_names = np.array(el_names, dtype='object')
    if tm:
        tm = np.array(tm)
    if ref:
        ref = np.array(ref)
    if dl_err is not None:
        dl_err = np.array(dl_err, dtype='object')[:]*3. # we plot 3sigma error bars.
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
        el_names = np.array(el_names, dtype='str')
        all_names = np.array([str(name) for name in np.nditer(el_names)])
    else:
        all_names = []
        for i in range(0,len(el_names)):
            for j in range(0, len(el_names[i])):
                all_names.append(el_names[i][j])
        all_names = np.array(all_names, dtype='str')
    el_z = np.array([Elements.getz(name.split(" ")[0]) for name in all_names])
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
    el_labels = ["-".join(n.split(" ")) for n in all_names[unique_id]]
    z_temp = el_z[unique_id]
    unique_el = np.array(el_labels)[np.argsort(z_temp)]
    z_temp = z_temp[np.argsort(z_temp)]
    # if for same element/Z multiple lines, join them.
    # K or Ka should be lowest, L above, M above that, ... (as typically K gives lowest DL, L higher, M higher still...)
    unique_z, unique_id = np.unique(z_temp, return_index=True)
    if unique_z.size != z_temp.size:
        new_z = np.zeros(unique_z.size)
        new_labels = []
        for i in range(0, unique_z.size):
            for j in range(unique_id[i], z_temp.size):
                if z_temp[unique_id[i]] == z_temp[j]: # same Z
                    new_z[i] = z_temp[unique_id[i]]
                    new_labels.append(el_labels[unique_id[i]])
                    if el_labels[unique_id[i]].split("-")[1] != el_labels[j].split("-")[1]: # different linetype
                        if el_labels[unique_id[i]].split("-")[1] == 'K' or el_labels[unique_id[i]].split("-")[1] == 'K$/alpha$':
                            new_labels = new_labels[:-2]
                            new_labels.append(el_labels[j] + "\n" + el_labels[unique_id[i]])
                        else:
                            new_labels = new_labels[:-2]
                            new_labels.append(el_labels[unique_id[i]] + "\n" + el_labels[j])
        new_labels = np.array(new_labels)
    else:
        new_z = np.array(z_temp)
        new_labels = np.array(el_labels)
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
            ax.set_xticklabels(unique_el, fontsize=tickfontsize)
        else:
            plt.errorbar(el_z, dl, yerr=dl_err, label=str(ref)+'_'+str(tm), linestyle='', fmt=next(marker), capsize=3)
            ax = plt.gca()
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            plt.xlabel("Atomic Number (Z)", fontsize=titlefontsize)
            secaxx = ax.secondary_xaxis('top')
            secaxx.set_xticks(new_z)
            secaxx.set_xticklabels(new_labels, fontsize=tickfontsize)
            # fit curve through points and plot as dashed line in same color
            fit_par = np.polyfit(el_z, np.log(dl), 2)
            func = np.poly1d(fit_par)
            fit_x = np.linspace(np.min(el_z), np.max(el_z), num=(np.max(el_z)-np.min(el_z))*2)
            plt.plot(fit_x, np.exp(func(fit_x)), linestyle='--', color=plt.legend().legendHandles[0].get_colors()[0]) #Note: this also plots a legend, which is removed later on.
            ax.get_legend().remove()
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
            ref = np.array(['DL '+str(int(n)) for n in np.linspace(0, dl.shape[0]-1, num=dl.shape[0])])
        if tm is not None and tm.size > 1:
            if ref is not None:
                label_prefix = str(ref[0])+"_"
            else:
                label_prefix = ''
            for i in range(0, tm.size):
                # plot curves and axes
                if bar is True:
                    el = np.array(["-".join(name.split(" ")) for name in el_names[i]])
                    bar_x = np.zeros(el.size)
                    for k in range(0,el.size):
                        bar_x[k] = list(unique_el).index(el[i]) + (0.9/tm.size)*(i-(tm.size-1)/2.)
                    plt.bar(bar_x, dl[i], yerr=dl_err[i], label=label_prefix+str(tm[i]), capsize=3, width=(0.9/tm.size))
                    ax = plt.gca()
                    if i == 0:
                        ax.set_xticks(np.linspace(0,unique_el.size-1,num=unique_el.size))
                        ax.set_xticklabels(unique_el, fontsize=tickfontsize)
                else:
                    el_z = np.array([Elements.getz(name.split(" ")[0]) for name in el_names[i]])
                    plt.errorbar(el_z, dl[i], yerr=dl_err[i], label=label_prefix+str(tm[i]), linestyle='', fmt=next(marker), capsize=3)
                    ax = plt.gca()
                    if i == 0:
                        plt.xlabel("Atomic Number (Z)", fontsize=titlefontsize)
                        ax.xaxis.set_minor_locator(MultipleLocator(1))
                        secaxx = ax.secondary_xaxis('top')
                        secaxx.set_xticks(new_z)
                        secaxx.set_xticklabels(new_labels, fontsize=tickfontsize)
                    # fit curve through points and plot as dashed line in same color
                    fit_par = np.polyfit(el_z, np.log(np.array(dl[i], dtype='float64')), 2)
                    func = np.poly1d(fit_par)
                    fit_x = np.linspace(np.min(el_z), np.max(el_z), num=(np.max(el_z)-np.min(el_z))*2)
                    plt.plot(fit_x, np.exp(func(fit_x)), linestyle='--', color=plt.legend().legendHandles[-1].get_colors()[0]) #Note: this also plots a legend, which is removed later on.
                    ax.get_legend().remove()
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
                    el = np.array(["-".join(name.split(" ")) for name in el_names[i]])
                    bar_x = np.zeros(el.size) + (0.9/ref.size)*(i-(ref.size-1)/2.)
                    for k in range(0,el.size):
                        bar_x[k] = list(unique_el).index(el[k])
                    plt.bar(bar_x, dl[i], yerr=dl_err[i], label=str(ref[i])+label_suffix, capsize=3, width=(0.9/ref.size))
                    ax = plt.gca()
                    if i == 0:
                        ax.set_xticks(np.linspace(0,unique_el.size-1,num=unique_el.size))
                        ax.set_xticklabels(unique_el, fontsize=tickfontsize)
                else:
                    el_z = np.array([Elements.getz(name.split(" ")[0]) for name in el_names[i]])
                    plt.errorbar(el_z, dl[i], yerr=dl_err[i], label=str(ref[i])+label_suffix, linestyle='', fmt=next(marker), capsize=3)
                    ax = plt.gca()
                    if i == 0:
                        plt.xlabel("Atomic Number (Z)", fontsize=titlefontsize)
                        ax.xaxis.set_minor_locator(MultipleLocator(1))
                        secaxx = ax.secondary_xaxis('top')
                        secaxx.set_xticks(new_z)
                        secaxx.set_xticklabels(new_labels, fontsize=tickfontsize)
                    # fit curve through points and plot as dashed line in same color
                    fit_par = np.polyfit(el_z, np.log(dl[i]), 2)
                    func = np.poly1d(fit_par)
                    fit_x = np.linspace(np.min(el_z), np.max(el_z), num=(np.max(el_z)-np.min(el_z))*2)
                    plt.plot(fit_x, np.exp(func(fit_x)), linestyle='--', color=plt.legend().legendHandles[-1].get_colors()[0]) #Note: this also plots a legend, which is removed later on.
                    ax.get_legend().remove()
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
            tm = np.array(['tm'+str(int(n)) for n in np.linspace(0, dl.shape[0]-1, num=dl.shape[0])])
        if ref is None:
            ref = np.array(['ref'+str(int(n)) for n in np.linspace(0, dl.shape[1]-1, num=dl.shape[1])])
        for i in range(0, ref.size):
            for j in range(0, tm.size):
                # plot curves and axes
                if bar is True:
                    el = np.array(["-".join(name.split(" ")) for name in el_names[i,j]])
                    bar_x = np.zeros(el.size)
                    for k in range(0,el.size):
                        bar_x[k] = list(unique_el.index(el[k]) + (0.9/(tm.size*ref.size))*(i*tm.size+j-(tm.size*ref.size-1)/2.))
                    plt.bar(bar_x, dl[i,j], yerr=dl_err[i,j], label=str(ref[i])+'_'+str(tm[j]), capsize=3, width=(0.9/(tm.size*ref.size)))
                    ax = plt.gca()
                    if i == 0 and j == 0:
                        ax.set_xticks(np.linspace(0, unique_el.size-1, num=unique_el.size))
                        ax.set_xticklabels(unique_el, fontsize=tickfontsize)
                else:
                    el_z = np.array([Elements.getz(name.split(" ")[0]) for name in el_names[i,j]])
                    plt.errorbar(el_z, dl[i,j], yerr=dl_err[i,j], label=str(ref[i])+'_'+str(tm[j]), linestyle='', fmt=next(marker), capsize=3)
                    ax = plt.gca()
                    if i == 0 and j == 0:
                        plt.xlabel("Atomic Number (Z)", fontsize=titlefontsize)
                        ax.xaxis.set_minor_locator(MultipleLocator(1))
                        secaxx = ax.secondary_xaxis('top')
                        secaxx.set_xticks(new_z)
                        secaxx.set_xticklabels(new_labels, fontsize=tickfontsize)
                    # fit curve through points and plot as dashed line in same color
                    fit_par = np.polyfit(el_z, np.log(dl[i,j]), 2)
                    func = np.poly1d(fit_par)
                    fit_x = np.linspace(np.min(el_z), np.max(el_z), num=(np.max(el_z)-np.min(el_z))*2)
                    plt.plot(fit_x, np.exp(func(fit_x)), linestyle='--', color=plt.legend().legendHandles[-1].get_colors()[0]) #Note: this also plots a legend, which is removed later on.
                    ax.get_legend().remove()
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
            tm = np.array(['tm'+str(int(n)) for n in np.linspace(0, dl.shape[0]-1, num=dl.shape[0])])
        if tm.size == 1:
            tm_tmp = [tm]
        if ref is None:
            ref = np.array(['ref'+str(int(n)) for n in np.linspace(0, dl.shape[1]-1, num=dl.shape[1])])
        for i in range(0, ref.size):
            el_names_tmp = el_names[i]
            dl_tmp = dl[i]
            dl_err_tmp = dl_err[i]
            for j in range(0, tm.size):
                # plot curves and axes
                if bar is True:
                    el = np.array(["-".join(name.split(" ")) for name in el_names_tmp])
                    bar_x = np.zeros(el.size)
                    for k in range(0,el.size):
                        bar_x[k] = list(unique_el).index(el[k]) + (0.9/(tm.size*ref.size))*(i*tm.size+j-(tm.size*ref.size-1)/2.)
                    plt.bar(bar_x, dl_tmp, yerr=dl_err_tmp, label=str(ref[i])+'_'+str(tm_tmp[j]), capsize=3, width=(0.9/(tm.size*ref.size)))
                    ax = plt.gca()
                    if i == 0 and j == 0:
                        ax.set_xticks(np.linspace(0, unique_el.size-1, num=unique_el.size))
                        ax.set_xticklabels(unique_el, fontsize=tickfontsize)
                else:
                    el_z = np.array([Elements.getz(name.split(" ")[0]) for name in el_names_tmp])
                    plt.errorbar(el_z, dl_tmp, yerr=dl_err_tmp, label=str(ref[i])+'_'+str(tm_tmp[j]), linestyle='', fmt=next(marker), capsize=3)
                    ax = plt.gca()
                    if i == 0 and j == 0:
                        plt.xlabel("Atomic Number (Z)", fontsize=titlefontsize)
                        ax.xaxis.set_minor_locator(MultipleLocator(1))
                        secaxx = ax.secondary_xaxis('top')
                        secaxx.set_xticks(new_z)
                        secaxx.set_xticklabels(new_labels, fontsize=tickfontsize)
                    # fit curve through points and plot as dashed line in same color
                    fit_par = np.polyfit(el_z, np.log(dl_tmp), 2)
                    func = np.poly1d(fit_par)
                    fit_x = np.linspace(np.min(el_z), np.max(el_z), num=(np.max(el_z)-np.min(el_z))*2)
                    plt.plot(fit_x, np.exp(func(fit_x)), linestyle='--', color=plt.legend().legendHandles[-1].get_colors()[0]) #Note: this also plots a legend, which is removed later on.
                    ax.get_legend().remove()
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
# calculate detection limits.
#   DL = 3*sqrt(Ib)/Ip * Conc
#   calculates 1s and 1000s DL
#   Also calculates elemental yields (Ip/conc [(ct/s)/(ug/cm²)]) 
def calc_detlim(h5file, cncfile, tmnorm=False, plotytitle="Detection Limit (ppm)"):
    # read in cnc file data
    cnc = read_cnc(cncfile)
    
    # read h5 file
    file = h5py.File(h5file, 'r+')
    try:
        sum_fit0 = np.array(file['norm/channel00/sum/int'])
        sum_bkg0 = np.array(file['norm/channel00/sum/bkg'])
        sum_fit0_err = np.array(file['norm/channel00/sum/int_stddev'])/sum_fit0
        sum_bkg0_err = np.array(file['norm/channel00/sum/bkg_stddev'])/sum_bkg0
        names0 = file['norm/channel00/names']
        try:
            sum_fit2 = np.array(file['norm/channel02/sum/int'])
            sum_bkg2 = np.array(file['norm/channel02/sum/bkg'])
            sum_fit2_err = np.array(file['norm/channel02/sum/int_stddev'])/sum_fit2
            sum_bkg2_err = np.array(file['norm/channel02/sum/bkg_stddev'])/sum_bkg2
            names2 = file['norm/channel02/names']
            chan02_flag = True
        except Exception:
            chan02_flag = False
        tm = np.array(file['raw/acquisition_time']) # Note this is pre-normalised tm! Correct for I0 value difference between raw and I0norm
        I0 = np.array(file['raw/I0'])
        I0norm = np.array(file['norm/I0'])
    except Exception:
        print("ERROR: calc_detlim: cannot open normalised data in "+h5file)
        return
    
    # correct tm for appropriate normalisation factor
    #   tm is time for which DL would be calculated using values as reported, taking into account the previous normalisation factor
    tm = np.sum(tm)
    if tmnorm is True:
        normfactor = I0norm/(np.sum(I0)*tm)
    else:
        normfactor = I0norm/np.sum(I0)
        
    # undo normalisation on intensities as performed during norm_xrf_batch
    #   in order to get intensities matching the current tm value (i.e. equal to raw fit values)
    names0 = np.array([n.decode('utf8') for n in names0[:]])
    sum_bkg0 = sum_bkg0/normfactor
    sum_fit0 = sum_fit0/normfactor
    if chan02_flag:
        names2 = np.array([n.decode('utf8') for n in names2[:]])
        sum_bkg2 = sum_bkg2/normfactor
        sum_fit2 = sum_fit2/normfactor
    # prune cnc.conc array to appropriate elements according to names0 and names2
    #   creates arrays of size names0 and names2, where 0 values in conc0 and conc2 represent elements not stated in cnc_files.
    conc0 = np.zeros(names0.size)
    conc0_err = np.zeros(names0.size)
    conc0_air = np.zeros(names0.size)
    conc0_air_err = np.zeros(names0.size)
    for j in range(0, names0.size):
        el_name = names0[j].split(" ")[0]
        for i in range(0, cnc.z.size):
            if el_name == Elements.getsymbol(cnc.z[i]):
                conc0_air[j] = cnc.conc[i]*cnc.density*cnc.thickness*1E-7 # unit: [ug/cm²]
                conc0_air_err[j] = (cnc.err[i]/cnc.conc[i])*conc0_air[j] # unit: [ug/cm²]
                conc0[j] = cnc.conc[i] # unit: [ppm]
                conc0_err[j] = (cnc.err[i]/cnc.conc[i])*conc0[j] # unit: [ppm]
    if chan02_flag:
        conc2 = np.zeros(names2.size)
        conc2_err = np.zeros(names2.size)
        conc2_air = np.zeros(names2.size)
        conc2_air_err = np.zeros(names2.size)
        for j in range(0, names2.size):
            el_name = names2[j].split(" ")[0]
            for i in range(0, cnc.z.size):
                if el_name == Elements.getsymbol(cnc.z[i]):
                    conc2_air[j] = cnc.conc[i]*cnc.density*cnc.thickness*1E-7 # unit: [ug/cm²]
                    conc2_air_err[j] = (cnc.err[i]/cnc.conc[i])*conc2_air[j] # unit: [ug/cm²]
                    conc2[j] = cnc.conc[i] # unit: [ppm]
                    conc2_err[j] = (cnc.err[i]/cnc.conc[i])*conc2[j] # unit: [ppm]

    
    # some values will be 0 (due to conc0 or conc2 being 0). Ignore these in further calculations.
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
            el_yield_0.append((sum_fit0[i]*normfactor/I0norm) / conc0_air[i]) # element yield expressed as cps/conc
            # calculate DL errors (based on standard error propagation)
            dl_1s_err_0.append(np.sqrt(sum_fit0_err[i]**2 + sum_bkg0_err[i]**2 +
                                     (conc0_err[i]/conc0[i])*(conc0_err[i]/conc0[i])) * dl_1s_0[j])
            dl_1000s_err_0.append(dl_1s_err_0[j] / dl_1s_0[j] * dl_1000s_0[j])
            el_yield_err_0.append(np.sqrt((conc0_air_err[i]/conc0_air[i])*(conc0_air_err[i]/conc0_air[i]) + sum_fit0_err[i]**2)*el_yield_0[j])
            names0_mod.append(names0[i])
    if chan02_flag:
        names2_mod = []
        dl_1s_err_2 = []
        dl_1000s_err_2 = []
        dl_1s_2 = []
        dl_1000s_2 = []
        el_yield_2 = []
        el_yield_err_2 = []
        for i in range(0, conc2.size):
            if conc2[i] > 0:
                # detection limit corresponding to tm=1s
                dl_1s_2.append( (3.*np.sqrt(sum_bkg2[i]/tm)/(sum_fit2[i]/tm)) * conc2[i])
                j = len(dl_1s_2)-1
                dl_1000s_2.append(dl_1s_2[j] / np.sqrt(1000.))
                el_yield_2.append((sum_fit2[i]*normfactor/I0norm) / conc2_air[i]) # element yield expressed as cps/conc
                # calculate DL errors (based on standard error propagation)
                dl_1s_err_2.append(np.sqrt(sum_fit2_err[i]**2 + sum_bkg2_err[i]**2 +
                                         (conc2_err[i]/conc2[i])*(conc2_err[i]/conc2[i])) * dl_1s_2[j])
                dl_1000s_err_2.append(dl_1s_err_2[j] / dl_1s_2[j] * dl_1000s_2[j])
                el_yield_err_2.append(np.sqrt((conc2_air_err[i]/conc2_air[i])*(conc2_air_err[i]/conc2_air[i]) +sum_fit2_err[i]**2)*el_yield_2[j])
                names2_mod.append(names2[i])
    # save DL data to file
    cncfile = cncfile.split("/")[-1]
    try:
        del file['detlim/'+cncfile+'/unit']
    except Exception:
        pass
    file.create_dataset('detlim/'+cncfile+'/unit', data='ppm')
    try:
        del file['detlim/'+cncfile+'/channel00/names']
        del file['detlim/'+cncfile+'/channel00/1s/data']
        del file['detlim/'+cncfile+'/channel00/1s/stddev']
        del file['detlim/'+cncfile+'/channel00/1000s/data']
        del file['detlim/'+cncfile+'/channel00/1000s/stddev']
        del file['elyield/'+cncfile+'/channel00/yield']
        del file['elyield/'+cncfile+'/channel00/stddev']
        del file['elyield/'+cncfile+'/channel00/names']
    except Exception:
        pass
    file.create_dataset('detlim/'+cncfile+'/channel00/names', data=[n.encode('utf8') for n in names0_mod[:]])
    file.create_dataset('detlim/'+cncfile+'/channel00/1s/data', data=dl_1s_0, compression='gzip', compression_opts=4)
    file.create_dataset('detlim/'+cncfile+'/channel00/1s/stddev', data=dl_1s_err_0, compression='gzip', compression_opts=4)
    file.create_dataset('detlim/'+cncfile+'/channel00/1000s/data', data=dl_1000s_0, compression='gzip', compression_opts=4)
    file.create_dataset('detlim/'+cncfile+'/channel00/1000s/stddev', data=dl_1000s_err_0, compression='gzip', compression_opts=4)    
    dset = file.create_dataset('elyield/'+cncfile+'/channel00/yield', data=el_yield_0, compression='gzip', compression_opts=4)
    dset.attrs["Unit"] = "(ct/s)/(ug/cm²)"
    dset = file.create_dataset('elyield/'+cncfile+'/channel00/stddev', data=el_yield_err_0, compression='gzip', compression_opts=4)
    dset.attrs["Unit"] = "(ct/s)/(ug/cm²)"
    file.create_dataset('elyield/'+cncfile+'/channel00/names', data=[n.encode('utf8') for n in names0_mod[:]])
    if chan02_flag:
        try:
            del file['detlim/'+cncfile+'/channel02/names']
            del file['detlim/'+cncfile+'/channel02/1s/data']
            del file['detlim/'+cncfile+'/channel02/1s/stddev']
            del file['detlim/'+cncfile+'/channel02/1000s/data']
            del file['detlim/'+cncfile+'/channel02/1000s/stddev']        
            del file['elyield/'+cncfile+'/channel02/yield']
            del file['elyield/'+cncfile+'/channel02/stddev']
            del file['elyield/'+cncfile+'/channel02/names']
        except Exception:
            pass
        file.create_dataset('detlim/'+cncfile+'/channel02/names', data=[n.encode('utf8') for n in names2_mod[:]])
        file.create_dataset('detlim/'+cncfile+'/channel02/1s/data', data=dl_1s_2, compression='gzip', compression_opts=4)
        file.create_dataset('detlim/'+cncfile+'/channel02/1s/stddev', data=dl_1s_err_2, compression='gzip', compression_opts=4)
        file.create_dataset('detlim/'+cncfile+'/channel02/1000s/data', data=dl_1000s_2, compression='gzip', compression_opts=4)
        file.create_dataset('detlim/'+cncfile+'/channel02/1000s/stddev', data=dl_1000s_err_2, compression='gzip', compression_opts=4)  
        dset = file.create_dataset('elyield/'+cncfile+'/channel02/yield', data=el_yield_2, compression='gzip', compression_opts=4)
        dset.attrs["Unit"] = "(ct/s)/(ug/cm²)"
        dset = file.create_dataset('elyield/'+cncfile+'/channel02/stddev', data=el_yield_err_2, compression='gzip', compression_opts=4)
        dset.attrs["Unit"] = "(ct/s)/(ug/cm²)"
        file.create_dataset('elyield/'+cncfile+'/channel02/names', data=[n.encode('utf8') for n in names2_mod[:]])
    file.close()
    
    # plot the DLs
    # plot_detlim(dl_1s_0, names0_mod, tm='1s', ref='atho-g', dl_err=dl_1s_err_0, bar=True, save=h5file.split('.')+'_DL.png')
    plot_detlim([dl_1s_0, dl_1000s_0],
                [names0_mod, names0_mod],
                tm=['1s','1000s'], ref=['DL'], 
                dl_err=[dl_1s_err_0, dl_1000s_err_0], bar=False, save=str(os.path.splitext(h5file)[0])+'_ch0_DL.png', ytitle=plotytitle)
    if chan02_flag:
        plot_detlim([dl_1s_2, dl_1000s_2],
                    [names2_mod, names2_mod],
                    tm=['1s','1000s'], ref=['DL'], 
                    dl_err=[dl_1s_err_2, dl_1000s_err_2], bar=False, save=str(os.path.splitext(h5file)[0])+'_ch2_DL.png', ytitle=plotytitle)

##############################################################################
# make publish-worthy overview images of all fitted elements in h5file (including scale bars, colorbar, ...)
# plot norm if present, otherwise plot fit/.../ims
def hdf_overview_images(h5file, datadir, ncols, pix_size, scl_size, log=False, rotate=0, fliph=False, cb_opts=None, clim=None):
    import plotims
    filename = os.path.splitext(h5file)[0]

    imsdata0 = plotims.read_h5(h5file, datadir+'/channel00/ims')
    imsdata0.data[imsdata0.data < 0] = 0.
    if log:
        filename += '_log'
    try:
        imsdata2 = plotims.read_h5(h5file, datadir+'/channel02/ims')
        if imsdata2 is None:
            chan02_flag = False
        else:
            imsdata2.data[imsdata2.data < 0] = 0.
            chan02_flag = True
    except Exception:
        chan02_flag = False

    # rotate where appropriate
    if rotate != 0:
        imsdata0.data = np.rot90(imsdata0.data, k=np.rint(rotate/90.), axes=(0,1))
        if chan02_flag == True:
            imsdata2.data = np.rot90(imsdata2.data, k=np.rint(rotate/90.), axes=(0,1))
    # flip image horizontally
    if fliph is True:
        imsdata0.data = np.flip(imsdata0.data, axis=0)
        if chan02_flag == True:
            imsdata2.data = np.flip(imsdata2.data, axis=0)
            
    # set plot options (color limits, clim) if appropriate
    if clim is not None:
        plt_opts = plotims.Plot_opts(clim=clim)
    else:
        plt_opts = None

    sb_opts = plotims.Scale_opts(xscale=True, x_pix_size=pix_size, x_scl_size=scl_size, x_scl_text=str(scl_size)+' µm')
    if cb_opts is None:
        if log:
            cb_opts = plotims.Colorbar_opt(title='log. Int.;[cts]')
        else:
            cb_opts = plotims.Colorbar_opt(title='Int.;[cts]')
    nrows = int(np.ceil(len(imsdata0.names)/ncols)) # define nrows based on ncols
    colim_opts = plotims.Collated_image_opts(ncol=ncols, nrow=nrows, cb=True)

    if log:
        imsdata0.data = np.log10(imsdata0.data)

    
    plotims.plot_colim(imsdata0, imsdata0.names, 'viridis', sb_opts=sb_opts, cb_opts=cb_opts, colim_opts=colim_opts, plt_opts=plt_opts, save=filename+'_ch0_'+datadir+'_overview.png')
    
    if chan02_flag == True:
        # set plot options (color limits, clim) if appropriate
        if clim is not None:
            plt_opts = plotims.Plot_opts(clim=clim)
        else:
            plt_opts = None


        nrows = int(np.ceil(len(imsdata2.names)/ncols)) # define nrows based on ncols
        colim_opts = plotims.Collated_image_opts(ncol=ncols, nrow=nrows, cb=True)

        if log:
            imsdata2.data = np.log10(imsdata2.data)
        
        plotims.plot_colim(imsdata2, imsdata2.names, 'viridis', sb_opts=sb_opts, cb_opts=cb_opts, colim_opts=colim_opts, plt_opts=plt_opts, save=filename+'_ch2_'+datadir+'_overview.png')


##############################################################################
# normalise IMS images to detector deadtime and I0 values.
#   When I0norm is supplied, a (long) int should be provided to which I0 value one should normalise. Otherwise the max of the I0 map is used.
#   tmnorm: sometimes the I0 values are not representative of acquisition time, then additional normalisation for acquisition time can be performed by setting tmnorm to True
def norm_xrf_batch(h5file, I0norm=None, snake=False, sort=False, timetriggered=False, tmnorm=False, halfpixshift=True, mot2nosort=False):
    print("Initiating data normalisation of <"+h5file+">...", end=" ")
    # read h5file
    file = h5py.File(h5file, 'r+')
    ims0 = np.squeeze(np.array(file['fit/channel00/ims']))
    names0 = file['fit/channel00/names']
    sum_fit0 = np.array(file['fit/channel00/sum/int'])
    sum_bkg0 = np.array(file['fit/channel00/sum/bkg'])
    I0 =  np.array(file['raw/I0'])
    tm = np.array(file['raw/acquisition_time'])
    mot1 = np.array(file['mot1'])
    mot1_name = str(file['mot1'].attrs["Name"])
    mot2 = np.array(file['mot2'])
    mot2_name = str(file['mot2'].attrs["Name"])
    cmd = str(np.array(file['cmd'])).split(' ')
    if len(ims0.shape) == 2 or len(ims0.shape) == 1:
        if len(ims0.shape) == 2:
            ims0 = ims0.reshape((ims0.shape[0], ims0.shape[1], 1))
            I0 = I0.reshape((np.squeeze(I0).shape[0], 1))
            tm = tm.reshape((np.squeeze(tm).shape[0], 1))
            mot1 = mot1.reshape((np.squeeze(mot1).shape[0], 1))
            mot2 = mot2.reshape((np.squeeze(mot2).shape[0], 1))
        else:
            ims0 = ims0.reshape((ims0.shape[0],1, 1))
            I0 = I0.reshape((I0.shape[0], 1))
            tm = tm.reshape((tm.shape[0], 1))
            mot1 = mot1.reshape((mot1.shape[0], 1))
            mot2 = mot2.reshape((mot2.shape[0], 1))
        if I0.shape[0] > ims0.shape[1]:
            I0 = I0[0:ims0.shape[1],:]
        if tm.shape[0] > ims0.shape[1]:
            tm = tm[0:ims0.shape[1],:]
        if mot1.shape[0] > ims0.shape[1]:
            mot1 = mot1[0:ims0.shape[1],:]            
        if mot2.shape[0] > ims0.shape[1]:
            mot2 = mot2[0:ims0.shape[1],:]
        if ims0.shape[1] > mot1.shape[0]:
            ims0 = ims0[:,0:mot1.shape[0],:]      
            I0 = I0[0:mot1.shape[0],:]
            tm = tm[0:mot1.shape[0],:]            
        if cmd[0] != "b'timescanc" and cmd[0] != "b'dscan" and cmd[0] != "timescanc" and cmd[0] != "dscan" and cmd[0] != "c" and cmd[0] != "b'c":
            print(cmd[0])
            snake = True
            timetriggered=True  #if timetriggered is true one likely has more datapoints than fit on the regular grid, so have to interpolate in different way

    try:
        ims2 = np.squeeze(np.array(file['fit/channel02/ims']))
        if len(ims2.shape) == 2:
            ims2 = ims2.reshape((ims2.shape[0], ims2.shape[1], 1))
        elif len(ims2.shape) == 1:
            ims2 = ims2.reshape((ims2.shape[0],1,1))
            if ims2.shape[1] > mot1.shape[0]:
                ims2 = ims2[:,0:mot1.shape[0],:]            
        names2 = file['fit/channel02/names']
        sum_fit2 = np.array(file['fit/channel02/sum/int'])
        sum_bkg2 = np.array(file['fit/channel02/sum/bkg'])
        chan02_flag = True
    except Exception:
        chan02_flag = False
    
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
            if chan02_flag:
                ims2[:,i,:] = ims2[:,i,sort_id]
        # To make sure (especially when merging scans) sort mot2 as well
        if mot2nosort is not True:
            for i in range(mot2[0,:].size):
                sort_id = np.argsort(mot2[:,i])
                ims0[:,:,i] = ims0[:,sort_id,i]
                mot1[:,i] = mot1[sort_id,i]
                mot2[:,i] = mot2[sort_id,i]
                I0[:,i] = I0[sort_id,i]
                tm[:,i] = tm[sort_id,i]
                if chan02_flag:
                    ims2[:,:,i] = ims2[:,sort_id,i]
        try:
            del file['fit/channel00/ims']
            del file['raw/I0']
            del file['mot1']
            del file['mot2']
            del file['raw/acquisition_time']
            if chan02_flag:
                del file['fit/channel02/ims']
        except Exception:
            pass
        file.create_dataset('fit/channel00/ims', data=ims0, compression='gzip', compression_opts=4)
        file.create_dataset('raw/I0', data=I0, compression='gzip', compression_opts=4)
        dset = file.create_dataset('mot1', data=mot1, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot1_name
        dset = file.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot2_name
        file.create_dataset('raw/acquisition_time', data=tm, compression='gzip', compression_opts=4)
        if chan02_flag:
            file.create_dataset('fit/channel02/ims', data=ims2, compression='gzip', compression_opts=4)
        

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
        sum_fit0 = sum_fit0/(np.sum(I0)*np.sum(tm)) * normto
        sum_bkg0 = sum_bkg0/(np.sum(I0)*np.sum(tm)) * normto
    else:
        sum_fit0 = (sum_fit0/np.sum(I0)) * normto
        sum_bkg0 = (sum_bkg0/np.sum(I0)) * normto
    ims0[np.isnan(ims0)] = 0.
    if chan02_flag:
        ims2[ims2 < 0] = 0.
        sum_fit2[sum_fit2 < 0] = 0.
        sum_bkg2[sum_bkg2 < 0] = 0.
        ims2_err = np.nan_to_num(np.sqrt(ims2)/ims2)
        sum_fit2_err = np.nan_to_num(np.sqrt(sum_fit2[:]+2*sum_bkg2[:])/sum_fit2[:])
        sum_bkg2_err = np.nan_to_num(np.sqrt(sum_bkg2[:])/sum_bkg2[:])
        for i in range(0, ims2.shape[0]):
            if tmnorm is True:
                ims2[i,:,:] = ims2[i,:,:]/(I0*tm) * normto
            else:
                ims2[i,:,:] = ims2[i,:,:]/(I0) * normto
        if tmnorm is True:
            sum_fit2 = sum_fit2/(np.sum(I0)*np.sum(tm)) * normto
            sum_bkg2 = sum_bkg2/(np.sum(I0)*np.sum(tm)) * normto
        else:
            sum_fit2 = (sum_fit2/np.sum(I0)) * normto
            sum_bkg2 = (sum_bkg2/np.sum(I0)) * normto
        ims2[np.isnan(ims2)] = 0.
        

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
            if chan02_flag:
                ims2_tmp = np.zeros((ims2.shape[0], ims2.shape[1], ims2.shape[2]))
                ims2_err_tmp = np.zeros((ims2.shape[0], ims2.shape[1], ims2.shape[2]))
        if timetriggered is True:
            if halfpixshift is True:
                # correct positions for half pixel shift
                mot1[0:mot1.size-1, 0] = mot1[0:mot1.size-1, 0] + np.diff(mot1[:,0])/2.
            # based on cmd determine regular grid positions
            mot1_pos = np.linspace(float(cmd[2]), float(cmd[3]), num=int(cmd[4]))
            mot2_pos = np.linspace(float(cmd[6]), float(cmd[7]), num=int(cmd[8])) 
            if cmd[0] == "b'cdmesh":
                mot1_pos = mot1_pos - (mot1_pos[0] - mot1[0,0])
                mot2_pos = mot2_pos - (mot2_pos[0] - mot2[0,0])
            ims0_tmp = np.zeros((ims0.shape[0], mot2_pos.shape[0], mot1_pos.shape[0]))
            ims0_err_tmp = np.zeros((ims0.shape[0], mot2_pos.shape[0], mot1_pos.shape[0]))
            if chan02_flag:
                ims2_tmp = np.zeros((ims2.shape[0], mot2_pos.shape[0], mot1_pos.shape[0]))
                ims2_err_tmp = np.zeros((ims2.shape[0], mot2_pos.shape[0], mot1_pos.shape[0]))
        # interpolate to the regular grid motor positions
        mot1_tmp, mot2_tmp = np.mgrid[mot1_pos[0]:mot1_pos[-1]:complex(mot1_pos.size),
                mot2_pos[0]:mot2_pos[-1]:complex(mot2_pos.size)]
        x = mot1.ravel()
        y = mot2.ravel()

        # import matplotlib.pyplot as plt
        # plt.imshow(mot1_tmp)
        # plt.colorbar()
        # plt.savefig('test_mot1.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
        # plt.imshow(mot2_tmp)
        # plt.colorbar()
        # plt.savefig('test_mot2.png', bbox_inches='tight', pad_inches=0)
        # plt.close()
        # plt.imshow(ims0[8,0:255530].reshape((505,506)))
        # plt.colorbar()
        # plt.savefig('test_ims.png', bbox_inches='tight', pad_inches=0)
        # plt.close()

        for i in range(names0.size):
            values = ims0[i,:,:].ravel()
            # ims0_tmp[i,:,:] = griddata((x, y), values, (mot1_tmp, mot2_tmp), method='cubic', rescale=True).T
            # ims0_err_tmp[i,:,:] = griddata((x, y), ims0_err[i,:,:].ravel(), (mot1_tmp, mot2_tmp), method='cubic', rescale=True).T
            ims0_tmp[i,:,:] = griddata((x, y), values, (mot1_tmp, mot2_tmp), method='nearest').T
            ims0_err_tmp[i,:,:] = griddata((x, y), ims0_err[i,:,:].ravel(), (mot1_tmp, mot2_tmp), method='nearest').T
        ims0 = np.nan_to_num(ims0_tmp)
        ims0_err = np.nan_to_num(ims0_err_tmp)*ims0
        if chan02_flag:
            for i in range(names2.size):
                values = ims2[i,:,:].ravel()
                # ims2_tmp[i,:,:] = griddata((x, y), values, (mot1_tmp, mot2_tmp), method='cubic', rescale=True).T
                # ims2_err_tmp[i,:,:] = griddata((x, y), ims2_err[i,:,:].ravel(), (mot1_tmp, mot2_tmp), method='cubic', rescale=True).T
                ims2_tmp[i,:,:] = griddata((x, y), values, (mot1_tmp, mot2_tmp), method='nearest').T
                ims2_err_tmp[i,:,:] = griddata((x, y), ims2_err[i,:,:].ravel(), (mot1_tmp, mot2_tmp), method='nearest').T
            ims2 = np.nan_to_num(ims2_tmp)
            ims2_err = np.nan_to_num(ims2_err_tmp)*ims2
        print("Done")

  
    # save normalised data
    print("     Writing...", end=" ")
    try:
        del file['norm/I0']
        del file['norm/channel00']
    except KeyError:
        pass
    file.create_dataset('norm/I0', data=normto)
    file.create_dataset('norm/channel00/ims', data=ims0, compression='gzip', compression_opts=4)
    file.create_dataset('norm/channel00/ims_stddev', data=ims0_err, compression='gzip', compression_opts=4)
    file.create_dataset('norm/channel00/names', data=names0)
    file.create_dataset('norm/channel00/sum/int', data=sum_fit0, compression='gzip', compression_opts=4)
    file.create_dataset('norm/channel00/sum/bkg', data=sum_bkg0, compression='gzip', compression_opts=4)
    file.create_dataset('norm/channel00/sum/int_stddev', data=sum_fit0*sum_fit0_err, compression='gzip', compression_opts=4)
    file.create_dataset('norm/channel00/sum/bkg_stddev', data=sum_bkg0*sum_bkg0_err, compression='gzip', compression_opts=4)
    if chan02_flag:
        try:
            del file['norm/channel02']
        except KeyError:
            pass
        file.create_dataset('norm/channel02/ims', data=ims2, compression='gzip', compression_opts=4)
        file.create_dataset('norm/channel02/ims_stddev', data=ims2_err, compression='gzip', compression_opts=4)
        file.create_dataset('norm/channel02/names', data=names2)
        file.create_dataset('norm/channel02/sum/int', data=sum_fit2, compression='gzip', compression_opts=4)
        file.create_dataset('norm/channel02/sum/bkg', data=sum_bkg2, compression='gzip', compression_opts=4)
        file.create_dataset('norm/channel02/sum/int_stddev', data=sum_fit2*sum_fit2_err, compression='gzip', compression_opts=4)
        file.create_dataset('norm/channel02/sum/bkg_stddev', data=sum_bkg2*sum_bkg2_err, compression='gzip', compression_opts=4)
    file.close()
    print("Done")


##############################################################################
# delete a line from a fitted dataset (can be useful when data interpolation has to occur but there are empty pixels)
#   Note that this also removes lines from I0, I1, acquisition_time, mot1 and mot2! 
#       If you want to recover this data one has to re-initiate processing from the raw data.
def rm_line(h5file, lineid, axis=1):
    import h5py
    
    f = h5py.File(h5file, 'r+')

    # read the data and determine which data flags apply
    i0 = np.array(f['raw/I0'])
    try:
        i1 = np.array(f['raw/I1'])
        i1_flag = True
    except KeyError:
        i1_flag = False
    mot1 = np.array(f['mot1'])
    mot2 = np.array(f['mot2'])
    mot1_name = str(f['mot1'].attrs["Name"])
    mot2_name = str(f['mot2'].attrs["Name"])
    tm = np.array(f['raw/acquisition_time'])
    spectra0 = np.array(f['raw/channel00/spectra'])
    icr0 = np.array(f['raw/channel00/icr'])
    ocr0 = np.array(f['raw/channel00/ocr'])
    try:
        spectra2 = np.array(f['raw/channel02/spectra'])
        chan02_flag = True
        icr2 = np.array(f['raw/channel02/icr'])
        ocr2 = np.array(f['raw/channel02/ocr'])
    except KeyError:
        chan02_flag = False
    try:
        ims0 = np.array(f['fit/channel00/ims'])
        fit_flag = True
        if chan02_flag:
            ims2 = np.array(f['fit/channel02/ims'])
    except KeyError:
        fit_flag = False

    if fit_flag:
        ims0 = np.delete(ims0, lineid, axis+1)
    spectra0 = np.delete(spectra0, lineid, axis)
    icr0 = np.delete(icr0, lineid, axis)
    ocr0 = np.delete(ocr0, lineid, axis)
    i0 = np.delete(i0, lineid, axis)
    i1 = np.delete(i1, lineid, axis)
    mot1 = np.delete(mot1, lineid, axis)
    mot2 = np.delete(mot2, lineid, axis)
    tm = np.delete(tm, lineid, axis)
    if chan02_flag:
        if fit_flag:
            ims2 = np.delete(ims2, lineid, axis+1)
        spectra2 = np.delete(spectra2, lineid, axis)
        icr2 = np.delete(icr2, lineid, axis)
        ocr2 = np.delete(ocr2, lineid, axis)

    # save the data
    print("Writing truncated data to "+h5file+"...", end=" ")
    if fit_flag:
        del f['fit/channel00/ims']
        f.create_dataset('fit/channel00/ims', data=ims0, compression='gzip', compression_opts=4)
        if chan02_flag:
            del f['fit/channel02/ims']
            f.create_dataset('fit/channel02/ims', data=ims2, compression='gzip', compression_opts=4)
        
    del f['raw/channel00/spectra']
    del f['raw/channel00/icr']
    del f['raw/channel00/ocr']
    del f['raw/I0']
    del f['mot1']
    del f['mot2']
    del f['raw/acquisition_time']
    f.create_dataset('raw/channel00/spectra', data=spectra0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/icr', data=icr0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/ocr', data=ocr0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/I0', data=i0, compression='gzip', compression_opts=4)
    dset = f.create_dataset('mot1', data=mot1, compression='gzip', compression_opts=4)
    dset.attrs['Name'] = mot1_name
    dset = f.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
    dset.attrs['Name'] = mot2_name
    f.create_dataset('raw/acquisition_time', data=tm, compression='gzip', compression_opts=4)
    if i1_flag:
        del f['raw/I1']
        f.create_dataset('raw/I1', data=i1, compression='gzip', compression_opts=4)
    if chan02_flag:
        del f['raw/channel02/spectra']
        del f['raw/channel02/icr']
        del f['raw/channel02/ocr']
        f.create_dataset('raw/channel02/spectra', data=spectra2, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel02/icr', data=icr2, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel02/ocr', data=ocr2, compression='gzip', compression_opts=4)
        
    f.close()
    print('Done')
    
##############################################################################
# fit a batch of xrf spectra using the PyMca fitting routines. A PyMca config file should be supplied.
#   NOTE: the cfg file sscan00188_merge.h5hould use the SNIP background method! Others will fail as considered 'too slow' by the PyMca fit routine itself
#   NOTE2: setting a standard also fits the separate spectra without bulk fit! This can take a long time!!
def  fit_xrf_batch(h5file, cfgfile, standard=None, ncores=None, verbose=None):
    # perhaps channel00 and channel02 need different cfg files. Allow for tuple or array in this case.
    cfgfile = np.array(cfgfile)
    if cfgfile.size == 1:
        cfg0 = str(cfgfile)
        cfg2 = str(cfgfile)
        cfglist = str(cfgfile)
    else:
        cfg0 = str(cfgfile[0])
        cfg2 = str(cfgfile[1])
        cfglist = ' ,'.join([cfgfile[0], cfgfile[1]])
        
    # let's read the h5file structure and launch our fit.
    file = h5py.File(h5file, 'r+')
    spectra0 = file['raw/channel00/spectra']
    sumspec0 = file['raw/channel00/sumspec']
    icr0 = np.array(file['raw/channel00/icr'])
    ocr0 = np.array(file['raw/channel00/ocr'])
    spec0_shape = spectra0.shape
    if len(spec0_shape) == 2:
        spectra0 = np.array(spectra0).reshape((spec0_shape[0], 1, spec0_shape[1]))
        icr0 = np.array(icr0).reshape((icr0.shape[0], 1))
        ocr0 = np.array(ocr0).reshape((ocr0.shape[0], 1))
    nchannels0 = spectra0.shape[2]
    try:
        spectra2 = file['raw/channel02/spectra']
        sumspec2 = file['raw/channel02/sumspec']
        icr2 = np.array(file['raw/channel02/icr'])
        ocr2 = np.array(file['raw/channel02/ocr'])
        if len(spectra2.shape) == 2:
            spectra2 = np.array(spectra2).reshape((spectra2.shape[0], 1, spectra2.shape[1]))
            icr2 = np.array(icr2).reshape((icr2.shape[0], 1))
            ocr2 = np.array(ocr2).reshape((ocr2.shape[0], 1))
        nchannels2 = spectra2.shape[2]
        chan02_flag = True
    except Exception:
        chan02_flag = False

    # First correct specscan00188_merge.h5tra for icr/ocr
    # find indices where icr=0 and put those values = icr
    if np.nonzero(ocr0==0)[0].size != 0:
        ocr0[np.nonzero(ocr0==0)[0]] = icr0[np.nonzero(ocr0==0)[0]]
        if np.nonzero(icr0==0)[0].size != 0:
            icr0[np.nonzero(icr0==0)[0]] = 1 #if icr0 equal to 0, let's set to 1
            ocr0[np.nonzero(ocr0==0)[0]] = icr0[np.nonzero(ocr0==0)[0]] # let's run this again. Seems redundant, but icr could have changed.
    if chan02_flag:
        if np.nonzero(ocr2==0)[0].size != 0:
            ocr2[np.nonzero(ocr2==0)[0]] = icr2[np.nonzero(ocr2==0)[0]]
            if np.nonzero(icr2==0)[0].size != 0:
                icr2[np.nonzero(icr2==0)[0]] = 1 #if icr0 equal to 0, let's set to 1
                ocr2[np.nonzero(ocr2==0)[0]] = icr2[np.nonzero(ocr2==0)[0]] # let's run this again. Seems redundant, but icr could have changed.

    # work PyMca's magic!
    t0 = time.time()
    print("Initiating fit of <"+h5file+"> using model(s) <"+cfglist+">...", end=" ")
    n_spectra = round(spectra0.size/nchannels0, 0)
    if chan02_flag:
        n_spectra += round(spectra2.size/nchannels2, 0)
    if standard is None:
        # read and set PyMca configuration for channel00
        fastfit = FastXRFLinearFit.FastXRFLinearFit()
        try: 
            fastfit.setFitConfigurationFile(cfg0)
        except Exception:
            print("-----------------------------------------------------------------------------")
            print("Error: %s is not a valid PyMca configuration file." % cfg0)
            print("-----------------------------------------------------------------------------")
        fitresults0 = fastfit.fitMultipleSpectra(x=range(0,nchannels0), y=np.array(spectra0[:,:,:]), ysum=sumspec0)
        #fit sumspec
        config = ConfigDict.ConfigDict()
        config.read(cfg0)
        config['fit']['use_limit'] = 1 # make sure the limits of the configuration will be taken
        mcafit = ClassMcaTheory.ClassMcaTheory()
        mcafit.configure(config)
        mcafit.setData(range(0,nchannels0), sumspec0)
        mcafit.estimate()
        fitresult0_sum, result0_sum = mcafit.startfit(digest=1)
        sum_fit0 = [result0_sum[peak]["fitarea"] for peak in result0_sum["groups"]]
        sum_bkg0 = [result0_sum[peak]["statistics"]-result0_sum[peak]["fitarea"] for peak in result0_sum["groups"]]

        if chan02_flag:
            # read and set PyMca configuration for channel02
            fastfit = FastXRFLinearFit.FastXRFLinearFit()
            try: 
                fastfit.setFitConfigurationFile(cfg2)
            except Exception:
                print("-----------------------------------------------------------------------------")
                print("Error: %s is not a valid PyMca configuration file." % cfg2)
                print("-----------------------------------------------------------------------------")
            fitresults2 = fastfit.fitMultipleSpectra(x=range(0,nchannels2), y=np.array(spectra2[:,:,:]), ysum=sumspec2)
            #fit sumspec
            config = ConfigDict.ConfigDict()
            config.read(cfg2)
            config['fit']['use_limit'] = 1 # make sure the limits of the configuration will be taken
            mcafit = ClassMcaTheory.ClassMcaTheory()
            mcafit.configure(config)
            mcafit.setData(range(0,nchannels2), sumspec2)
            mcafit.estimate()
            fitresult2_sum, result2_sum = mcafit.startfit(digest=1)
            sum_fit2 = [result2_sum[peak]["fitarea"] for peak in result2_sum["groups"]]
            sum_bkg2 = [result2_sum[peak]["statistics"]-result2_sum[peak]["fitarea"] for peak in result2_sum["groups"]]

        print("Done")
        # actual fit results are contained in fitresults['parameters']
        #   labels of the elements are fitresults.labels("parameters"), first # are 'A0, A1, A2' of polynomial background, which we don't need
        peak_int0 = np.array(fitresults0['parameters'])
        names0 = fitresults0.labels("parameters")
        names0 = [n.replace('Scatter Peak000', 'Rayl') for n in names0]
        names0 = np.array([n.replace('Scatter Compton000', 'Compt') for n in names0])
        cutid0 = 0
        for i in range(names0.size):
            if names0[i] == 'A'+str(i):
                cutid0 = i+1
        if chan02_flag:
            peak_int2 = np.array(fitresults2['parameters'])
            names2 = fitresults2.labels("parameters")
            names2 = [n.replace('Scatter Peak000', 'Rayl') for n in names2]
            names2 = np.array([n.replace('Scatter Compton000', 'Compt') for n in names2])
            cutid2 = 0
            for i in range(names2.size):
                if names2[i] == 'A'+str(i):
                    cutid2 = i+1
        # for i in range(peak_int0.shape[0]):
        #     plt.imshow(peak_int0[i,:,:])
        #     plt.title(fitresults0.labels("parameters")[i])
        #     plt.colorbar()
        #     plt.show()
        
    else: #standard is not None; this is a srm spectrum and as such we would like to obtain the background values.
        # channel00
        config = ConfigDict.ConfigDict()
        config.read(cfg0)
        config['fit']['use_limit'] = 1 # makmot1e sure the limits of the configuration will be taken
        mcafit = ClassMcaTheory.ClassMcaTheory()
        mcafit.configure(config)
        if ncores is None or ncores == -1 or ncores == 0:
            ncores = multiprocessing.cpu_count()-1
        spec_chansum = np.sum(spectra0, axis=2)
        spec2fit_id = np.array(np.where(spec_chansum.ravel() > 0.)).squeeze()
        spec2fit = np.array(spectra0).reshape((spectra0.shape[0]*spectra0.shape[1], spectra0.shape[2]))[spec2fit_id,:]
        if spectra0.shape[0]*spectra0.shape[1] > 1:
            print("Using "+str(ncores)+" cores...")
            pool = multiprocessing.Pool(processes=ncores)
            results, groups = zip(*pool.map(partial(Pymca_fit, mcafit=mcafit, verbose=verbose), spec2fit))
            results = list(results)
            groups = list(groups)
            if groups[0] is None: #first element could be None, so let's search for first not-None item.
                for i in range(0, np.array(groups, dtype='object').shape[0]):
                    if groups[i] is not None:
                        groups[0] = groups[i]
                        break
            none_id = [i for i, x in enumerate(results) if x is None]
            if none_id != []:
                for i in range(0, np.array(none_id).size):
                    results[none_id[i]] = [0]*np.array(groups[0]).shape[0] # set None to 0 values
            peak_int0 = np.zeros((spectra0.shape[0]*spectra0.shape[1], np.array(groups[0]).shape[0]))
            peak_int0[spec2fit_id,:] = np.array(results).reshape((spec2fit_id.size, np.array(groups[0]).shape[0]))
            peak_int0 = np.moveaxis(peak_int0.reshape((spectra0.shape[0], spectra0.shape[1], np.array(groups[0]).shape[0])),-1,0)
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
        names0 = np.array([n.replace('Scatter Compton000', 'Compt') for n in names0])
        cutid0 = 0
        for i in range(names0.size):
            if names0[i] == 'A'+str(i):
                cutid0 = i+1
        sum_fit0 = [result0_sum[peak]["fitarea"] for peak in result0_sum["groups"]]
        sum_bkg0 = [result0_sum[peak]["statmot1istics"]-result0_sum[peak]["fitarea"] for peak in result0_sum["groups"]]

        if chan02_flag:
            # channel02
            config = ConfigDict.ConfigDict()
            config.read(cfg2)
            config['fit']['use_limit'] = 1 # make sure the limits of the configuration will be taken
            mcafit = ClassMcaTheory.ClassMcaTheory()
            mcafit.configure(config)
            spec_chansum = np.sum(spectra2, axis=2)
            spec2fit_id = np.array(np.where(spec_chansum.ravel() > 0.)).squeeze()
            spec2fit = np.array(spectra2).reshape((spectra2.shape[0]*spectra2.shape[1], spectra2.shape[2]))[spec2fit_id,:]
            if spectra2.shape[0]*spectra2.shape[1] > 1:
                pool = multiprocessing.Pool(processes=ncores)
                results, groups = zip(*pool.map(partial(Pymca_fit, mcafit=mcafit, verbose=verbose), spec2fit))
                results = list(results)
                groups = list(groups)
                if groups[0] is None: #first element could be None, so let's search for first not-None item.
                    for i in range(0, np.array(groups, dtype='object').shape[0]):
                        if groups[i] is not None:
                            groups[0] = groups[i]
                            break
                none_id = [i for i, x in enumerate(results) if x is None]
                if none_id != []:
                    for i in range(0, np.array(none_id).size):
                        results[none_id[i]] = [0]*np.array(groups[0]).shape[0] # set None to 0 values
                peak_int2 = np.zeros((spectra2.shape[0]*spectra2.shape[1], np.array(groups[0]).shape[0]))
                peak_int2[spec2fit_id,:] = np.array(results).reshape((spec2fit_id.size, np.array(groups[0]).shape[0]))
                peak_int2 = np.moveaxis(peak_int2.reshape((spectra2.shape[0], spectra2.shape[1], np.array(groups[0]).shape[0])),-1,0)
                peak_int2[np.isnan(peak_int2)] = 0.
                pool.close()
                pool.join()
            else:
                mcafit.setData(range(0,nchannels0), spectra2[0,0,:])
                mcafit.estimate()
                fitresult2, result2 = mcafit.startfit(digest=1)
                peak_int2 = np.asarray([result2[peak]["fitarea"] for peak in result2["groups"]])
                peak_int2 = peak_int2.reshape((peak_int2.shape[0],1,1))

            #fit sumspec
            mcafit.setData(range(0,nchannels2), sumspec2)
            mcafit.estimate()
            fitresult2_sum, result2_sum = mcafit.startfit(digest=1)
            names2 = [n.replace('Scatter Peak000', 'Rayl') for n in result2_sum["groups"]]
            names2 = np.array([n.replace('Scatter Compton000', 'Compt') for n in names2])
            cutid2 = 0
            for i in range(names2.size):
                if names2[i] == 'A'+str(i):
                    cutid2 = i+1
            sum_fit2 = [result2_sum[peak]["fitarea"] for peak in result2_sum["groups"]]
            sum_bkg2 = [result2_sum[peak]["statistics"]-result2_sum[peak]["fitarea"] for peak in result2_sum["groups"]]
    print("Fit finished after "+str(time.time()-t0)+" seconds for "+str(n_spectra)+" spectra.")

    ims0 = peak_int0[cutid0:,:,:]
    if chan02_flag:
        ims2 = peak_int2[cutid2:,:,:]

    # correct for deadtime  
    # check if icr/ocr values are appropriate!
    if np.average(ocr0/icr0) > 1.:
        print("ERROR: ocr0/icr0 is larger than 1!")
    if chan02_flag:
        if np.average(ocr2/icr2) > 1.:
            print("ERROR: ocr2/icr2 is larger than 1!")
    #TODO: something goes wrong with the dimensions in case of 1D scan.
    if icr0.shape[0] > ims0.shape[1]:
        icr0 = icr0[0:ims0.shape[1],:]
        ocr0 = ocr0[0:ims0.shape[1],:]
    for i in range(names0.size):
        ims0[i,:,:] = ims0[i,:,:] * icr0/ocr0
    if chan02_flag:
        if icr2.shape[0] > ims2.shape[1]:
            icr2 = icr2[0:ims0.shape[1],:]
            ocr2 = ocr2[0:ims0.shape[1],:]
        for i in range(names2.size):        
            ims2[i,:,:] = ims2[i,:,:] * icr2/ocr2
    sum_fit0 = np.array(sum_fit0)*np.sum(icr0)/np.sum(ocr0)
    sum_bkg0 = np.array(sum_bkg0)*np.sum(icr0)/np.sum(ocr0)
    if len(spec0_shape) == 2:
        ims0 = np.squeeze(ims0)
    if chan02_flag:
        sum_fit2 = np.array(sum_fit2)*np.sum(icr2)/np.sum(ocr2)
        sum_bkg2 = np.array(sum_bkg2)*np.sum(icr2)/np.sum(ocr2)
        if len(spec0_shape) == 2:
            ims2 = np.squeeze(ims2)

    # save the fitted data
    print("Writing fit data to "+h5file+"...", end=" ")
    try:
        del file['fit/channel00/ims']
        del file['fit/channel00/names']
        del file['fit/channel00/cfg']
        del file['fit/channel00/sum/int']
        del file['fit/channel00/sum/bkg']
    except Exception:
        pass
    file.create_dataset('fit/channel00/ims', data=ims0, compression='gzip', compression_opts=4)
    file.create_dataset('fit/channel00/names', data=[n.encode('utf8') for n in names0[cutid0:]])
    file.create_dataset('fit/channel00/cfg', data=cfg0)
    file.create_dataset('fit/channel00/sum/int', data=sum_fit0, compression='gzip', compression_opts=4)
    file.create_dataset('fit/channel00/sum/bkg', data=sum_bkg0, compression='gzip', compression_opts=4)
    if chan02_flag:
        try:
            del file['fit/channel02/ims']
            del file['fit/channel02/names']
            del file['fit/channel02/cfg']
            del file['fit/channel02/sum/int']
            del file['fit/channel02/sum/bkg']
        except Exception:
            pass
        file.create_dataset('fit/channel02/ims', data=ims2, compression='gzip', compression_opts=4)
        file.create_dataset('fit/channel02/names', data=[n.encode('utf8') for n in names2[cutid2:]])
        file.create_dataset('fit/channel02/cfg', data=cfg2)
        file.create_dataset('fit/channel02/sum/int', data=sum_fit2, compression='gzip', compression_opts=4)
        file.create_dataset('fit/channel02/sum/bkg', data=sum_bkg2, compression='gzip', compression_opts=4)
    file.close()
    print('Done')

##############################################################################
# def Pymca_fit(spectra, mcafit):
def Pymca_fit(spectra, mcafit, verbose=None):

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
# add together multiple h5 files of the same dimensions...
#   Make sure it is also ordered similarly etc...
#   Use at your own risk.
def add_h5s(h5files, newfilename):
    if type(h5files) is not type(list()):
        print("ERROR: h5files must be a list!")
        return
    else:
        for i in range(len(h5files)):
            if i == 0:
                print("Reading "+h5files[i]+"...", end="")
                f = h5py.File(h5files[i],'r')
                cmd = f['cmd'][()].decode('utf8')
                mot1 = np.array(f['mot1'])
                mot1_name = str(f['mot1'].attrs["Name"])
                mot2 = np.array(f['mot2'])
                mot2_name = str(f['mot2'].attrs["Name"])
                i0 = np.array(f['raw/I0'])
                try:
                    i1 = np.array(f['raw/I1'])
                    i1flag = True
                except KeyError:
                    i1flag = False
                tm = np.array(f['raw/acquisition_time'])
                icr0 = np.array(f['raw/channel00/icr'])
                ocr0 = np.array(f['raw/channel00/ocr'])
                spectra0 = np.array(f['raw/channel00/spectra'])
                maxspec0 = np.array(f['raw/channel00/maxspec'])
                sumspec0 = np.array(f['raw/channel00/sumspec'])
                try:
                    icr2 = np.array(f['raw/channel02/icr'])
                    ocr2 = np.array(f['raw/channel02/ocr'])
                    spectra2 = np.array(f['raw/channel02/spectra'])
                    maxspec2 = np.array(f['raw/channel02/maxspec'])
                    sumspec2 = np.array(f['raw/channel02/sumspec'])
                    ch2flag = True
                except KeyError:
                    ch2flag = False
                f.close()
                print("Done")
            else:
                print("Reading "+h5files[i]+"...", end="")
                f = h5py.File(h5files[i],'r')
                mot1 += np.array(f['mot1'])
                mot2 += np.array(f['mot2'])
                i0 += np.array(f['raw/I0'])
                if i1flag:
                    i1 += np.array(f['raw/I1'])
                tm += np.array(f['raw/acquisition_time'])
                icr0 += np.array(f['raw/channel00/icr'])
                ocr0 += np.array(f['raw/channel00/ocr'])
                spectra0 += np.array(f['raw/channel00/spectra'])
                maxspec_tmp = np.array(f['raw/channel00/maxspec'])
                for j in range(len(maxspec0)):
                    if maxspec_tmp[j] > maxspec0[j]:
                        maxspec0[j] = maxspec_tmp[j]
                sumspec0 += np.array(f['raw/channel00/sumspec'])
                if ch2flag:
                    icr2 += np.array(f['raw/channel02/icr'])
                    ocr2 += np.array(f['raw/channel02/ocr'])
                    spectra2 += np.array(f['raw/channel02/spectra'])
                    maxspec_tmp = np.array(f['raw/channel02/maxspec'])
                    for j in range(len(maxspec2)):
                        if maxspec_tmp[j] > maxspec2[j]:
                            maxspec2[j] = maxspec_tmp[j]
                    sumspec2 += np.array(f['raw/channel02/sumspec'])
                    ch2flag = True
                f.close()
                print("Done")
        # make the motor positions the average
        mot1 /= len(h5files)
        mot2 /= len(h5files)
        # write the new file
        print("writing "+newfilename+"...", end="")
        f = h5py.File(newfilename, 'w')
        f.create_dataset('cmd', data=cmd)
        f.create_dataset('raw/channel00/spectra', data=spectra0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/icr', data=icr0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/ocr', data=ocr0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/sumspec', data=sumspec0, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel00/maxspec', data=maxspec0, compression='gzip', compression_opts=4)
        if ch2flag:
            f.create_dataset('raw/channel02/spectra', data=spectra2, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel02/icr', data=icr2, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel02/ocr', data=ocr2, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel02/sumspec', data=sumspec2, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel02/maxspec', data=maxspec2, compression='gzip', compression_opts=4)
        f.create_dataset('raw/I0', data=i0, compression='gzip', compression_opts=4)
        if i1flag:
            f.create_dataset('raw/I1', data=i1, compression='gzip', compression_opts=4)
        dset = f.create_dataset('mot1', data=mot1, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot1_name
        dset = f.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot2_name
        f.create_dataset('raw/acquisition_time', data=tm, compression='gzip', compression_opts=4)
        f.close()                   
        print("Done")

##############################################################################
# Read the spectra nexus files
def read_P06_spectra(file, sc_id, ch):
    if type(ch[0]) is str:
        # Reading the spectra files, icr and ocr
        print("Reading " +sc_id+"/"+ch[0]+"/"+file +"...", end=" ")
        f = h5py.File(sc_id+"/"+ch[0]+"/"+file, 'r')
        if type(ch[1]) is str:
            spe0_arr = f['entry/instrument/xspress3/'+ch[1]+'/histogram'][:]
            try:
                icr0_arr = f['entry/instrument/xspress3/'+ch[1]+'/scaler/allEvent'][:]
                ocr0_arr = f['entry/instrument/xspress3/'+ch[1]+'/scaler/allGood'][:]
            except KeyError:
                icr0_arr = f['entry/instrument/xspress3/'+ch[1]+'/scaler/allevent'][:]
                ocr0_arr = f['entry/instrument/xspress3/'+ch[1]+'/scaler/allgood'][:]
        elif type(ch[1]) is list: #if multiple channels provided we want to add them 
            spe0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0]+'/histogram'][:])
            try:
                icr0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allEvent'][:])
                ocr0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allGood'][:])
            except KeyError:
                icr0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allevent'][:])
                ocr0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allgood'][:])
            for chnl in ch[1][1:]:
                spe0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/histogram'][:])
                try:
                    icr0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/scaler/allEvent'][:])
                    ocr0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/scaler/allGood'][:])
                except KeyError:
                    icr0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/scaler/allevent'][:])
                    ocr0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/scaler/allgood'][:])
        f.close()
        print("read")
    elif type(ch[0]) is list:
        # Reading the spectra files, icr and ocr
        dev = ch[0][0]
        print("Reading " +sc_id+"/"+dev+"/"+file +"...", end=" ")
        f = h5py.File(sc_id+"/"+dev+"/"+file, 'r')
        # read spectra and icr/ocr from first device (ch[0][0]) so we can later add the rest
        #   as ch[0] is a list, ch[1] is also expected to be a list!
        if type(ch[1][0]) is str:
            spe0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0]+'/histogram'][:])
            try:
                icr0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allEvent'][:])
                ocr0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allGood'][:])
            except KeyError:
                icr0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allevent'][:])
                ocr0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0]+'/scaler/allgood'][:])
        elif type(ch[1][0]) is list: #if multiple channels provided for this device we want to add them 
            spe0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0][0]+'/histogram'][:])
            try:
                icr0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0][0]+'/scaler/allEvent'][:])
                ocr0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0][0]+'/scaler/allGood'][:])
            except KeyError:
                icr0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0][0]+'/scaler/allevent'][:])
                ocr0_arr = np.array(f['entry/instrument/xspress3/'+ch[1][0][0]+'/scaler/allgood'][:])
            for chnl in ch[1][0][1:]:
                spe0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/histogram'][:])
                try:
                    icr0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/scaler/allEvent'][:])
                    ocr0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/scaler/allGood'][:])
                except KeyError:
                    icr0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/scaler/allevent'][:])
                    ocr0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/scaler/allgood'][:])
        f.close()
        print("read")
        # Now let's read the following spectra/devices
        for i in range(len(ch[0])-1):
            dev = ch[0][i+1]
            # Reading the spectra files, icr and ocr
            print("Reading " +sc_id+"/"+dev+"/"+file +"...", end=" ")
            f = h5py.File(sc_id+"/"+dev+"/"+file, 'r')
            if type(ch[1][i+1]) is str:
                spe0_arr += np.array(f['entry/instrument/xspress3/'+ch[1][i+1]+'/histogram'][:])
                try:
                    icr0_arr += np.array(f['entry/instrument/xspress3/'+ch[1][i+1]+'/scaler/allEvent'][:])
                    ocr0_arr += np.array(f['entry/instrument/xspress3/'+ch[1][i+1]+'/scaler/allGood'][:])
                except KeyError:
                    icr0_arr += np.array(f['entry/instrument/xspress3/'+ch[1][i+1]+'/scaler/allevent'][:])
                    ocr0_arr += np.array(f['entry/instrument/xspress3/'+ch[1][i+1]+'/scaler/allgood'][:])
            elif type(ch[1][i+1]) is list: #if multiple channels provided for this device we want to add them 
                for chnl in ch[1][i+1][:]:
                    spe0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/histogram'][:])
                    try:
                        icr0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/scaler/allEvent'][:])
                        ocr0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/scaler/allGood'][:])
                    except KeyError:
                        icr0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/scaler/allevent'][:])
                        ocr0_arr += np.array(f['entry/instrument/xspress3/'+chnl+'/scaler/allgood'][:])
            f.close()    
            print("read")
    return spe0_arr, icr0_arr, ocr0_arr
    
##############################################################################
# Merges separate P06 nxs files to 1 handy h5 file containing 2D array of spectra, relevant motor positions, I0 counter, ICR, OCR and mesaurement time.
def MergeP06Nxs(scanid, sort=True, ch0=['xspress3_01','channel00'], ch2=['xspress3_01','channel02'], readas1d=False):
    scanid = np.array(scanid)
    if scanid.size == 1:
        scan_suffix = '/'.join(str(scanid).split('/')[0:-2])+'/scan'+str(scanid).split("_")[-1]
    else:
        scan_suffix = '/'.join(scanid[0].split('/')[0:-2])+'/scan'+str(scanid[0]).split("_")[-1]+'-'+str(scanid[-1]).split("_")[-1]

    for k in range(scanid.size):
        if scanid.size == 1:
            sc_id = str(scanid)
        else:
            sc_id = str(scanid[k])
        # file with name scanid contains info on scan command
        f = h5py.File(sc_id+'.nxs', 'r')
        scan_cmd = str(f['scan/program_name'].attrs["scan_command"][:])
        scan_cmd = np.array(scan_cmd.strip("[]'").split(" "))
        print(' '.join(scan_cmd))
        f.close()

        spectra0 = []
        icr0 = []
        ocr0 = []
        spectra2 = []
        icr2 = []
        ocr2 = []
        i0 = []
        i1 = []
        tm = []
        mot1 = []
        mot2 = []
        files = list("")
        # actual spectrum scan files are in dir scanid/scan_0XXX/xspress3_01
        for file in sorted(os.listdir(sc_id+"/adc01")):
            if file.endswith(".nxs"):
                files.append(file)
        for file in files:
            spe0_arr, icr0_arr, ocr0_arr = read_P06_spectra(file, sc_id, ch0)
            if ch2 is not None:
                spe2_arr, icr2_arr, ocr2_arr = read_P06_spectra(file, sc_id, ch2)
            for i in range(spe0_arr.shape[0]):
                spectra0.append(spe0_arr[i,:])
                icr0.append(icr0_arr[i])
                ocr0.append(ocr0_arr[i])
                if ch2 is not None:
                    spectra2.append(spe2_arr[i,:])
                    icr2.append(icr2_arr[i])
                    ocr2.append(ocr2_arr[i])
        for file in files:
            # Reading I0 and measurement time data
            print("Reading " +sc_id+"/adc01/"+file +"...", end=" ")
            f = h5py.File(sc_id+"/adc01/"+file, 'r')
            i0_arr = f['entry/data/value1'][:]
            i1_arr = f['entry/data/value2'][:]
            tm_arr = f['entry/data/exposuretime'][:]
            for i in range(i0_arr.shape[0]):
                i0.append(i0_arr[i])
                i1.append(i1_arr[i])
                tm.append(tm_arr[i])
            f.close()
            print("read")
        # actual pilcgenerator files can be different structure depending on type of scan
        files = list("")
        #TODO: we can clean this up, into 1 try block, pilctrigger can run from 01 to 05...
        try:
            for file in sorted(os.listdir(sc_id+"/pilctriggergenerator_03")):
                if file.endswith(".nxs"):
                    files.append(file)
            pilcid = "/pilctriggergenerator_03/"
        except FileNotFoundError:
            try:
                for file in sorted(os.listdir(sc_id+"/pilctriggergenerator_02")):
                    if file.endswith(".nxs"):
                        files.append(file)
                pilcid = "/pilctriggergenerator_02/"
            except FileNotFoundError:
                for file in sorted(os.listdir(sc_id+"/pilctriggergenerator_01")):
                    if file.endswith(".nxs"):
                        files.append(file)
                pilcid = "/pilctriggergenerator_01/"
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
        for file in files:
            # Reading motor positions. Assumes only 2D scans are performed (stores encoder1 and 2 values)
            print("Reading " +sc_id+pilcid+file +"...", end=" ")
            f = h5py.File(sc_id+pilcid+file, 'r')
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
                    mot1_arr = mot1a_contrib*(np.array(mot1a)-pivot[0])+mot1b_contrib*(np.array(mot1b)-pivot[1]) + pivot[0] #just took first as in this case it's twice the same i.e. [250,250]
                    mot1_name = str(scan_cmd[1])
                except Exception:
                    try:
                        if scan_cmd[0] == 'timescanc' or scan_cmd[0] == 'timescan' or scan_cmd[0] == 'dscan':
                            print("Warning: timescan(c) command; using "+str(enc_names[0])+" encoder value...", end=" ")
                            mot1_arr = enc_vals[0]
                            mot1_name = enc_names[0]
                        else:
                            f2 = h5py.File(sc_id+'.nxs','r')
                            mot1_arr = np.array(f2["scan/data/"+str(scan_cmd[1])][:])
                            mot1_name = str(scan_cmd[1])
                            f2.close()
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
            if scan_cmd.shape[0] > 6 and scan_cmd[5] in enc_names:
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
                    mot2_arr = mot2a_contrib*(np.array(mot2a)-pivot[0])+mot2b_contrib*(np.array(mot2b)-pivot[1]) + pivot[0] #just took first as in this case it's twice the same i.e. [250,250]
                    mot2_name = str(scan_cmd[5])
                except Exception:
                    try:
                        if scan_cmd[0] == 'timescanc' or scan_cmd[0] == 'timescan' or scan_cmd[0] == 'dscan':
                            print("Warning: timescan(c) command; using "+str(enc_names[1])+" encoder value...", end=" ")
                            mot2_arr = enc_vals[1]
                            mot2_name = enc_names[1]
                        else:
                            f2 = h5py.File(sc_id+'.nxs','r')
                            mot2_arr = np.array(f2["scan/data/"+str(scan_cmd[5])][:])
                            mot2_name = str(scan_cmd[5])
                            f2.close()
                    except KeyError:
                        if scan_cmd[0] == 'timescanc' or scan_cmd[0] == 'timescan' or scan_cmd[0] == 'dscan':
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
            for i in range(mot1_arr.shape[0]):
                mot1.append(mot1_arr[i])
                mot2.append(mot2_arr[i])
            f.close()
            print("read")
        # try to reshape if possible (for given scan_cmd and extracted data points), else just convert to np.array
        # let's translate scan command to figure out array dimensions we want to fill
        #   1D scan (ascan, dscan, timescan, ...) contain 7 parts, i.e. dscan samx 0 1 10 1 False
        #       sometimes False at end appears to be missing
        if scan_cmd[0][0] == 'c' and  scan_cmd[0] != 'cnt':
            xdim = int(scan_cmd[4])
        elif scan_cmd[0] == 'cnt':
            xdim = 1
        elif scan_cmd[0] == 'timescanc' or scan_cmd[0] == 'timescan':
            xdim = int(scan_cmd[1])+1
        else:
            xdim = int(scan_cmd[4])+1
        ydim = 1
        if scan_cmd.shape[0] > 7:
            ydim = int(scan_cmd[8])+1
        if np.asarray(spectra0).shape[0] == xdim*ydim and readas1d is not True:
            spectra0 = np.asarray(spectra0)
            spectra0 = spectra0.reshape((ydim, xdim, spectra0.shape[1]))
            icr0 = np.asarray(icr0).reshape((ydim, xdim))
            ocr0 = np.asarray(ocr0).reshape((ydim, xdim))
            i0 = np.asarray(i0).reshape((ydim, xdim))
            i1 = np.asarray(i1).reshape((ydim, xdim))
            tm = np.asarray(tm).reshape((ydim, xdim))
            if np.asarray(mot1).shape[0] < xdim*ydim:
                print("Warning: mot1 has less than "+str(xdim*ydim)+" elements; padding with zero.")
                mot1 = np.concatenate((np.asarray(mot1), np.zeros(xdim*ydim-np.asarray(mot1).shape[0])))
            if np.asarray(mot2).shape[0] < xdim*ydim:
                print("Warning: mot2 has less than "+str(xdim*ydim)+" elements; padding with zero.")
                mot2 = np.concatenate((np.asarray(mot2), np.zeros(xdim*ydim-np.asarray(mot2).shape[0])))
            mot1 = np.asarray(mot1[0:xdim*ydim]).reshape((ydim, xdim))
            mot2 = np.asarray(mot2[0:xdim*ydim]).reshape((ydim, xdim))
            timetrig = False
        elif np.asarray(spectra0).shape[0] < xdim*ydim and readas1d is not True:
            spectra0 = np.asarray(spectra0)
            zerosize = xdim*ydim-spectra0.shape[0]
            zeros = np.zeros((zerosize, spectra0.shape[1]))
            spectra0 = np.concatenate((spectra0, zeros)).reshape((ydim, xdim, spectra0.shape[1]))
            zeros = np.zeros((zerosize))
            icr0 = np.concatenate((np.asarray(icr0), zeros)).reshape((ydim, xdim))
            ocr0 = np.concatenate((np.asarray(ocr0), zeros)).reshape((ydim, xdim))
            i0 = np.concatenate((np.asarray(i0), zeros)).reshape((ydim, xdim))
            i1 = np.concatenate((np.asarray(i1), zeros)).reshape((ydim, xdim))
            tm = np.concatenate((np.asarray(tm), zeros)).reshape((ydim, xdim))
            mot1 = np.concatenate((np.asarray(mot1), zeros))[0:xdim*ydim].reshape((ydim, xdim))
            mot2 = np.concatenate((np.asarray(mot2), zeros))[0:xdim*ydim].reshape((ydim, xdim))
            timetrig = False
        else:            
            spectra0 = np.asarray(spectra0)
            icr0 = np.asarray(icr0)
            ocr0 = np.asarray(ocr0)
            if ch2 is not None:
                spectra2 = np.asarray(spectra2)
            i0 = np.asarray(i0)
            i1 = np.asarray(i1)
            tm = np.asarray(tm)
            mot1 = np.asarray(mot1)
            mot2 = np.asarray(mot2)
            # in this case we should never sort or flip data
            sort = False
            timetrig = True
        if ch2 is not None:
            if np.asarray(spectra2).shape[0] == xdim*ydim and readas1d is not True:
                spectra2 = np.asarray(spectra2)
                spectra2 = spectra2.reshape((ydim, xdim, spectra2.shape[1]))
                icr2 = np.asarray(icr2).reshape((ydim, xdim))
                ocr2 = np.asarray(ocr2).reshape((ydim, xdim))
            elif np.asarray(spectra2).shape[0] < xdim*ydim and readas1d is not True:
                spectra2 = np.asarray(spectra2)
                zerosize = xdim*ydim-spectra2.shape[0]
                zeros = np.zeros((zerosize, spectra2.shape[1]))
                spectra2 = np.asarray(spectra2)
                spectra2 = np.concatenate((spectra2, zeros)).reshape((ydim, xdim, spectra2.shape[1]))
                zeros = np.zeros((zerosize))
                icr2 = np.concatenate((np.asarray(icr2), zeros)).reshape((ydim, xdim))
                ocr2 = np.concatenate((np.asarray(ocr2), zeros)).reshape((ydim, xdim))
            else:            
                spectra2 = np.asarray(spectra2)
                icr2 = np.asarray(icr2)
                ocr2 = np.asarray(ocr2)
        # store data arrays so they can be concatenated in case of multiple scans
        if k == 0:
            spectra0_tmp = spectra0
            del spectra0
            icr0_tmp = icr0
            ocr0_tmp = ocr0
            if ch2 is not None:
                spectra2_tmp = spectra2
                del spectra2
                icr2_tmp = icr2
                ocr2_tmp = ocr2
            mot1_tmp = mot1
            mot2_tmp = mot2
            i0_tmp = i0
            i1_tmp = i1
            tm_tmp = tm
        else:
            spectra0_tmp = np.concatenate((spectra0_tmp,spectra0), axis=0)
            del spectra0
            icr0_tmp = np.concatenate((icr0_tmp,icr0), axis=0)
            ocr0_tmp = np.concatenate((ocr0_tmp,ocr0), axis=0)
            if ch2 is not None:
                spectra2_tmp = np.concatenate((spectra2_tmp,spectra2), axis=0)
                del spectra2
                icr2_tmp = np.concatenate((icr2_tmp,icr2), axis=0)
                ocr2_tmp = np.concatenate((ocr2_tmp,ocr2), axis=0)
            mot1_tmp = np.concatenate((mot1_tmp,mot1), axis=0)
            mot2_tmp = np.concatenate((mot2_tmp,mot2), axis=0)
            i0_tmp = np.concatenate((i0_tmp,i0), axis=0)
            i1_tmp = np.concatenate((i1_tmp,i1), axis=0)
            tm_tmp = np.concatenate((tm_tmp,tm), axis=0)

    # redefine as original arrays for further processing
    spectra0 = spectra0_tmp
    del spectra0_tmp
    icr0 = icr0_tmp 
    ocr0 = ocr0_tmp 
    i0 = i0_tmp 
    i1 = i1_tmp 
    tm = tm_tmp 
    mot1 = mot1_tmp 
    mot2 = mot2_tmp 

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
    f = h5py.File(scan_suffix+"_merge.h5", 'w')
    f.create_dataset('cmd', data=' '.join(scan_cmd))
    f.create_dataset('raw/channel00/spectra', data=np.squeeze(spectra0), compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/icr', data=np.squeeze(icr0), compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/ocr', data=np.squeeze(ocr0), compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/sumspec', data=np.squeeze(sumspec0), compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/maxspec', data=np.squeeze(maxspec0), compression='gzip', compression_opts=4)
    f.create_dataset('raw/I0', data=np.squeeze(i0), compression='gzip', compression_opts=4)
    f.create_dataset('raw/I1', data=np.squeeze(i1), compression='gzip', compression_opts=4)
    f.create_dataset('raw/acquisition_time', data=np.squeeze(tm), compression='gzip', compression_opts=4)
    f.close()
    del spectra0

    # redefine as original arrays for further processing
    if ch2 is not None:
        spectra2 = spectra2_tmp 
        del spectra2_tmp
        icr2 = icr2_tmp 
        ocr2 = ocr2_tmp 

    # for continuous scans, the mot1 position runs in snake-type fashion
    #   so we need to sort the positions line per line and adjust all other data accordingly
    if sort is True:
        for i in range(mot1[:,0].size):
            sort_id = np.argsort(mot1[i,:])
            if ch2 is not None:
                spectra2[i,:,:] = spectra2[i,sort_id,:]
                icr2[i,:] = icr2[i,sort_id]
                ocr2[i,:] = ocr2[i,sort_id]
            mot1[i,:] = mot1[i,sort_id]
            mot2[i,:] = mot2[i,sort_id]
    
        # To make sure (especially when merging scans) sort mot2 as well
        for i in range(mot2[0,:].size):
            sort_id = np.argsort(mot2[:,i])
            if ch2 is not None:
                spectra2[:,i,:] = spectra2[sort_id,i,:]
                icr2[:,i] = icr2[sort_id,i]
                ocr2[:,i] = ocr2[sort_id,i]
            mot1[:,i] = mot1[sort_id,i]
            mot2[:,i] = mot2[sort_id,i]

    # calculate sumspec and maxspec spectra
    if ch2 is not None:
        if timetrig is False:
            sumspec2 = np.sum(spectra2[:], axis=(0,1))
            maxspec2 = np.zeros(sumspec2.shape[0])
            for i in range(sumspec2.shape[0]):
                maxspec2[i] = spectra2[:,:,i].max()
        else:
            sumspec2 = np.sum(spectra2[:], axis=(0))
            maxspec2 = np.zeros(sumspec2.shape[0])
            for i in range(sumspec2.shape[0]):
                maxspec2[i] = spectra2[:,i].max()

    # Hooray! We read all the information! Let's write it to a separate file
    f = h5py.File(scan_suffix+"_merge.h5", 'r+')
    if ch2 is not None:
        f.create_dataset('raw/channel02/spectra', data=np.squeeze(spectra2), compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel02/icr', data=np.squeeze(icr2), compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel02/ocr', data=np.squeeze(ocr2), compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel02/sumspec', data=np.squeeze(sumspec2), compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel02/maxspec', data=np.squeeze(maxspec2), compression='gzip', compression_opts=4)
    dset = f.create_dataset('mot1', data=np.squeeze(mot1), compression='gzip', compression_opts=4)
    dset.attrs['Name'] = mot1_name
    dset = f.create_dataset('mot2', data=np.squeeze(mot2), compression='gzip', compression_opts=4)
    dset.attrs['Name'] = mot2_name
    f.close()
    if ch2 is not None:
        del spectra2
    print("ok")

##############################################################################
# convert id15a bliss h5 format to our h5 structure file
#   syntax: h5id15convert('exp_file.h5', '3.1', (160,1), mot1_name='hry', mot2_name='hrz')
#   when scanid is an array or list of multiple elements, the images will be stitched together to 1 file
def h5id15convert(h5id15, scanid, scan_dim, mot1_name='hry', mot2_name='hrz', ch0id='falconx_det0', ch2id='falconx2_det0', i0id='fpico2', i0corid=None, i1id='fpico3', i1corid=None, icrid='trigger_count_rate', ocrid='event_count_rate', atol=None, sort=True):
    scan_dim = np.array(scan_dim)
    scanid = np.array(scanid)
    if scan_dim.size == 1:
        scan_dim = np.array((scan_dim, 1))
    if atol is None:
        atol = 1e-4

    if scanid.size == 1:
        scan_suffix = '_scan'+str(scanid).split(".")[0]
    else:
        scan_suffix = '_scan'+str(scanid[0]).split(".")[0]+'-'+str(scanid[-1]).split(".")[0]

    if np.array(h5id15).size == 1:
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
            f = h5py.File(file, 'r')
            scan_cmd = f[sc_id+'/title'][()].decode('utf8')+' '+mot1_name+' '+str(sc_dim[0])+' '+mot2_name+' '+str(sc_dim[1])
            spectra0_temp = np.array(f[sc_id+'/measurement/'+ch0id])
            spectra0 = np.zeros((sc_dim[0], sc_dim[1], spectra0_temp.shape[1]))
            for i in range(0, spectra0_temp.shape[1]):
                spectra0[:,:,i] = spectra0_temp[:sc_dim[0]*sc_dim[1],i].reshape(sc_dim)
            icr0 = np.array(f[sc_id+'/measurement/'+ch0id+'_'+icrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
            ocr0 = np.array(f[sc_id+'/measurement/'+ch0id+'_'+ocrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
            if ch2id is not None:
                spectra2_temp = np.array(f[sc_id+'/measurement/'+ch2id])
                spectra2 = np.zeros((sc_dim[0], sc_dim[1], spectra2_temp.shape[1]))
                for i in range(0, spectra2_temp.shape[1]):
                    spectra2[:,:,i] = spectra2_temp[:sc_dim[0]*sc_dim[1],i].reshape(sc_dim)
                icr2 = np.array(f[sc_id+'/measurement/'+ch2id+'_'+icrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                ocr2 = np.array(f[sc_id+'/measurement/'+ch2id+'_'+ocrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
            i0 = np.array(f[sc_id+'/measurement/'+i0id][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
            if i0corid is not None:
                try:
                    i0cor = np.average(np.array(f[str(scanid).split('.')[0]+'.2/measurement/'+i0corid][:]))
                except KeyError:
                    try:
                        i0cor = np.average(np.array(f[str(scanid)+'/instrument/'+i0corid+'/data'][:]))
                    except KeyError:
                        print("***ERROR: no viable i0cor value obtained. Set to 1.")
                        i0cor = 1.
                i0 = i0/np.average(i0) * i0cor
            if i1id is not None:
                i1 = np.array(f[sc_id+'/measurement/'+i1id][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                if i1corid is not None:
                    try:
                        i1cor = np.average(np.array(f[str(scanid).split('.')[0]+'.2/measurement/'+i1corid][:]))
                    except KeyError:
                        try:
                            i1cor = np.average(np.array(f[str(scanid)+'/instrument/'+i1corid+'/data'][:]))
                        except KeyError:
                            print("***ERROR: no viable i1cor value obtained. Set to 1.")
                            i1cor = 1.
                    i1 = i1/np.average(i1) * i1cor
            try:
                mot1 = np.array(f[sc_id+'/measurement/'+mot1_name][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                mot2 = np.array(f[sc_id+'/measurement/'+mot2_name][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
            except KeyError:
                mot1 = np.array(f[sc_id+'/instrument/positioners/'+mot1_name][()]).reshape(sc_dim)
                mot2 = np.array(f[sc_id+'/instrument/positioners/'+mot2_name][()]).reshape(sc_dim)
            tm = np.array(f[sc_id+'/measurement/'+ch0id+'_elapsed_time'][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
            f.close()

            # find direction in which mot1 and mot2 increase  #TODO: this probably fails in case of line scans...
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
            f = h5py.File(file, 'r')
            scan_cmd += ' '+(f[sc_id+'/title'][()].decode('utf8')+' '+mot1_name+' '+str(sc_dim[0])+' '+mot2_name+' '+str(sc_dim[1]))
            #the other arrays we can't simply append: have to figure out which side to stitch them to, and if there is overlap between motor positions
            spectra0_tmp = np.array(f[sc_id+'/measurement/'+ch0id])
            spectra0_temp = np.zeros((sc_dim[0], sc_dim[1], spectra0_tmp.shape[1]))
            for i in range(0, spectra0_tmp.shape[1]):
                spectra0_temp[:,:,i] = spectra0_tmp[:sc_dim[0]*sc_dim[1],i].reshape(sc_dim)
            icr0_temp = np.array(f[sc_id+'/measurement/'+ch0id+'_'+icrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
            ocr0_temp = np.array(f[sc_id+'/measurement/'+ch0id+'_'+ocrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
            if ch2id is not None:
                spectra2_tmp = np.array(f[sc_id+'/measurement/'+ch2id])
                spectra2_temp = np.zeros((sc_dim[0], sc_dim[1], spectra2_temp.shape[1]))
                for i in range(0, spectra2_tmp.shape[1]):
                    spectra2_temp[:,:,i] = spectra2_tmp[:sc_dim[0]*sc_dim[1],i].reshape(sc_dim)
                icr2_temp = np.array(f[sc_id+'/measurement/'+ch2id+'_'+icrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                ocr2_temp = np.array(f[sc_id+'/measurement/'+ch2id+'_'+ocrid][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
            i0_temp = np.array(f[sc_id+'/measurement/'+i0id][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
            if i0corid is not None:
                try:
                    i0cor = np.average(np.array(f[str(scanid).split('.')[0]+'.2/measurement/'+i0corid][:]))
                except KeyError:
                    try:
                        i0cor = np.average(np.array(f[str(scanid)+'/instrument/'+i0corid+'/data'][:]))
                    except KeyError:
                        print("***ERROR: no viable i0cor value obtained. Set to 1.")
                        i0cor = 1.
                i0_temp = i0_temp/np.average(i0_temp) * i0cor
            if i1id is not None:
                i1_temp = np.array(f[sc_id+'/measurement/'+i1id][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                if i1corid is not None:
                    try:
                        i1cor = np.average(np.array(f[str(scanid).split('.')[0]+'.2/measurement/'+i1corid][:]))
                    except KeyError:
                        try:
                            i1cor = np.average(np.array(f[str(scanid)+'/instrument/'+i1corid+'/data'][:]))
                        except KeyError:
                            print("***ERROR: no viable i1cor value obtained. Set to 1.")
                            i1cor = 1.
                    i1_temp = i1_temp/np.average(i1_temp) * i1cor
            try:
                mot1_temp = np.array(f[sc_id+'/measurement/'+mot1_name][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
                mot2_temp = np.array(f[sc_id+'/measurement/'+mot2_name][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
            except KeyError:
                mot1_temp = np.array(f[sc_id+'/instrument/positioners/'+mot1_name][()]).reshape(sc_dim)
                mot2_temp = np.array(f[sc_id+'/instrument/positioners/'+mot2_name][()]).reshape(sc_dim)
            tm_temp = np.array(f[sc_id+'/measurement/'+ch0id+'_elapsed_time'][:sc_dim[0]*sc_dim[1]]).reshape(sc_dim)
            f.close()

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
                    print('Here1',end=' ')
            elif mot1_id == 0 and not np.allclose(mot1[(mot1[:,0].shape[0]-mot1lim):,0], mot1_temp[:mot1lim,0], atol=atol):
                    mot_flag = 1
                    print('Here2',end=' ')
            elif mot2_id == 1 and not np.allclose(mot2[0,(mot2[0,:].shape[0]-mot2lim):], mot2_temp[0,:mot2lim], atol=atol):
                    mot_flag = 2
                    print('Here3',end=' ')
            elif mot2_id == 0 and not np.allclose(mot2[(mot2[:,0].shape[0]-mot2lim):,0], mot2_temp[:mot2lim,0], atol=atol):
                    mot_flag = 2
                    print('Here4')
                    print(mot2[(mot2[:,0].shape[0]-mot2_temp[:,0].shape[0]):,0])
                    print(mot2_temp[:,0])
            
            # check if several regions have identical mot1 or mot2 positions
            if mot_flag == 2:
                # as mot1 and mot1_temp are identical, it must be that mot2 changes
                if mot2.max() < mot2_temp.min():
                    spectra0 = np.concatenate((spectra0, spectra0_temp),axis=mot1_id)
                    icr0 = np.concatenate((icr0, icr0_temp), axis=mot1_id)
                    ocr0 = np.concatenate((ocr0, ocr0_temp), axis=mot1_id)
                    if ch2id is not None:
                        spectra2 = np.concatenate((spectra2, spectra2_temp),axis=mot1_id)
                        icr2 = np.concatenate((icr2, icr2_temp), axis=mot1_id)
                        ocr2 = np.concatenate((ocr2, ocr2_temp), axis=mot1_id)
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
                    if ch2id is not None:
                        spectra2 = np.concatenate((spectra2_temp, spectra2),axis=mot1_id)
                        icr2 = np.concatenate((icr2_temp, icr2), axis=mot1_id)
                        ocr2 = np.concatenate((ocr2_temp, ocr2), axis=mot1_id)
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
                            keep_id = np.array(np.where(mot2[:,0] < mot2_temp.min())).max()+1 #add one as we also need last element of id's
                            spectra0 = np.concatenate((spectra0[0:keep_id,:,:], spectra0_temp),axis=mot1_id)
                            icr0 = np.concatenate((icr0[0:keep_id,:], icr0_temp), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0[0:keep_id,:], ocr0_temp), axis=mot1_id)
                            if ch2id is not None:
                                spectra2 = np.concatenate((spectra2[0:keep_id,:,:], spectra2_temp),axis=mot1_id)
                                icr2 = np.concatenate((icr2[0:keep_id,:], icr2_temp), axis=mot1_id)
                                ocr2 = np.concatenate((ocr2[0:keep_id,:], ocr2_temp), axis=mot1_id)
                            i0 = np.concatenate((i0[0:keep_id,:], i0_temp), axis=mot1_id)
                            if i1id is not None:
                                i1 = np.concatenate((i1[0:keep_id,:], i1_temp), axis=mot1_id)
                            mot1 = np.concatenate((mot1[0:keep_id,:], mot1_temp), axis=mot1_id)
                            mot2 = np.concatenate((mot2[0:keep_id,:], mot2_temp), axis=mot1_id)
                            tm = np.concatenate((tm[0:keep_id,:], tm_temp), axis=mot1_id)
                        else:
                            keep_id = np.array(np.where(mot2[0,:] < mot2_temp.min())).max()+1 #add one as we also need last element of id's
                            spectra0 = np.concatenate((spectra0[:,0:keep_id,:], spectra0_temp),axis=mot1_id)
                            icr0 = np.concatenate((icr0[:,0:keep_id], icr0_temp), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0[:,0:keep_id], ocr0_temp), axis=mot1_id)
                            if ch2id is not None:
                                spectra2 = np.concatenate((spectra2[:,0:keep_id,:,:], spectra2_temp),axis=mot1_id)
                                icr2 = np.concatenate((icr2[:,0:keep_id], icr2_temp), axis=mot1_id)
                                ocr2 = np.concatenate((ocr2[:,0:keep_id], ocr2_temp), axis=mot1_id)
                            i0 = np.concatenate((i0[:,0:keep_id], i0_temp), axis=mot1_id)
                            if i1id is not None:
                                i1 = np.concatenate((i1[:,0:keep_id], i1_temp), axis=mot1_id)
                            mot1 = np.concatenate((mot1[:,0:keep_id], mot1_temp), axis=mot1_id)
                            mot2 = np.concatenate((mot2[:,0:keep_id], mot2_temp), axis=mot1_id)
                            tm = np.concatenate((tm[:,0:keep_id], tm_temp), axis=mot1_id)
                    else:
                        # first mot2_temp, followed by remainder of mot2 (where no more overlap)
                        keep_id = np.array(np.where(mot2_temp[mot2_id] < mot2.min())).max()+1 #add one as we also need last element of id's
                        if mot2_id == 0:
                            spectra0 = np.concatenate((spectra0_temp[:,0:keep_id,:], spectra0),axis=mot1_id)
                            icr0 = np.concatenate((icr0_temp[:,0:keep_id], icr0), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0_temp[:,0:keep_id], ocr0), axis=mot1_id)
                            if ch2id is not None:
                                spectra2 = np.concatenate((spectra2_temp[:,0:keep_id,:,:], spectra2),axis=mot1_id)
                                icr2 = np.concatenate((icr2_temp[:,0:keep_id], icr2), axis=mot1_id)
                                ocr2 = np.concatenate((ocr2_temp[:,0:keep_id], ocr2), axis=mot1_id)
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
                            if ch2id is not None:
                                spectra2 = np.concatenate((spectra2_temp[:,0:keep_id,:,:], spectra2),axis=mot1_id)
                                icr2 = np.concatenate((icr2_temp[:,0:keep_id], icr2), axis=mot1_id)
                                ocr2 = np.concatenate((ocr2_temp[:,0:keep_id], ocr2), axis=mot1_id)
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
                    if ch2id is not None:
                        spectra2 = np.concatenate((spectra2, spectra2_temp),axis=mot1_id)
                        icr2 = np.concatenate((icr2, icr2_temp), axis=mot1_id)
                        ocr2 = np.concatenate((ocr2, ocr2_temp), axis=mot1_id)
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
                    if ch2id is not None:
                        spectra2 = np.concatenate((spectra2_temp, spectra2),axis=mot1_id)
                        icr2 = np.concatenate((icr2_temp, icr2), axis=mot1_id)
                        ocr2 = np.concatenate((ocr2_temp, ocr2), axis=mot1_id)
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
                            keep_id = np.array(np.where(mot1[:,0] < mot1_temp.min())).max()+1 #add one as we also need last element of id's
                            spectra0 = np.concatenate((spectra0[0:keep_id,:,:], spectra0_temp),axis=mot1_id)
                            icr0 = np.concatenate((icr0[0:keep_id,:], icr0_temp), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0[0:keep_id,:], ocr0_temp), axis=mot1_id)
                            if ch2id is not None:
                                spectra2 = np.concatenate((spectra2[0:keep_id,:,:], spectra2_temp),axis=mot1_id)
                                icr2 = np.concatenate((icr2[0:keep_id,:], icr2_temp), axis=mot1_id)
                                ocr2 = np.concatenate((ocr2[0:keep_id,:], ocr2_temp), axis=mot1_id)
                            i0 = np.concatenate((i0[0:keep_id,:], i0_temp), axis=mot1_id)
                            if i1id is not None:
                                i1 = np.concatenate((i1[0:keep_id,:], i1_temp), axis=mot1_id)
                            mot1 = np.concatenate((mot1[0:keep_id,:], mot1_temp), axis=mot1_id)
                            mot2 = np.concatenate((mot2[0:keep_id,:], mot2_temp), axis=mot1_id)
                            tm = np.concatenate((tm[0:keep_id,:], tm_temp), axis=mot1_id)
                        else:
                            keep_id = np.array(np.where(mot1[0,:] < mot1_temp.min())).max()+1 #add one as we also need last element of id's
                            spectra0 = np.concatenate((spectra0[:,0:keep_id,:], spectra0_temp),axis=mot1_id)
                            icr0 = np.concatenate((icr0[:,0:keep_id], icr0_temp), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0[:,0:keep_id], ocr0_temp), axis=mot1_id)
                            if ch2id is not None:
                                spectra2 = np.concatenate((spectra2[:,0:keep_id,:,:], spectra2_temp),axis=mot1_id)
                                icr2 = np.concatenate((icr2[:,0:keep_id], icr2_temp), axis=mot1_id)
                                ocr2 = np.concatenate((ocr2[:,0:keep_id], ocr2_temp), axis=mot1_id)
                            i0 = np.concatenate((i0[:,0:keep_id], i0_temp), axis=mot1_id)
                            if i1id is not None:
                                i1 = np.concatenate((i1[:,0:keep_id], i1_temp), axis=mot1_id)
                            mot1 = np.concatenate((mot1[:,0:keep_id], mot1_temp), axis=mot1_id)
                            mot2 = np.concatenate((mot2[:,0:keep_id], mot2_temp), axis=mot1_id)
                            tm = np.concatenate((tm[:,0:keep_id], tm_temp), axis=mot1_id)
                    else:
                        # first mot1_temp, followed by remainder of mot1 (where no more overlap)
                        if mot1_id == 0:
                            keep_id = np.array(np.where(mot1_temp[:,0] < mot1.min())).max()+1 #add one as we also need last element of id's
                            spectra0 = np.concatenate((spectra0_temp[0:keep_id,:,:], spectra0),axis=mot1_id)
                            icr0 = np.concatenate((icr0_temp[0:keep_id,:], icr0), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0_temp[0:keep_id,:], ocr0), axis=mot1_id)
                            if ch2id is not None:
                                spectra2 = np.concatenate((spectra2_temp[0:keep_id,:,:], spectra2),axis=mot1_id)
                                icr2 = np.concatenate((icr2_temp[0:keep_id,:], icr2), axis=mot1_id)
                                ocr2 = np.concatenate((ocr2_temp[0:keep_id,:], ocr2), axis=mot1_id)
                            i0 = np.concatenate((i0_temp[0:keep_id,:], i0), axis=mot1_id)
                            if i1id is not None:
                                i1 = np.concatenate((i1_temp[0:keep_id,:], i1), axis=mot1_id)
                            mot1 = np.concatenate((mot1_temp[0:keep_id,:], mot1), axis=mot1_id)
                            mot2 = np.concatenate((mot2_temp[0:keep_id,:], mot2), axis=mot1_id)
                            tm = np.concatenate((tm_temp[0:keep_id,:], tm), axis=mot1_id)
                        else:
                            keep_id = np.array(np.where(mot1_temp[0,:] < mot1.min())).max()+1 #add one as we also need last element of id's
                            spectra0 = np.concatenate((spectra0_temp[:,0:keep_id,:], spectra0),axis=mot1_id)
                            icr0 = np.concatenate((icr0_temp[:,0:keep_id], icr0), axis=mot1_id)
                            ocr0 = np.concatenate((ocr0_temp[:,0:keep_id], ocr0), axis=mot1_id)
                            if ch2id is not None:
                                spectra2 = np.concatenate((spectra2_temp[:,0:keep_id,:,:], spectra2),axis=mot1_id)
                                icr2 = np.concatenate((icr2_temp[:,0:keep_id], icr2), axis=mot1_id)
                                ocr2 = np.concatenate((ocr2_temp[:,0:keep_id], ocr2), axis=mot1_id)
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
            if ch2id is not None:
            	spectra2[i,:,:] = spectra2[i,sort_id,:]
            	icr2[i,:] = icr2[i,sort_id]
            	ocr2[i,:] = ocr2[i,sort_id]
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
            if ch2id is not None:
                spectra2[:,i,:] = spectra2[sort_id,i,:]
                icr2[:,i] = icr2[sort_id,i]
                ocr2[:,i] = ocr2[sort_id,i]
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
    if ch2id is not None:
        sumspec2 = np.sum(spectra2[:], axis=(0,1))
        maxspec2 = np.zeros(sumspec2.shape[0])
        for i in range(sumspec2.shape[0]):
            maxspec2[i] = spectra2[:,:,i].max()
    
    # write h5 file in our structure
    filename = h5id15[0].split(".")[0]+scan_suffix+'.h5' #scanid is of type 1.1,  2.1,  4.1
    f = h5py.File(filename, 'w')
    f.create_dataset('cmd', data=' '.join(scan_cmd))
    f.create_dataset('raw/channel00/spectra', data=spectra0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/icr', data=icr0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/ocr', data=ocr0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/sumspec', data=sumspec0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/maxspec', data=maxspec0, compression='gzip', compression_opts=4)
    if ch2id is not None:
        f.create_dataset('raw/channel02/spectra', data=spectra2, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel02/icr', data=icr2, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel02/ocr', data=ocr2, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel02/sumspec', data=sumspec2, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel02/maxspec', data=maxspec2, compression='gzip', compression_opts=4)
    f.create_dataset('raw/I0', data=i0, compression='gzip', compression_opts=4)
    if i1id is not None:
        f.create_dataset('raw/I1', data=i1, compression='gzip', compression_opts=4)
    dset = f.create_dataset('mot1', data=mot1, compression='gzip', compression_opts=4)
    dset.attrs['Name'] = mot1_name
    dset = f.create_dataset('mot2', data=mot2, compression='gzip', compression_opts=4)
    dset.attrs['Name'] = mot2_name
    f.create_dataset('raw/acquisition_time', data=tm, compression='gzip', compression_opts=4)
    f.close()
    print("Done")


##############################################################################
# if __name__ == '__main__':
#     # MergeP06Nxs("scan_00183")
#     # fit_xrf_batch('scan_00183_merge.h5', 'mod_gu19.cfg', standard=None)
#     # norm_xrf_batch('scan_00183_merge.h5', I0norm=200000)
#     # hdf_overview_images('scan_00183_merge.h5', 4, 0.5, 25) # h5file, ncols, pix_size[µm], scl_size[µm]
    
#     # MergeP06Nxs("scan_00024")
#     # fit_xrf_batch('scan_00024_merge.h5', 'athog.cfg', standard='atho_g.cnc')
#     # norm_xrf_batch('scan_00024_merge.h5', I0norm=200000)
#     # calc_detlim('scan_00024_merge.h5', 'atho_g.cnc')

    # fit_xrf_batch('D:\School\PhD\python_pro\CM2_45mm_hotspot_0001_scan1.h5', 'D:\School\PhD\python_pro\cm2.cfg', standard='some', ncores=-1, verbose=True)
#     None    





