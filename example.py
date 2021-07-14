# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:55:01 2021

@author: prrta
"""

import sys
sys.path.insert(1, 'D:/School/PhD/python_pro/plotims')
sys.path.insert(1, 'D:/School/PhD/python_pro/xrf_proc')
import xrf_fit as Fit
import plotims as Ims


def prepare_p06():
    Fit.MergeP06Nxs('dir/ref/scan_00001')
    Fit.MergeP06Nxs(['dir/ct/scan_00002','dir/ct/scan_00003'])


def prepare_id15():
    Fit.h5id15convert('dir/srmscan.h5', '1.1', (10, 10), mot1_name='hrz', mot2_name='hry')
    Fit.h5id15convert('dir/ctscan.h5', '1.1', (100, 101), mot1_name='hrrz', mot2_name='hry')
#%%
def fast_process():
    # fit the spectra usin linear fast fit (cfg file must be SNIP background!)
    Fit.fit_xrf_batch('dir/preppedfile.h5', 'dir/pymca_config.cfg', standard=None, ncores=10)
    # normalise the data for detector deadtime, primary beam flux, ...
    Fit.norm_xrf_batch('dir/preppedfile.h5', I0norm=1E6)
    # create images; linear and log scaled
    Fit.hdf_overview_images('dir/preppedfile.h5', 'norm', 4, 15, 250) # h5file, ncols, pix_size[µm], scl_size[µm]
    Fit.hdf_overview_images('dir/preppedfile.h5', 'norm', 4, 15, 250, log=True) # h5file, ncols, pix_size[µm], scl_size[µm]
#%%    
def precise_process():
    # Fit reference material
    #       set standard to directory of appropriate cnc file
    Fit.fit_xrf_batch('dir/srmfile.h5', ['dir/srm_config_ch0.cfg','dir/srm_config_ch2.cfg'], standard='dir/nist613.cnc', ncores=10)
    Fit.norm_xrf_batch('dir/srmfile.h5', I0norm=1E6)
    # Calculate detection limits and elemental yields
    Fit.calc_detlim('dir/srmfile.h5', 'dir/nist613.cnc')


    # fit the spectra using precise (yet slow!) point-by-point fit #time estimate: with 12 cores approx 0.4s/spectrum
    #       can provide separate config files for each detector channel (0 or 2) as a list, in that order. Otherwise all channels are fit with same config.
    Fit.fit_xrf_batch('dir/preppedfile.h5', ['dir/pymca_config_ch0.cfg','dir/pymca_config_ch2.cfg'], standard='some', ncores=10)
    # normalise the data for detector deadtime, primary beam flux, ...
    #       If snake mesh set True to interpolate grid positions
    Fit.norm_xrf_batch('dir/preppedfile.h5', I0norm=1E6, snake=True)
    # create images; linear and log scaled
    Fit.hdf_overview_images('dir/preppedfile.h5', 'norm', 4, 15, 250) # h5file, ncols, pix_size[µm], scl_size[µm]
    Fit.hdf_overview_images('dir/preppedfile.h5', 'norm', 4, 15, 250, log=True) # h5file, ncols, pix_size[µm], scl_size[µm]
    # quantify the normalised data using the srmfile.h5 data. 
    #       Normalise for 'Rayl' signal, calculate self-absorption correction terms for Fe based on Fe-Ka/Kb ratio in 'CI_chondrite.cnc' matrix
    Fit.quant_with_ref('dir/preppedfile.h5', 'dir/srmfile.h5', channel='channel02', norm='Rayl', absorb=(['Fe'], 'CI_chondrite.cnc'), snake=True) 
#%%
def tomo_reconstruction():
    import tomo_proc
    
    # calculate tomographic reconstruction for 'channel02' signal of 'norm' data
    #       estimate centre of rotation value based from 'Sr-K' signal
    #       display final reconstruction images in collimated overview style with 8 colums
    tomo_proc.h5_tomo_proc('dir/preppedfile.h5', rot_mot='mot2', channel='channel02', signal='Sr-K', rot_centre=None, ncol=8)
#%%
def correlation_plots():
    import h5py
    
    f = h5py.File('dir/preppedfile.h5','r')
    data = np.moveaxis(np.array(f['tomo/channel02/slices']),0,-1)
    data[np.isnan(data)] = 0.
    names = [n.decode('utf8') for n in f['tomo/channel02/names']]
    f.close()
    print([names[i] for i in [1,2,3,4,5,6,8,9,14]])
    Ims.plot_correl(data, names, el_id=[1,2,3,4,5,6,8,9,14], save='dir/preppedfile_correlation.png')

        