# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 09:15:31 2020

@author: prrta
"""

import tomopy
import h5py
import numpy as np
import plotims


def h5_tomo_proc(h5file, rot_mot=None, rot_centre=None, signal='Ba-K', channel='channel00', ncol=8):
    if rot_mot is None:
        mot1 = 'mot1'
    else:
        mot1 = rot_mot
    
    h5f = h5py.File(h5file, 'r+')
    ims = np.array(h5f['norm/'+channel+'/ims'])
    # ims = ims[:,25:,:] #only do this if omitting certain angles from the scan...
    names = ["-".join(name.decode('utf8').split(" ")) for name in h5f['norm/'+channel+'/names']]
    angle = np.array(h5f[mot1])[:,0]*np.pi /180 #motor positions expected in degrees, convert to rad
    
    proj = np.zeros((ims.shape[1],ims.shape[0],ims.shape[2]))
    # remove negative and NaN values, remove stripe artefacts, ...
    for k in range(0, ims.shape[0]):
        proj[:,k,:] = tomopy.remove_neg(tomopy.remove_nan(ims[k, :, :], 0)) #also replace all nan values and negative values with 0
    # proj = tomopy.prep.stripe.remove_stripe_sf(proj, size=1)
    # proj = tomopy.prep.stripe.remove_dead_stripe(proj, snr=5, size=20, norm=False)
    
    
    # find centre of rotation and perform reconstruction        
    if rot_centre is None:
        rot_center = tomopy.find_center(proj[:,names.index(signal),:].reshape((proj.shape[0],1,proj.shape[2])), angle, ind=0, init=ims.shape[2]/2, tol=0.5, sinogram_order=False)
    else:
        rot_center = rot_centre
    print("Center of rotation: ", rot_center)


    # extra_options = {'MinConstraint': 0}
    # options = {
    #     'proj_type': 'cuda',
    #     'method': 'SIRT_CUDA',
    #     'num_iter': 200,
    #     'extra_options': extra_options
    # }        
    recon = tomopy.recon(proj, angle, center=rot_center, algorithm='gridrec', sinogram_order=False, filter_name='shepp') #tomopy.astra, options=options)#
    # Algorithms: 'gridrec', 'mlem'
    # filter_name: 'shepp' (default), 'parzen'
    
    # Ring removal attempt
    # recon = tomopy.misc.corr.remove_ring(recon)
    
    # prepare data for imaging in plotims
    data = plotims.ims()
    data.data = np.zeros((recon.shape[1], recon.shape[2], recon.shape[0]))
    for k in range(0, recon.shape[0]):
        print
        data.data[:,:,k] = tomopy.remove_neg(tomopy.remove_nan(np.flip(recon[k, :, :], 0)))
    data.names = names
    
    # save tomo data in h5 file
    try:
        del h5f['tomo/'+channel+'/rotation_center']
        del h5f['tomo/'+channel+'/ims']
        del h5f['tomo/'+channel+'/names']
    except Exception:
        pass
    h5f.create_dataset('tomo/'+channel+'/rotation_center', data=rot_center, compression='gzip', compression_opts=4)
    h5f.create_dataset('tomo/'+channel+'/ims', data=recon, compression='gzip', compression_opts=4)
    h5f.create_dataset('tomo/'+channel+'/names', data=h5f['norm/'+channel+'/names'])
    h5f.close()
 
    
    # plot data
    colim_opts = plotims.Collated_image_opts()
    colim_opts.ncol = ncol
    colim_opts.nrow = int(np.ceil(len(names)/colim_opts.ncol))
    plotims.plot_colim(data, names, 'viridis', colim_opts=colim_opts, save=h5file.split(".")[0]+'_tomo_overview.png')
    data.data = np.log10(data.data)
    plotims.plot_colim(data, names, 'viridis', colim_opts=colim_opts, save=h5file.split(".")[0]+'_log_tomo_overview.png')
