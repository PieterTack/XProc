# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 09:15:31 2020

@author: prrta
"""

import tomopy
import h5py
import numpy as np
import plotims
from scipy.interpolate import griddata


def h5_tomo_proc(h5file, rot_mot=None, rot_centre=None, signal='Ba-K', channel='channel00', ncol=8, selfabs=None):
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
    

    #do self-absorption correction if requested
    if selfabs is not None:
        #selfabs should contain the directory to the trained neural network (Gao Bo, 2021 https://dx.doi.org/10.1021/acs.analchem.0c03828)
        proj = Gao_tomo_selfabscorr(selfabs, proj)
    

    # find centre of rotation and perform reconstruction        
    if rot_centre is None:
        rot_center = tomopy.find_center(proj[:,names.index(signal),:].reshape((proj.shape[0],1,proj.shape[2])), angle, ind=0, init=ims.shape[2]/2, tol=0.5, sinogram_order=False)
    else:
        rot_center = rot_centre
    print("Center of rotation: ", rot_center)


    # proj = tomopy.prep.stripe.remove_stripe_sf(proj, size=1)
    # proj = tomopy.prep.stripe.remove_dead_stripe(proj, snr=5, size=20, norm=False)

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
    
    # check if I1 tag is present, to also reconstruct the transmission tomo image
    try:
        h5f['raw/I1']
        h5f.close()
        h5_i1tomo_recon(h5file, rot_centre=rot_center)
    except Exception:
        h5f.close()
     
    # plot data
    colim_opts = plotims.Collated_image_opts()
    colim_opts.ncol = ncol
    colim_opts.nrow = int(np.ceil(len(names)/colim_opts.ncol))
    plotims.plot_colim(data, names, 'viridis', colim_opts=colim_opts, save=h5file.split(".")[0]+'_tomo_overview.png')
    data.data = np.log10(data.data)
    plotims.plot_colim(data, names, 'viridis', colim_opts=colim_opts, save=h5file.split(".")[0]+'_log_tomo_overview.png')


def h5_i1tomo_recon(h5file, rot_centre=None):
    import tomopy
    from scipy.interpolate import griddata
    import tifffile
    
    # try to open i1 directory and reconstruct
    h5f = h5py.File(h5file, 'r+')
    try:
        i1 = np.array(h5f['raw/I1'])
    except Exception:
        i1 = None
        h5f.close()
        return

    if i1 is not None:
        i0 = np.array(h5f['raw/I0'])
        mot1 = np.array(h5f['mot1'])
        mot2 = np.array(h5f['mot2'])
        # norm I1
        normfactor = i0/np.max(i0)
        i1 = i1/normfactor
        i1 = tomopy.remove_neg(tomopy.remove_nan(i1, 0))
        i1 = i1/np.max(i1)

        # Interpolating image for motor position
        pos_low = min(mot1[:,0])
        pos_high = max(mot1[:,0])
        for i in range(0, mot1[:,0].size): #correct for half a pixel shift
            if mot1[i,0] <= np.average((pos_high,pos_low)):
                mot1[i,:] += abs(mot1[i,1]-mot1[i,0])/2.
            else:
                mot1[i,:] -= abs(mot1[i,1]-mot1[i,0])/2.
        mot1_pos = np.average(mot1, axis=0) #mot1[0,:]
        mot2_pos = np.average(mot2, axis=1) #mot2[:,0]
        i1_tmp = np.zeros((i1.shape[0], i1.shape[1]))
        # interpolate to the regular grid motor positions
        mot1_tmp, mot2_tmp = np.mgrid[mot1_pos[0]:mot1_pos[-1]:complex(mot1_pos.size), mot2_pos[0]:mot2_pos[-1]:complex(mot2_pos.size)]
        x = mot1.ravel()
        y = mot2.ravel()
        values = i1.ravel()
        i1_tmp = griddata((x, y), values, (mot1_tmp, mot2_tmp), method='cubic', rescale=True).T
        i1 = i1_tmp
        i1 = tomopy.remove_neg(tomopy.remove_nan(i1, 0))
        
        # i1 = i1[1:-1,:] #remove first and last angular line, as these are empty
        
        angle = mot2_pos*np.pi /180 #motor positions expected in degrees, convert to rad    
        proj = i1.reshape((i1.shape[0], 1, i1.shape[1]))
        proj = tomopy.prep.normalize.minus_log(proj)
        proj = tomopy.remove_neg(tomopy.remove_nan(proj, 0))
        proj[np.isinf(proj)] = 1.0

        if rot_centre is None:
            try:
                rot_center = float(h5f['tomo/'+channel+'/rotation_center'])
            except Exception:
                rot_center = tomopy.find_center(proj, angle, ind=0, init=i1.shape[1]/2, tol=0.5, sinogram_order=False)
        else:
            rot_center = rot_centre
            
        
        # proj = tomopy.prep.normalize.normalize_bg(proj)
        # proj = tomopy.prep.stripe.remove_stripe_sf(proj, size=1)
        # proj = tomopy.prep.stripe.remove_dead_stripe(proj, snr=5, size=20, norm=False)
        # proj = tomopy.prep.stripe.remove_all_stripe(proj, snr=3, la_size=10, sm_size=2)
        recon = tomopy.recon(proj, angle, center=rot_center, algorithm='gridrec', sinogram_order=False, filter_name='shepp')
        recon = tomopy.remove_neg(tomopy.remove_nan(recon, 0))
    
        # Ring removal attempt
        # recon = tomopy.misc.corr.remove_ring(recon, thresh_max=200, thresh=200)
        # recon = tomopy.misc.corr.circ_mask(recon, 0, ratio=0.9, val=0.0, ncore=None)    
    
        try:
            del h5f['tomo/I1/rotation_center']
            del h5f['tomo/I1/ims']
            del h5f['tomo/I1/names']
        except Exception:
            pass
        h5f.create_dataset('tomo/I1/rotation_center', data=rot_center, compression='gzip', compression_opts=4)
        h5f.create_dataset('tomo/I1/ims', data=recon, compression='gzip', compression_opts=4)
        h5f.create_dataset('tomo/I1/names', data=['transmission'.encode('utf8')])
        h5f.close()
        
        # make a plot
        recon = np.flip(tomopy.remove_neg(tomopy.remove_nan(recon[0, :, :], 0)), 0)
        Ims.plot_image(recon, 'transmission', 'gray', plt_opts=None, sb_opts=None, cb_opts=None, clim=None, save=h5file.split('.')[0]+'_i1tomo.png', subplot=None)




def Gao_tomo_selfabscorr(neuraldir, data):
    # Here are the functions pertaining to Bo Gao's self-absorption algorithm https://dx.doi.org/10.1021/acs.analchem.0c03828
    #   Please cite this research when you use this function
    #   Neuraldir is the path directory to the neural network h5 file
    #   Data is a 3D numpy array of size M*N*O with N the amount of elements, M the angular axis and O the translational axis 
    import tensorflow as tf
    import tensorflow.keras.backend as K

    """
    Load in the trained Neural network
    
    Change the directory if necessary
    """
    filename = neuraldir
    model = tf.keras.models.load_model(filename, custom_objects = {'compute_loss':keras_customized_loss()})
    
    """
    Preprocess the fluorescence sinogram:
    1. Flip the fluorescence sinogram so its left side has stronger self-absorption effect
    2. Normalize the fluorescence sinogram
    3. Resize the fluorescence sinogram to [128, 256]
    4. Resize the predicted/corrected sinogram back to the original dimensions
    """
    data = np.array(data) # np.fliplr(data)
    data[np.isnan(data)] = 0.
    data[data<0] = 0.
    for k in range(data.shape[1]):
        sino = data[:,k,:] / np.max(data[:,k,:])
        a, b = sino.shape
        sino = np.reshape(sino, [1, a, b, 1])
        sino = tf.image.resize(sino, [128, 256])   # the size [128,256] depends on your neural network training! Don't just change if you don't know what you're doing!
        pred = model.predict(sino)
        data[:,k,:] = np.array(tf.image.resize(pred, [a, b])).reshape((a,b))
       
    return data


"""
Define the loss function for neural network training --> equation (1) in the manuscript
"""
# Combine of l2 norm
def keras_customized_loss(lambda1 = 1.0, lambda2 = 0.05):
    def grad_x(image):
        return K.abs(image[:, 1:] - image[:, :-1])

    def grad_y(image):
        return K.abs(image[:, :, 1:] - image[:, :, :-1])

    def compute_loss(y_true, y_pred):
        pred_grad_x = grad_x(y_pred)
        pred_grad_y = grad_y(y_pred)
        true_grad_x = grad_x(y_true)
        true_grad_y = grad_y(y_true)
        # Based on my current understanding, axis=-1 is not necessary here, the reason
        # why it is presented in Keras code is because sometimes the losses need to have 
        # the same size as y_true
        loss1 = K.mean(K.square(y_pred-y_true)) 
        loss2 = K.mean(K.square(pred_grad_x-true_grad_x))
        loss3 = K.mean(K.square(pred_grad_y-true_grad_y))
        
        return (lambda1*loss1+lambda2*loss2+lambda2*loss3)

    return compute_loss

