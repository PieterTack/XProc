# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:32:27 2023

@author: prrta
"""

import numpy as np

##############################################################################
# Convert XProcH5 intensity data to csv format
#   Column headers are the respective element names, whereas rows represent the different motor1 coordinates
#       (in handheld XRF data these are the separate file names)
#TODO: we may want to expand this function to allow for multiple h5files and h5dirs combined to a single csv file. At that point,
#   we should include an additional column stating the respective h5file name, and, more importantly, make sure all elements are represented across all files
def XProcH5toCSV(h5file, h5dir, csvfile, overwrite=False):
    import h5py
    import os

    # check if csvfile already exists, and warn if we don't want to overwrite (default)
    if overwrite is False:
        if os.path.isfile(csvfile) is True:
            raise ValueError("Warning: CSV file "+csvfile+" already exists.\n Set keyword overwrite=True if you wish to overwrite this file.")

    # determine the path containing the element names
    namespath = h5dir.split('/')
    if namespath[-2] == 'sum':
        namespath = '/'.join(namespath[:-2])+'/names' #if sum directory the names path is in the directory above
    else:
        namespath = '/'.join(namespath[:-1])+'/names'
        
    # read the h5 data
    with h5py.File(h5file, 'r') as h5:
        data = np.asarray(h5[h5dir]).astype(str)
        names = [n.decode("utf8") for n in h5[namespath]]
        mot1_name = str(h5['mot1'].attrs["Name"])
        if mot1_name == "hxrf":
            rowID = [n.decode("utf8") for n in h5["mot1"]]
        else:
            rowID = np.asarray(h5["mot1"]).astype(str)+'_'+np.asarray(h5["mot2"]).astype(str)

    # 'flatten' the data
    if len(data.shape) == 3:
        data = data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
        rowID = rowID.reshape((rowID.shape[0]*rowID.shape[1]))

    # write data as csv
    print("Writing "+csvfile+"...", end="")
    with open(csvfile, 'w') as csv:
        csv.write('RowID'+';'.join(names))
        for i in range(rowID.shape[0]):
            csv.write(rowID[i]+';'.join(data[:,i]))
    print("Done.")
        
    
##############################################################################
# join two files together, stitched one after the other
def XProcH5_combine(files=['A0186_1_3_xrfct_0003_scan1.h5', 'A0186_1_3_xrfct_0004_scan1.h5', 'A0186_1_3_xrfct_0005_scan1.h5'], newfile='A0186_1_3_xrfct_scan1.h5', ax=0):
    import h5py
    import numpy as np
    
    cmd = ''
    mot1 = []
    mot2 = []
    i0 = []
    i1 = []
    tm = []
    icr0 = []
    ocr0 = []
    spectra0 = []
    icr1 = []
    ocr1 = []
    spectra1 = []
    
    for file in files:
        print("Reading "+file+"...", end="")
        f = h5py.File(file,'r')
        try:
            cmd += f['cmd'][()].decode('utf8')
        except Exception as ex:
            cmd += f['cmd'][()]

        mot1.append(np.array(f['mot1']))
        mot1_name = str(f['mot1'].attrs["Name"])
        mot2.append(np.array(f['mot2']))
        mot2_name = str(f['mot2'].attrs["Name"])
        i0.append(np.array(f['raw/I0']))
        try:
            i1.append(np.array(f['raw/I1']))
            i1flag = True
        except KeyError:
            i1flag = False
        tm.append(np.array(f['raw/acquisition_time']))
        icr0.append(np.array(f['raw/channel00/icr']))
        ocr0.append(np.array(f['raw/channel00/ocr']))
        spectra0.append(np.array(f['raw/channel00/spectra']))
        try:
            icr1.append(np.array(f['raw/channel01/icr']))
            ocr1.append(np.array(f['raw/channel01/ocr']))
            spectra1.append(np.array(f['raw/channel01/spectra']))
            ch1flag = True
        except KeyError:
            ch1flag = False
        f.close()

    # add in one array
    
    spectra0 = np.concatenate(spectra0, axis=ax)
    icr0 = np.concatenate(icr0, axis=ax)
    ocr0 = np.concatenate(ocr0, axis=ax)
    if ch1flag:
        spectra1 = np.concatenate(spectra1, axis=ax)
        icr1 = np.concatenate(icr1, axis=ax)
        ocr1 = np.concatenate(ocr1, axis=ax)
    i0 = np.concatenate(i0, axis=ax)
    if i1flag:
        i1 = np.concatenate(i1, axis=ax)
    mot1 = np.concatenate(mot1, axis=ax)
    mot2 = np.concatenate(mot2, axis=ax)
    tm = np.concatenate(tm, axis=ax)

    sumspec0 = np.sum(spectra0[:], axis=(0,1))
    maxspec0 = np.zeros(sumspec0.shape[0])
    for i in range(sumspec0.shape[0]):
        maxspec0[i] = spectra0[:,:,i].max()
    if ch1flag is True:
        sumspec1 = np.sum(spectra1[:], axis=(0,1))
        maxspec1 = np.zeros(sumspec1.shape[0])
        for i in range(sumspec1.shape[0]):
            maxspec1[i] = spectra1[:,:,i].max()
        
    # write the new file
    print("writing "+newfile+"...", end="")
    f = h5py.File(newfile, 'w')
    f.create_dataset('cmd', data=cmd)
    f.create_dataset('raw/channel00/spectra', data=spectra0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/icr', data=icr0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/ocr', data=ocr0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/sumspec', data=sumspec0, compression='gzip', compression_opts=4)
    f.create_dataset('raw/channel00/maxspec', data=maxspec0, compression='gzip', compression_opts=4)
    if ch1flag:
        f.create_dataset('raw/channel01/spectra', data=spectra1, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel01/icr', data=icr1, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel01/ocr', data=ocr1, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel01/sumspec', data=sumspec1, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel01/maxspec', data=maxspec1, compression='gzip', compression_opts=4)
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
        spectra1 = np.array(f['raw/channel01/spectra'])
        chan01_flag = True
        icr1 = np.array(f['raw/channel01/icr'])
        ocr1 = np.array(f['raw/channel01/ocr'])
    except KeyError:
        chan01_flag = False
    try:
        ims0 = np.array(f['fit/channel00/ims'])
        fit_flag = True
        if chan01_flag:
            ims1 = np.array(f['fit/channel01/ims'])
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
    if chan01_flag:
        if fit_flag:
            ims1 = np.delete(ims1, lineid, axis+1)
        spectra1 = np.delete(spectra1, lineid, axis)
        icr1 = np.delete(icr1, lineid, axis)
        ocr1 = np.delete(ocr1, lineid, axis)

    # save the data
    print("Writing truncated data to "+h5file+"...", end=" ")
    if fit_flag:
        del f['fit/channel00/ims']
        f.create_dataset('fit/channel00/ims', data=ims0, compression='gzip', compression_opts=4)
        if chan01_flag:
            del f['fit/channel01/ims']
            f.create_dataset('fit/channel01/ims', data=ims1, compression='gzip', compression_opts=4)
        
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
    if chan01_flag:
        del f['raw/channel01/spectra']
        del f['raw/channel01/icr']
        del f['raw/channel01/ocr']
        f.create_dataset('raw/channel01/spectra', data=spectra1, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel01/icr', data=icr1, compression='gzip', compression_opts=4)
        f.create_dataset('raw/channel01/ocr', data=ocr1, compression='gzip', compression_opts=4)
        
    f.close()
    print('Done')
    
##############################################################################
# add together multiple h5 files of the same dimensions...
#   Make sure it is also ordered similarly etc...
#   Use at your own risk.
def add_h5s(h5files, newfilename):
    import h5py
    
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
                    icr1 = np.array(f['raw/channel01/icr'])
                    ocr1 = np.array(f['raw/channel01/ocr'])
                    spectra1 = np.array(f['raw/channel01/spectra'])
                    maxspec1 = np.array(f['raw/channel01/maxspec'])
                    sumspec1 = np.array(f['raw/channel01/sumspec'])
                    ch1flag = True
                except KeyError:
                    ch1flag = False
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
                if ch1flag:
                    icr1 += np.array(f['raw/channel01/icr'])
                    ocr1 += np.array(f['raw/channel01/ocr'])
                    spectra1 += np.array(f['raw/channel01/spectra'])
                    maxspec_tmp = np.array(f['raw/channel01/maxspec'])
                    for j in range(len(maxspec1)):
                        if maxspec_tmp[j] > maxspec1[j]:
                            maxspec1[j] = maxspec_tmp[j]
                    sumspec1 += np.array(f['raw/channel01/sumspec'])
                    ch1flag = True
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
        if ch1flag:
            f.create_dataset('raw/channel01/spectra', data=spectra1, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel01/icr', data=icr1, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel01/ocr', data=ocr1, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel01/sumspec', data=sumspec1, compression='gzip', compression_opts=4)
            f.create_dataset('raw/channel01/maxspec', data=maxspec1, compression='gzip', compression_opts=4)
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

