# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:32:27 2023

@author: prrta
"""

import numpy as np

def getZ(element):
    table = [  
        'H',                                                                                                                                            'He',
        'Li', 'Be',                                                                                                       'B',  'C',  'N',  'O',  'F',  'Ne',
        'Na', 'Mg',                                                                                                       'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
        'K',  'Ca', 'Sc',                                           'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
        'Rb', 'Sr', 'Y',                                            'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',  'Xe',
        'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
        'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    ]
    return table.index(element)+1

##############################################################################
#TODO: we may want to expand this function to allow for multiple h5files and h5dirs combined to a single csv file. At that point,
#   we should include an additional column stating the respective h5file name, and, more importantly, make sure all elements are represented across all files
def XProcH5toCSV(h5file, h5dir, csvfile, overwrite=False):
    """
    Convert XProcH5 intensity data to csv format (column separation by ;)
    Column headers are the respective element names, whereas rows represent the different motor1 coordinates
        (in handheld XRF data these are the separate file names)

    Parameters
    ----------
    h5file : string or list of strings
        File directory path(s) to the H5 file(s) containing the data.
    h5dir : string
        Data directory within the H5 file containing the data to be converted, e.g. "/norm/channel00/ims". 
        A 'names' directory should be present in the same parent folder, or the grandeparent folder in case of sumspectra results. 
    csvfile : string
        Output file path name.
    overwrite : Boolean, optional
        If True, allows to overwrite the csvfile. The default is False, preventing overwriting CSV files.

    Raises
    ------
    ValueError
        Returned when the supplied CSVfile already exists and overwrite is False.

    Returns
    -------
    None.

    """
    import h5py
    import os

    # check if csvfile already exists, and warn if we don't want to overwrite (default)
    if overwrite is False:
        if os.path.isfile(csvfile) is True:
            raise ValueError("Warning: CSV file "+csvfile+" already exists.\n Set keyword overwrite=True if you wish to overwrite this file.")

    # determine the patush containing the element names
    namespath = h5dir.split('/')
    if namespath[-2] == 'sum':
        namespath = '/'.join(namespath[:-2])+'/names' #if sum directory the names path is in the directory above
        sumdata = True
    else:
        namespath = '/'.join(namespath[:-1])+'/names'
        sumdata = False

    # Check if h5file is list of files or single string, and act accordingly
    if type(h5file) is type(list()):
        # go through all h5files and make array containing all unique element identifiers.
        allnames = []
        nrows = 0
        for file in h5file:
            with h5py.File(file, 'r') as h5:
                for n in h5[namespath]: allnames.append(n.decode("utf8"))
                if sumdata is False:
                    nrows += np.asarray([n.decode("utf8") for n in h5["mot1"]]).size
                else:
                    nrows += 1
        unique_names = [name for name in np.unique(allnames)]
        # Now we know the data dimensions to expect
        data = np.zeros((len(unique_names),nrows))
        rowID = []
        fileID = []
        # go through all h5 files again, and sort the data in the appropriate data column
        for file in h5file:
            with h5py.File(file, 'r') as h5:
                temp = np.asarray(h5[h5dir])
                names = [n.decode("utf8") for n in h5[namespath]]
                if sumdata is False:
                    mot1_name = str(h5['mot1'].attrs["Name"])
                    if mot1_name == "hxrf":
                        rows = [n.decode("utf8") for n in h5["mot1"]]
                    else:
                        rows = np.asarray(h5["mot1"]).astype(str)+'_'+np.asarray(h5["mot2"]).astype(str)
                    # 'flatten' the data
                    if len(temp.shape) == 3:
                        temp = temp.reshape((temp.shape[0],temp.shape[1]*temp.shape[2]))
                        rows = rows.reshape((rows.shape[0]*rows.shape[1]))
                    for j,n in enumerate(rows):
                        rowID.append(n)
                        fileID.append(file)
                        for i,x in enumerate(names):
                            dataid = unique_names.index(x)
                            data[dataid,len(rowID)-1] = temp[i,j]
                else:
                    rowID.append(h5dir)
                    fileID.append(file)
                    for i,x in enumerate(names):
                        dataid = unique_names.index(x)
                        data[dataid,len(rowID)-1] = temp[i]
        data = np.asarray(data).astype(str)
    else: #h5file is a single string (or should be)   
        # read the h5 data
        with h5py.File(h5file, 'r') as h5:
            data = np.asarray(h5[h5dir]).astype(str)
            unique_names = [n.decode("utf8") for n in h5[namespath]]
            if sumdata is False:
                mot1_name = str(h5['mot1'].attrs["Name"])
                if mot1_name == "hxrf":
                    rowID = [n.decode("utf8") for n in np.squeeze(h5["mot1"])]
                else:
                    rowID = np.asarray(h5["mot1"]).astype(str)+'_'+np.asarray(h5["mot2"]).astype(str)
            else:
                rowID = h5dir
        rowID = np.array(rowID)
        fileID = rowID.copy()
        fileID[:] = h5file
    
        # 'flatten' the data
        if len(data.shape) == 3:
            rowID = rowID.reshape((data.shape[1]*data.shape[2]))
            fileID = fileID.reshape((data.shape[1]*data.shape[2]))
            data = data.reshape((data.shape[0],data.shape[1]*data.shape[2]))
        
    # at this point data is ordered alphabetically, will want to order this by atomic number Z.
    unique_names = np.asarray(unique_names)
    scatter = []
    scattername=[]
    if 'Compt' in unique_names:
        scattername.append('Compt')
        if rowID.size == 1:
            scatter.append(data[list(unique_names).index('Compt')])
            data = data[np.arange(len(unique_names))!=list(unique_names).index('Compt')]
        else:
            scatter.append(data[list(unique_names).index('Compt'),:])
            data = data[np.arange(len(unique_names))!=list(unique_names).index('Compt'),:]
        unique_names = unique_names[np.arange(len(unique_names))!=list(unique_names).index('Compt')]
    if 'Rayl' in unique_names:
        scattername.append('Rayl')
        if rowID.size == 1:
            scatter.append(data[list(unique_names).index('Rayl')])
            data = data[np.arange(len(unique_names))!=list(unique_names).index('Rayl')]
        else:
            scatter.append(data[list(unique_names).index('Rayl'),:])
            data = data[np.arange(len(unique_names))!=list(unique_names).index('Rayl'),:]
        unique_names = unique_names[np.arange(len(unique_names))!=list(unique_names).index('Rayl')]
    scatter = np.asarray(scatter)
    Z_array = [getZ(name.split(' ')[0]) for name in unique_names]
    unique_names = unique_names[np.argsort(Z_array)]
    rowID = np.asarray(rowID)
    if rowID.size == 1:
        data = data[np.argsort(Z_array)]
    else:
        sortID = np.argsort(Z_array)
        for i in range(rowID.shape[0]):
            data[:,i] = data[:,i][sortID]

    # write data as csv
    fileID = np.asarray(fileID)
    print("Writing "+csvfile+"...", end="")
    with open(csvfile, 'w') as csv:
        csv.write('FileID;RowID;'+';'.join(unique_names)+';'+str(';'.join(scattername))+'\n')
        if rowID.size == 1:
            if scattername != []:
                csv.write(str(fileID)+';'+str(rowID)+';'+str(';'.join(data[:]))+';'+str(';'.join(scatter[:]))+'\n')
            else:                
                csv.write(str(fileID)+';'+str(rowID)+';'+str(';'.join(data[:]))+'\n')
        else:
            for i in range(rowID.shape[0]):
                if scattername != []:
                    csv.write(str(fileID[i])+';'+str(rowID[i])+';'+str(';'.join(data[:,i]))+';'+str(';'.join(scatter[:,i]))+'\n')
                else:
                    csv.write(str(fileID[i])+';'+str(rowID[i])+';'+str(';'.join(data[:,i]))+'\n')
    print("Done.")
        
    
##############################################################################
def XProcH5_combine(files, newfile, ax=0):
    """
    Join two files together, stitched one after the other. This only combines raw files, 
    and as such should be done before any fitting or further processing.

    Parameters
    ----------
    files : list of strings, optional
        The H5 file paths to be combined.
    newfile : string, optional
        H5 file path of the new file.
    ax : integer, optional
        Axis along which the data should be concatenated. The default is 0.

    Returns
    -------
    None.

    """
    import h5py
    import numpy as np
    from datetime import datetime

    
    for index, file in enumerate(files):
        print("Reading "+file+"...", end="")
        f = h5py.File(file,'r')
        if index == 0:
            cmd = ''
            mot1 = []
            mot2 = []
            i0 = []
            tm = []
            icr0 = []
            ocr0 = []
            spectra0 = []
            if 'raw/I1' in f:
                i1flag = True
                i1 = []
            if 'raw/channel01' in f:
                ch1flag = True
                icr1 = []
                ocr1 = []
                spectra1 = []
            if 'fit' in f.keys():
                fitflag = True
                fit0 = []
                fitsum0 = []
                fitbkg0 = []
                if ch1flag:
                    fit1 = []
                    fitnms1 = []
                    fitsum1 = []
                    fitbkg1 = []
            if 'norm' in f.keys():
                normflag = True
                tmnorm = str(f['norm'].attrs["TmNorm"])
                if tmnorm == "True":
                    tmnorm = True
                else:
                    tmnorm = False
                normto = np.asarray(f['norm/I0'])
                norm0 = []
                norm0_err = []
                normnms0 = []
                normsum0 = []
                normbkg0 = []
                normsum0_err = []
                normbkg0_err = []
                if ch1flag:
                    norm1 = []
                    norm1_err = []
                    normnms1 = []
                    normsum1 = []
                    normbkg1 = []
                    normsum1_err = []
                    normbkg1_err = []

        try:
            cmd += f['cmd'][()].decode('utf8')
        except AttributeError:
            cmd += f['cmd'][()]

        mot1.append(np.array(f['mot1']))
        mot1_name = str(f['mot1'].attrs["Name"])
        mot2.append(np.array(f['mot2']))
        mot2_name = str(f['mot2'].attrs["Name"])
        i0.append(np.array(f['raw/I0']))
        if i1flag:
            i1.append(np.array(f['raw/I1']))
        tm.append(np.array(f['raw/acquisition_time']))
        icr0.append(np.array(f['raw/channel00/icr']))
        ocr0.append(np.array(f['raw/channel00/ocr']))
        spectra0.append(np.array(f['raw/channel00/spectra']))
        if ch1flag:
            icr1.append(np.array(f['raw/channel01/icr']))
            ocr1.append(np.array(f['raw/channel01/ocr']))
            spectra1.append(np.array(f['raw/channel01/spectra']))
        if fitflag:
            try:
                cfg0 = str(f['fit/channel00/cfg'][()].decode('utf8'))
            except AttributeError:
                cfg0 = str(f['fit/channel00/cfg'][()])
            fit0.append(np.array(f['fit/channel00/ims']))
            fitnms0 = [n.decode('utf8') for n in f['fit/channel00/names']]
            fitsum0.append(np.array(f['fit/channel00/sum/int']))
            fitbkg0.append(np.array(f['fit/channel00/sum/bkg']))
            if ch1flag:
                try:
                    cfg1 = str(f['fit/channel01/cfg'][()].decode('utf8'))
                except AttributeError:
                    cfg1 = str(f['fit/channel01/cfg'][()])
                fit1.append(np.array(f['fit/channel01/ims']))
                fitnms1 = [n.decode('utf8') for n in f['fit/channel01/names']]
                fitsum1.append(np.array(f['fit/channel01/sum/int']))
                fitbkg1.append(np.array(f['fit/channel01/sum/bkg']))
        if normflag:
            norm0.append(np.array(f['norm/channel00/ims']))
            norm0_err.append(np.array(f['norm/channel00/ims_stddev']))
            normnms0 = [n.decode('utf8') for n in f['norm/channel00/names']]
            normsum0.append(np.array(f['norm/channel00/sum/int']))
            normbkg0.append(np.array(f['norm/channel00/sum/bkg']))
            normsum0_err.append(np.array(f['norm/channel00/sum/int_stddev']))
            normbkg0_err.append(np.array(f['norm/channel00/sum/bkg_stddev']))
            if ch1flag:
                norm1.append(np.array(f['norm/channel01/ims']))
                norm1_err.append(np.array(f['norm/channel01/ims_stddev']))
                normnms1 = [n.decode('utf8') for n in f['norm/channel01/names']]
                normsum1.append(np.array(f['norm/channel01/sum/int']))
                normbkg1.append(np.array(f['norm/channel01/sum/bkg']))
                normsum1_err.append(np.array(f['norm/channel01/sum/int_stddev']))
                normbkg1_err.append(np.array(f['norm/channel01/sum/bkg_stddev']))

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

    if len(spectra0.shape) == 3:
        sumspec0 = np.sum(spectra0, axis=(0,1))
        maxspec0 = np.zeros(sumspec0.shape[0])
        for i in range(sumspec0.shape[0]):
            maxspec0[i] = spectra0[:,:,i].max()
    else:
        sumspec0 = np.sum(spectra0, axis=0)
        maxspec0 = np.zeros(sumspec0.shape[0])
        for i in range(sumspec0.shape[0]):
            maxspec0[i] = spectra0[:,i].max()
    if ch1flag is True:
        if len(spectra0.shape) == 3:
            sumspec1 = np.sum(spectra1, axis=(0,1))
            maxspec1 = np.zeros(sumspec1.shape[0])
            for i in range(sumspec1.shape[0]):
                maxspec1[i] = spectra1[:,:,i].max()
        else:
            sumspec1 = np.sum(spectra1, axis=0)
            maxspec1 = np.zeros(sumspec1.shape[0])
            for i in range(sumspec1.shape[0]):
                maxspec1[i] = spectra1[:,i].max()
            
    if fitflag:
        fit0 = np.concatenate(fit0, axis=ax+1)
        fitsum0 = np.sum(fitsum0, axis=(0))
        fitbkg0 = np.sum(fitbkg0, axis=(0))
        if ch1flag:
            fit1 = np.concatenate(fit1, axis=ax+1)
            fitsum1 = np.sum(fitsum1, axis=(0))
            fitbkg1 = np.sum(fitbkg1, axis=(0))
            
    if normflag:    
        norm0 = np.concatenate(norm0, axis=ax+1)
        norm0_err = np.concatenate(norm0_err, axis=ax+1)
        normsum0 = np.sum(normsum0, axis=(0))
        normbkg0 = np.sum(normbkg0, axis=(0))
        normsum0_err = np.sqrt(np.sum(normsum0_err, axis=(0)))
        normbkg0_err = np.sqrt(np.sum(normbkg0_err, axis=(0)))
        if ch1flag:
             norm1 = np.concatenate(norm1, axis=ax+1)
             norm1_err = np.concatenate(norm1_err, axis=ax+1)
             normsum1 = np.sum(normsum1, axis=(0))
             normbkg1 = np.sum(normbkg1, axis=(0))
             normsum1_err = np.sqrt(np.sum(normsum1_err, axis=(0)))
             normbkg1_err = np.sqrt(np.sum(normbkg1_err, axis=(0)))
          
        
        
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
    if fitflag:
        f.create_dataset('fit/channel00/ims', data=fit0, compression='gzip', compression_opts=4)
        f.create_dataset('fit/channel00/names', data=[n.encode('utf8') for n in fitnms0])
        f.create_dataset('fit/channel00/cfg', data=cfg0)
        f.create_dataset('fit/channel00/sum/int', data=fitsum0, compression='gzip', compression_opts=4)
        f.create_dataset('fit/channel00/sum/bkg', data=fitbkg0, compression='gzip', compression_opts=4)
        if ch1flag:
            f.create_dataset('fit/channel01/ims', data=fit1, compression='gzip', compression_opts=4)
            f.create_dataset('fit/channel01/names', data=[n.encode('utf8') for n in fitnms1])
            f.create_dataset('fit/channel01/cfg', data=cfg1)
            f.create_dataset('fit/channel01/sum/int', data=fitsum1, compression='gzip', compression_opts=4)
            f.create_dataset('fit/channel01/sum/bkg', data=fitbkg1, compression='gzip', compression_opts=4)
        dset = f['fit']
        dset.attrs["LastUpdated"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    if normflag:
        f.create_dataset('norm/I0', data=normto)
        f.create_dataset('norm/channel00/ims', data=norm0, compression='gzip', compression_opts=4)
        f.create_dataset('norm/channel00/ims_stddev', data=norm0_err, compression='gzip', compression_opts=4)
        f.create_dataset('norm/channel00/names', data=normnms0)
        f.create_dataset('norm/channel00/sum/int', data=normsum0, compression='gzip', compression_opts=4)
        f.create_dataset('norm/channel00/sum/bkg', data=normbkg0, compression='gzip', compression_opts=4)
        f.create_dataset('norm/channel00/sum/int_stddev', data=normsum0_err, compression='gzip', compression_opts=4)
        f.create_dataset('norm/channel00/sum/bkg_stddev', data=normbkg0_err, compression='gzip', compression_opts=4)
        if ch1flag:
            f.create_dataset('norm/channel01/ims', data=norm1, compression='gzip', compression_opts=4)
            f.create_dataset('norm/channel01/ims_stddev', data=norm1_err, compression='gzip', compression_opts=4)
            f.create_dataset('norm/channel01/names', data=normnms1)
            f.create_dataset('norm/channel01/sum/int', data=normsum1, compression='gzip', compression_opts=4)
            f.create_dataset('norm/channel01/sum/bkg', data=normbkg1, compression='gzip', compression_opts=4)
            f.create_dataset('norm/channel01/sum/int_stddev', data=normsum1_err, compression='gzip', compression_opts=4)
            f.create_dataset('norm/channel01/sum/bkg_stddev', data=normbkg1_err, compression='gzip', compression_opts=4)
        dset = f['norm']
        dset.attrs["LastUpdated"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        if tmnorm is True:
            dset.attrs["TmNorm"] = "True"
        else:
            dset.attrs["TmNorm"] = "False"
    f.close()                   
    print("Done")

##############################################################################
def rm_line(h5file, lineid, axis=1):
    """
    Delete a line or group of lines from a fitted dataset (can be useful when data interpolation has to occur but there are empty pixels)
    Note that this also removes lines from I0, I1, acquisition_time, mot1 and mot2! 
        If you want to recover this data one has to re-initiate processing from the raw data.

    Parameters
    ----------
    h5file : string
        File directory path to the H5 file containing the data to be removed.
    lineid : (list of) integer(s)
        Line id integers to be removed.
    axis : integer, optional
        axis along which the lines should be removed (i.e. row or column). The default is 1.

    Returns
    -------
    None.

    """
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
    """
    Sum together multiple h5 files of the same dimensions.
      Make sure it is also ordered similarly etc...
      Motor positions are averaged.
      Use at your own risk.

    Parameters
    ----------
    h5file : string
        File paths to the H5 files containing the data to be summed.
    newfilename : string
        File path to the newly generated H5 file.

    Returns
    -------
    None.

    """
    import h5py
    
    if type(h5files) is not type(list()):
        print("ERROR: h5files must be a list!")
        return
    else:
        for i in range(len(h5files)):
            if i == 0:
                print("Reading "+h5files[i]+"...", end="")
                f = h5py.File(h5files[i],'r')
                try:
                    cmd = f['cmd'][()].decode('utf8')
                except AttributeError:
                    cmd = f['cmd'][()]
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

##############################################################################
# Join Kmeans clustering clusters together, to allow for more straightforward imaging and sumspectrum representation.
#   Note that it is up to the user's discretion which clusters are combined, with respect to the physical interpretation of the results.
def join_clrs(h5file, channel='channel00', join=[[0,3],[1,2,4]], nosumspec=False):
    """
    Join Kmeans clustering clusters together, to allow for more straightforward imaging and sumspectrum representation.
    Note that it is up to the user's discretion which clusters are combined, with respect to the physical interpretation of the results.
    Joined data will be stored within the same h5file, in the 'joined_clr/'+channel data directory.

    Parameters
    ----------
    h5file : string
        File paths to the H5 files containing the data to be summed.
    channel : string, optional
        Detector channel for which clusters should be joined. The default is 'channel00', indicating that the software will sum data in the folder 'kmeans/channel00/'.
    join : list of lists containing integers, optional
        The integer values of the clusters that should be joined. Place the integers that should be joined in a single list. 
        The default is [[0,3],[1,2,4]], indicating that clusters 0 will be joined with 3 (new index:0), and additionally clusters 1, 2 and 4 will be joined together (new index:1).
    nosumspec : Bool, optional
        Set to True in order to omit calculating sumspectra, for instance when no sumspectra are available. The default is False.

    Returns
    -------
    None.

    """
    import h5py
    from datetime import datetime

    
    datadir = "kmeans/"+channel+"/"
    # read the h5 file
    with h5py.File(h5file,'r') as f:
        ims  = np.array(f[datadir+"ims"])
        if nosumspec is False:
            sumspeckeys = [key for key in f[datadir].keys() if 'sumspec' in key] #these should be ordered alphabetically
            sumspectra = []
            for key in sumspeckeys:
                sumspectra.append(np.array(f[datadir+"/"+key]))
    
    # join the clusters
    joined_ims = np.zeros(ims.shape)
    joined_sum = []
    for i in len(join):
        joined_ims[np.isin(ims, join[i])] = i
        if nosumspec is False:
            temp = sumspectra[0]*0.
            for index in join[i]:
                temp += sumspectra[index]
            joined_sum.append(temp)
        
    # write the joined cluster data
    with h5py.File(h5file,'r+') as file:
        try:
            del file['joined_clr/'+channel]
        except Exception:
            pass
        file.create_dataset('joined_clr/'+channel+'/nclusters', data=len(join))
        file.create_dataset('joined_clr/'+channel+'/data_dir_clustered', data=datadir.encode('utf8'))
        file.create_dataset('joined_clr/'+channel+'/ims', data=joined_ims, compression='gzip', compression_opts=4)
        dset = file['joined_clr']
        dset.attrs["LastUpdated"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        file.create_dataset('joined_clr/'+channel+'/join_id', data=join)
        if nosumspec is False:
            for i in len(join):
                dset = file.create_dataset('joined_clr/'+channel+'/sumspec_'+str(i), data=np.asarray(joined_sum)[i], compression='gzip', compression_opts=4)    
                dset.attrs["NPixels"] = np.asarray(np.where(joined_ims.ravel() == i)).size

        
        
##############################################################################
# bins the raw data of an h5 file by a given factor, using the bin_ndarray function from https://gist.github.com/derricw/95eab740e1b08b78c03f.
def bin_h5(h5file, binfactor):
    """
    Bin the raw data directory of a XProc H5 file by a given binning factor, thus reducing the amount of spectra to integrate at the cost of spatial resolution.
        A binning factor of 2 will group 2x2 pixels.
        Motor positions will be averaged during binning, whereas other counters are summed. Detector channels are not binned, i.e. the energy resolution is not compromised.
        Be wary of binning raw data of (snake wise) continuous scans as motor positions may not match on a pixel-id level.
        Binned data is overwritten, so be wary of raw data loss.

    Parameters
    ----------
    h5file : String
        File directory to the H5 file that should be binned. This file should contain a /raw directory, as generated by the XProc software.
    binfactor : int
        Factor with which the spatial resolution should decrease. I.e. a binning factor of 2 effectively halves the resolution (2x2 pixels are reduced to 1).

    Returns
    -------
    None.

    """
    import h5py
    
    print("Processing "+h5file+"...", end="")
    with h5py.File(h5file,'r+') as f:
        keys = [key for key in f['raw'].keys() if 'channel' in key]
        mot1 = np.array(f['mot1'])
        newshape = (int(mot1.shape[0]/binfactor), int(mot1.shape[1]/binfactor))
        rest = (int((mot1.shape[0]%binfactor)/2), int((mot1.shape[1]%binfactor)/2))
        mot1 = bin_ndarray(mot1[rest[0]:rest[0]+newshape[0]*binfactor,rest[1]:rest[1]+newshape[1]*binfactor], new_shape=newshape, operation='mean')
        mot1_name = str(f['mot1'].attrs["Name"])
        del f['mot1']
        dset = f.create_dataset('mot1', data=mot1, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot1_name
        mot2 = np.array(f['mot2'])
        newshape = (int(mot2.shape[0]/binfactor), int(mot2.shape[1]/binfactor))
        rest = (int((mot2.shape[0]%binfactor)/2), int((mot2.shape[1]%binfactor)/2))
        mot2 = bin_ndarray(mot2[rest[0]:rest[0]+newshape[0]*binfactor,rest[1]:rest[1]+newshape[1]*binfactor], new_shape=newshape, operation='mean')
        mot2_name = str(f['mot2'].attrs["Name"])
        del f['mot2']
        dset = f.create_dataset('mot2', data=mot1, compression='gzip', compression_opts=4)
        dset.attrs['Name'] = mot2_name
        i0 = np.array(f['raw/I0'])
        newshape = (int(i0.shape[0]/binfactor), int(i0.shape[1]/binfactor))
        rest = (int((i0.shape[0]%binfactor)/2), int((i0.shape[1]%binfactor)/2))
        i0 = bin_ndarray(i0[rest[0]:rest[0]+newshape[0]*binfactor,rest[1]:rest[1]+newshape[1]*binfactor], new_shape=newshape, operation='sum')
        del f['raw/I0']
        f.create_dataset('raw/I0', data=i0, compression='gzip', compression_opts=4)
        try:
            i1 = np.array(f['raw/I1'])
            newshape = (int(i1.shape[0]/binfactor), int(i1.shape[1]/binfactor))
            rest = (int((i1.shape[0]%binfactor)/2), int((i1.shape[1]%binfactor)/2))
            i1 = bin_ndarray(i1[rest[0]:rest[0]+newshape[0]*binfactor,rest[1]:rest[1]+newshape[1]*binfactor], new_shape=newshape, operation='sum')
            del f['raw/I1']
            f.create_dataset('raw/I1', data=i1, compression='gzip', compression_opts=4)
        except KeyError:
            pass
        tm = np.array(f['raw/acquisition_time'])
        newshape = (int(tm.shape[0]/binfactor), int(tm.shape[1]/binfactor))
        rest = (int((tm.shape[0]%binfactor)/2), int((tm.shape[1]%binfactor)/2))
        tm = bin_ndarray(tm[rest[0]:rest[0]+newshape[0]*binfactor,rest[1]:rest[1]+newshape[1]*binfactor], new_shape=newshape, operation='sum')
        del f['raw/acquisition_time']
        f.create_dataset('raw/acquisition_time', data=tm, compression='gzip', compression_opts=4)
        
        for key in keys:
            icr0 = np.array(f['raw/'+key+'/icr'])
            newshape = (int(icr0.shape[0]/binfactor), int(icr0.shape[1]/binfactor))
            rest = (int((icr0.shape[0]%binfactor)/2), int((icr0.shape[1]%binfactor)/2))
            icr0 = bin_ndarray(icr0[rest[0]:rest[0]+newshape[0]*binfactor,rest[1]:rest[1]+newshape[1]*binfactor], new_shape=newshape, operation='sum')
            del f['raw/'+key+'/icr']
            f.create_dataset('raw/'+key+'/icr', data=icr0, compression='gzip', compression_opts=4)
            ocr0 = np.array(f['raw/'+key+'/ocr'])
            newshape = (int(ocr0.shape[0]/binfactor), int(ocr0.shape[1]/binfactor))
            rest = (int((ocr0.shape[0]%binfactor)/2), int((ocr0.shape[1]%binfactor)/2))
            ocr0 = bin_ndarray(ocr0[rest[0]:rest[0]+newshape[0]*binfactor,rest[1]:rest[1]+newshape[1]*binfactor], new_shape=newshape, operation='sum')
            del f['raw/'+key+'/ocr']
            f.create_dataset('raw/'+key+'/ocr', data=ocr0, compression='gzip', compression_opts=4)
            spectra0 = np.array(f['raw/'+key+'/spectra'])
            newshape = (int(spectra0.shape[0]/binfactor), int(spectra0.shape[1]/binfactor), spectra0.shape[2])
            rest = (int((spectra0.shape[0]%binfactor)/2), int((spectra0.shape[1]%binfactor)/2))
            spectra0 = bin_ndarray(spectra0[rest[0]:rest[0]+newshape[0]*binfactor,rest[1]:rest[1]+newshape[1]*binfactor,:], new_shape=newshape, operation='sum')
            del f['raw/'+key+'/spectra']
            f.create_dataset('raw/'+key+'/spectra', data=spectra0, compression='gzip', compression_opts=4)
            sumspec = np.sum(spectra0[:], axis=(0,1))
            maxspec = np.zeros(sumspec.shape[0])
            for i in range(sumspec.shape[0]):
                maxspec[i] = spectra0[:,:,i].max()
            del f['raw/'+key+'/sumspec']
            del f['raw/'+key+'/maxspec']
            f.create_dataset('raw/'+key+'/sumspec', data=sumspec, compression='gzip', compression_opts=4)
            f.create_dataset('raw/'+key+'/maxspec', data=maxspec, compression='gzip', compression_opts=4)
    print("Done")


def bin_ndarray(ndarray, new_shape, operation='sum'):
    # obtained from https://gist.github.com/derricw/95eab740e1b08b78c03f.
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.
    Number of output dimensions must match number of input dimensions.
    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray
