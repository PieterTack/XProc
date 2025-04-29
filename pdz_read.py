# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 08:49:22 2025

@author: prrta
"""

# strongly based on https://github.com/fligt/read_pdz/tree/master
import struct
import re
import pandas as pd
import numpy as np
import os

PDZ11_STRUCT_DICT = {
    'pdz11_2048_channels' : {'xformat': '2X-i-h-34X-2d-86X-2i-10X-2f-188X-Z-*X', 
                          'param_keys': ['pdz-version', '??', 'NumberOfChannels', '??', '??', 
                                         'eVPerChannel', '??', 'RawCounts', 'ValidCounts', '??',  'XrayVoltageInkV', 
                                         'XrayFilamentCurrentInMicroAmps', '??', 'Intensity (2048 channels)', '??']}, 
    'pdz11_1024_channels' : {'xformat': '2X-i-h-34X-2d-86X-2i-10X-2f-24X-z-*X', 
                          'param_keys': ['pdz-version', '??', 'NumberOfChannels', '??', '??', 
                                         'eVPerChannel', '??', 'RawCounts', 'ValidCounts', '??',  'XrayVoltageInkV', 
                                         'XrayFilamentCurrentInMicroAmps', '??', 'Intensity (1024 channels)', '??']} 
}
PDZ_25_STRUCTURE_DICT = {
25:  {'xformat': 'hi-10X-i', 
      'param_keys': ['pdz_type', 'block_size', 'FileFormatString?', '??']}, 
1:   {'xformat': 'hi-2S-6s-2S-h-S-T', 
      'param_keys': ['block_type', 'block_size', '??', 'SerialString', '??', '??', '??', '??', '??', '??', 
                     '??', '??', '??', '??', '??', '??', '??']}, 
2:   {'xformat': 'hi3i8f-*X', 
      'param_keys': ['block_type', 'block_size', '??', 'RawCounts', 'ValidCounts', '??', '??', 
                     '??', 'ActiveTimeInSeconds', 'DeadTimeInSeconds', 'ResetTimeInSeconds', 
                     'LiveTimeInSeconds', 'TotalElapsedTimeInSeconds', '??']}, 
3:   {'xformat': 'hi-3i9f7hfhfhfhf8hfhi-*Z', 
      'param_keys': ['block_type', 'block_size', '??', 'RawCounts', 'ValidCounts', 
                     '??', '??', '??', 'ActiveTimeInSeconds', 'DeadTimeInSeconds', 
                     'ResetTimeInSeconds', 'LiveTimeInSeconds', 'XrayVoltageInkV', 'XrayFilamentCurrentInMicroAmps', 
                     'Filter1ElementAtomicNum', 'Filter1Thickness', 'Filter2ElementAtomicNum', 'Filter2Thickness', 
                     'Filter3ElementAtomicNum', 'Filter3Thickness', '??', 'DetectorTempInC', '??', 
                     '??', '??', 'eVPerChannel', '??', 'eVStart', 
                     'Year', 'Month', 'AM/PM code?', 'Day', 'Hour', 'Minutes', 'Seconds', 
                     '??', 'NosePressureInMilliBars', 'NumberOfChannels', 'NoseTemperatureInC', 
                     '??', 'Intensity_2048_channels']}}


def skip_bytes(xformat, arr, verbose=True): 
    '''Skip a number of bytes as specified in `xformat` string. 
    
    If multiplier is `*` then skip all. 
    '''

    # get multiplier  
    if xformat == 'X': 
        n_bytes = 1
    elif xformat == '*X': 
        n_bytes = len(arr)    
    else: 
        n_bytes = int(re.sub('(^\d+)X', '\g<1>', xformat))

    skipped = [b''.join(arr[0:n_bytes])] 

    arr = arr[n_bytes:] 

    if verbose: 
        print(skipped)

    return skipped, arr 
def read_table(xformat, arr, verbose=True): 
    '''Extract numbered table'''

    assert xformat == 'T', 'Incorrect format string'

    [table_length], arr = parse('<i', arr, verbose=False) 

    table = []
    for i in range(table_length): 
        [num], arr = parse('<h', arr, verbose=False)  
        [string], arr = read_strings('S', arr, verbose=False)
        table.append([f'#{num}', string]) 
        
    if verbose: 
        print(table)

    return table, arr
def read_counts(xformat, arr, verbose=True): 
    '''Extract counts. '''

    assert xformat == 'Z'  or xformat == 'z', 'Incorrect spectral data format string. Should be `Z` or `z`' 
    
    if xformat == 'Z': 
        n_channels = 2048 
    elif xformat == 'z': 
        n_channels = 1024 
        
    # make struct compatible format string 
    _format = f'<{n_channels}i'

    counts, arr = parse(_format, arr, verbose=False) 
    counts = np.array(counts)
        
    if verbose: 
        print(counts)

    return counts, arr
def read_strings(xformat, arr, verbose=True): 
    '''Parse `n` variable length character strings preceded by a length integer. 
     
    '''
    # get multiplier  
    if xformat == 'S': 
        n = 1
    else: 
        n = int(re.sub('(^\d+)S', '\g<1>', xformat))

    # parse strings 
    string_list = [] 
    while n > 0: 
        [length], arr = parse('<i', arr, verbose=False) # read length 
        n_bytes = 2 * length 

        # do some testing 
        assert (n_bytes > 1) and (type(n_bytes) is int), f'{n_bytes} is invalid string length' 
        
        char_list, arr = parse(f'<{n_bytes}c', arr, verbose=False) 
        string = b''.join(char_list).decode(encoding='utf-16') 
        string_list.append(string) 
        n = n -1 
        
    if verbose: 
        print(string_list) 

    return string_list, arr
def multiparse(xformat, arr, param_keys=None, verbose=True): 
    '''Parse segments in extendend format string `xformat` e.g. '<i5f-2S-T-3S-S-f' '''

    
    parts = re.split('-', xformat) 

    result_list = []
    for p in parts: 
        if 'S' in p:
            result, arr = read_strings(p, arr, verbose=False) 
        elif p == 'T': 
            result, arr = read_table(p, arr, verbose=False) 
        elif 'X' in p: 
            result, arr = skip_bytes(p, arr, verbose=False) 
            
        # four spectral data scenarios here: 

        # (1) 2048 channels at end of array and skip any bytes before
        elif p == '*Z': 
            # split array  
            n_channels = 2048 
            arr_0 = arr[:-n_channels*4] # head 
            arr_1 = arr[-n_channels*4:] # tail             
            skipped, _ = skip_bytes('*X', arr_0, verbose=False)
            counts, arr = read_counts('Z', arr_1 , verbose=False) # arr should now be empty 
            result = [skipped, counts]             
        # (2) 1024 channels at end of array and skip any bytes before
        elif p == '*z': 
            # split array  
            n_channels = 1024 
            arr_0 = arr[:-n_channels*4] # head 
            arr_1 = arr[-n_channels*4:] # tail 
            
            skipped, _ = skip_bytes('*X', arr_0, verbose=False)
            counts, arr = read_counts('z', arr_1 , verbose=False) # arr should now be empty 
            result = [skipped, counts]             
        # (3) 2048 channels not at end of array 
        elif p == 'Z': 
            result, arr = read_counts(p, arr, verbose=False)
            result = [result]
        # (4) 1024 channels not at end of array 
        elif p == 'z': 
            result, arr = read_counts(p, arr, verbose=False)
            result = [result]    
        
        else: 
            result, arr = parse(p, arr, verbose=False) 
            
        result_list.extend(result)   

    # if verbose: 
    #     if param_keys == None: 
    #         result_df = pd.DataFrame({'values': result_list})
    #     else: 
    #         result_df = pd.DataFrame({'values': result_list, 'param_keys': param_keys})      

    return result_list, arr   
def parse(format, arr, verbose=True): 
    '''Parse first bytes from bytes array `arr` into human readable text according to `format` string. 
    
    See struct library for format string specification. For example, '<ff' would result 
    the first 8 bytes to be converted into two Little-Endian floats. 
    
    Returns: `parsed` list and remaining bytes array of `tail_arr` unprocessed `values`.'''

    if not format.startswith('<'): 
        format = f'<{format}' 
    size = struct.calcsize(format)
    buffer = arr[0:size]
    tail_arr = arr[size:]

    parsed = list(struct.unpack(format, buffer)) 

    if verbose: 
        print(parsed)
    
    return parsed, tail_arr
def get_block_at(pdz_arr, start): 
    '''Read first data block from bytes array `pdz_arr` from index position `start`. 

    Assumes that first 4 bytes are (block type and size)
    
    Returns: `block_dict`, `block` 
    '''

    file_size = len(pdz_arr)
    
    [block_type, block_size], arr = parse('hi', pdz_arr[start:], verbose=False)
    
    stop = start + block_size + 6 # four bytes extra due to `dtype` and `size` shorts plus two empty pad bytes? 

    # read block bytes
    arr = pdz_arr[start:stop] 
    
    block_dict = {'block_type': block_type, 'block_size': block_size, 'start': start, 'stop': stop, 
                  'file_size': file_size, 'bytes': arr}

    return block_dict
def file_to_bytes(pdz_file): 
    '''Read all bytes from filepath `pdz_file` into a byte array. 
    
    Returns: `pdz_arr` (numpy array of bytes)
    '''

    with open(pdz_file, 'rb') as fh: 
        blob = fh.read() 
        
    pdz_arr = np.array([v[0] for v in struct.iter_unpack('c', blob)])
    #pdz_arr = bytearray(blob)

    return pdz_arr 
def check_pdz_type(pdz_file, verbose=True): 
    '''Read first two bytes and for legacy pdz files number of detector channels to check pdz file type.'''

    pdz_bytes = file_to_bytes(pdz_file) 
    first_two_bytes = struct.unpack('<h', pdz_bytes[0:2])[0] 

    if first_two_bytes == 25: 
        pdz_type = 'pdz25' 
    elif first_two_bytes == 257: 
        n_channels = struct.unpack('<h', pdz_bytes[6:8])[0] 
        if n_channels == 1024:
            pdz_type = 'pdz11_1024_channels'
        elif n_channels == 2048:
            pdz_type = 'pdz11_2048_channels'  
        else: 
            pdz_type = f'pdz11_with_unexpected_number_of_{n_channels}_channels'
    else:
        pdz_type = f'pdz_type_unknown:{first_two_bytes}'
                
    return pdz_type 

def read_pdz(pdzfile):
    pdz_type = check_pdz_type(pdzfile)
    pdz_bytes = file_to_bytes(pdzfile) 

    # get spectral data xformat string for pdz_type 
    if pdz_type == 'pdz25': 
        xformat = PDZ_25_STRUCTURE_DICT[3]['xformat']      
    elif pdz_type == 'pdz11_2048_channels': 
        xformat = PDZ11_STRUCT_DICT['pdz11_2048_channels']['xformat']
    elif pdz_type == 'pdz11_1024_channels': 
        xformat = PDZ11_STRUCT_DICT['pdz11_1024_channels']['xformat'] 
    else: 
        return f'Unknown pdz type: {pdz_type}'
 

    # pdz25 
    if pdz_type == 'pdz25': 
        # extracting spectral data in block 
        block_list = [] 
        start = 0
        total_size = len(pdz_bytes)
        while start < total_size: 
            block_dict = get_block_at(pdz_bytes, start)
            start = block_dict['stop']
            block_list.append(block_dict)
        
        # select type 3 blocks
        b3_list = [b for b in block_list if b['block_type'] == 3] 
        n_spectra = len(b3_list) 
        if n_spectra > 1: 
            print(f'Found multiple spectral data blocks: {n_spectra}. Only parsing first spectrum. ')
        arr = b3_list[0]['bytes'] 
    
        # parsing spectrum parameters and data 
        parsed, tail = multiparse(xformat, arr, verbose=False)
        n_channels = parsed[37]
        if n_channels != 2048: 
            print(f'Found unexpected number of channels in pdz metadata: {n_channels}')
        spe = parsed[-1] #element 40
        tm = parsed[8]
        tubecurrent = parsed[13]
        icr = parsed[3]
        ocr = parsed[4]
        basename = os.path.basename(pdzfile)


    # pdz11_2048_channels 
    elif pdz_type == 'pdz11_2048_channels': 
        
        xformat = PDZ11_STRUCT_DICT['pdz11_2048_channels']['xformat']
        parsed, tail = multiparse(xformat, pdz_bytes, verbose=False) 

        # The problem with legacy files is that we do not know if and at which position 
        # the energy offset is stored. So we need to set start_keV=0  
        # parsing spectrum parameters and data 
        parsed, tail = multiparse(xformat, pdz_bytes, verbose=False)
        
        n_channels = parsed[2]
        if n_channels != 2048: 
            print(f'Found unexpected number of channels in pdz metadata: {n_channels}')
        spe = parsed[13] #element 40
        tm = 1. #TODO: no measurement time is found in the data structure?!
        print('No measurement time identified... Selected default 1s/pt.')
        tubecurrent = parsed[11]
        icr = parsed[7]
        ocr = parsed[8]
        basename = os.path.basename(pdzfile)
        
    
    # pdz11_1024_channels
    elif pdz_type == 'pdz11_1024_channels': 
        
        xformat = PDZ11_STRUCT_DICT['pdz11_1024_channels']['xformat']
        parsed, tail = multiparse(xformat, pdz_bytes, verbose=False) 

        # The problem with legacy files is that we do not know if and at which position 
        # the energy offset is stored. So we need to set start_keV=0  
        # parsing spectrum parameters and data 
        parsed, tail = multiparse(xformat, pdz_bytes, verbose=False)
        
        n_channels = parsed[2]
        if n_channels != 1024: 
            print(f'Found unexpected number of channels in pdz metadata: {n_channels}')
        spe = parsed[13] #element 40
        tm = 1. #TODO: no measurement time is found in the data structure?!
        print('No measurement time identified... Selected default 1s/pt.')
        tubecurrent = parsed[11]
        icr = parsed[7]
        ocr = parsed[8]
        basename = os.path.basename(pdzfile)
    
    else: 
        print(f'pdz_type: {pdz_type} Sorry, this specific pdz type is not yet implemented...')
        return 

    return {'ID':basename, 
            'spectrum':spe.astype(float), 
            'current':float(tubecurrent), 
            'icr':float(icr), 
            'ocr':float(ocr), 
            'realtime':float(tm)}
