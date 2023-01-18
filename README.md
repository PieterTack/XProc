# xrf_proc
Program with data stream for XRF analysis, including fitting, normalisation, quantification, absorption corrections and imaging.
The repository can be easily installed using the following command in a (unix) terminal:

`git clone https://www.github.com/PieterTack/xrf_proc`

## Data processing using xrf_proc and plotims
### Using xrf_proc
Xrf_proc is a python based API that should be used for XRF spectral integration (making use of PyMca5 routines), and subsequent data processing.
Relies on the following packages:
* PyMca5 v5.3 or higher
* plotims (https://github.com/PieterTack/plotims)
* numpy
* scipy
* h5py
* matplotlib v3.1 or higher

optional:
* tomopy (required for ct reconstruction)
* xraylib

Install using:
```
pip install numpy
pip install scipy
pip install h5py
pip install PyMca5
pip install PyQt5
conda install -c conda-forge tomopy
conda install -c conda-forge xraylib
```

#### Uniform h5 file structure
For convenient data processing, it is advised to use a uniform data structure. Until such time as all synchrotrons and subsequent beamlines unanimously decide to follow the same data structure, it is up to the multiple-beamline-frequenting user to define their own structure. This is the structure that is expected for the data routines provided by xrf_proc.py and plotims.py:

- `Filename.h5`
	- `cmd`	the scan command line executed during the experiment
	- `raw`	contains all raw data – should never be overwritten!
		- `I0`	primary X-ray beam intensities
		- `I1` 	transmission signal intensity
		- `acquisition_time`	nD array of acquisition time/point
		- `channel00`	uniform name of the first fluorescence detector
			- `icr`	nD array of incoming count rates/point
			- `maxspec`	the max spectrum over all measurement points
			- `ocr`	nD array of outgoing count rates/point
			- `spectra`	(n+1)D array containing the spectra for each point. Shape: M×N×C (C=#detector channels)
			- `sumspec`	the sum spectrum of all measurement points
		- `channel01` [optional] the same structure of channel00 is applied here for channel02 if a second detector channel was available.
	- `mot1`	nD array containing mot1 motor positions (snakemesh motor)
	- `mot2`	nD array containing mot2 motor positions

--- folders below are added by additional data processing---

	- `fit`	contains the fitted data
		- `channel00`
			- `cfg`		file path of the used PyMca config file
			- `ims`		(n+1)D array of integrated peak intensities for each measurement point. Shape: C×M×N (C=#lines)
			- `names`	string array containing the integrated line names (e.g. ‘Fe K’), the string is encoded as a ‘utf8’ for writing, so use `[n.decode(‘utf8’) for n in names]` to use in python scripts
			- `sum`
				- `bkg`	background intensities for each integrated line of the sum spectrum, as defined by the fitting algorithm. For PyMca this is the background in the range of 3-sigma around the centre peak value
				- `int`	peak intensities for each line from the sum spectrum
		- `channel01` [optional]
	- `norm`	contains normalised data. Normalisation here includes corrections for I0, acquisition time, detector dead time. Normalisations for e.g. Compton intensity or self-absorption are preserved for the quantification step.
		- `I0`		I0 value to which each data point is normalised
		- `channel00`
			- `ims`	(n+1)D array of normalised peak intensities for each measurement point. Shape: C×M×N (C=#lines)
			- `names`	string array containing the integrated line names (e.g. ‘Fe K’), the string is encoded as a ‘utf8’ for writing, so use `[n.decode(‘utf8’) for n in names]` to use in python scripts
			- `sum`
				- `bkg`	Normalised background intensities for each integrated line of the sum spectrum.
				- `int`	Normalised peak intensities for each integrated line of the sum spectrum.
	- `tomo`	tomographic reconstruction images are stored here
		- `channel<XX>` or `I1`	
			- `names`	string array containing the reconstructed line names (e.g. ‘Fe K’), the string is encoded as a ‘utf8’ for writing, so use `[n.decode(‘utf8’) for n in names]` to use in python scripts
			- `rotation_center`	the used centre of rotation during the reconstruction algorithm
			- `ims`	(n+1)D array of reconstructed image intensities for each line. Shape: C×M×N (C=#lines)
	- `detlim`	detection limits calculated by 〖DL〗_i=(3∙√(I_(p,i) ))⁄I_(b,i) ∙c_i Note that detection limits can only be calculated for samples with known composition
		- `<cncfile>`	the name of the used cnc file is mentioned here. A cnc file contains the full compositional information for a given reference material, and can be found in lvserver.ugent.be:/usr/local/cnc_files/
			- `channel00`
				- `1000s`
					- `data`	array containing the detection limits for 1000s live time acquisition time (as calculated from DL1s / sqrt(1000).
					- `stddev`	absolute 1-sigma standard error on the detection limits
				- `1s`
					- `data`	array containing the detection limits for 1s live time acquisition time as calculated from the sum spectrum data
					- `stddev`	absolute 1-sigma standard error on the detection limits
				- `names`	string array containing the element lines’ names (e.g. ‘Fe K’), the string is encoded as a ‘utf8’ for writing, so use `[n.decode(‘utf8’) for n in names]` to use in python scripts
			- `channel01` [optional]
		- `unit`		The unit in which all detection limits and errors are expressed (default: ppm)
	- `elyield`	contains the elemental yields for a given reference material and fluorescence line. (typical unit: ppm/ct/s)
		- `<cncfile>`		the name of the used cnc file
			- `channel00`
				- `names`	string array containing the element lines’ names (e.g. ‘Fe K’)
				- `stddev`	absolute 1-sigma standard error on the elemental yield
				- `yield`	array containing the elemental yields. The corresponding unit is defined as an attribute (typically ppm/ct/s)
			- `channel01` [optional]
	- `quant`	contains the quantified data
		- `channel<XX>`
			- `ims`		(n+1)D array of quantified peak intensities for each measurement point. Shape: C×M×N (C=#lines)
			- `names`	string array containing the element lines’ names (e.g. ‘Fe K’)
			- `ratio_exp`	If self-absorption correction is performed during quantification, this is a nD array containing the experimental Ka/Kb ratio of the element used to judge the absorption factor (e.g. Fe-Ka/Fe-Kb)
			- `ratio_th`	If self-absorption correction is performed during quantification, this is float containing the theoretical Ka/Kb ratio of the element used to judge the absorption factor (e.g. Fe-Ka/Fe-Kb)
			- `refs`		string array containing the file paths to the reference material scans that were used during the quantification process (i.e. whose elemental yields were used)
			- `rhot`		a nD array containing the rho*t factor calculated based on the ratio_exp and ratio_th values for the given sample matrix.
	- `kmeans`	Kmeans clustering results
		- `channel<XX>`
			- `data_dir_clustered`	directory within the h5 file containing the data that was clustered (e.g. norm/channel02/ims)
			- `el_id`	string array containing the elements on which the clustering was based
			- `ims`		nD array where each pixel value represents the attributed Kmeans cluster
			- `nclusters`	total amount of clusters the data was clustered in
			- `sumspec_0`	sum spectrum of all points allocated to cluster 0
			- `…`
			- `sumspec_N` (N=nclusters-1) sum spectrum of all points allocated to cluster N
	- `PCA`	PCA data reduction results
		- `channel<XX>`
			- `el_id`	string array containing the elements on which PCA was based
			- `nclusters`	total amount of principal components that were extracted from the PCA
			- `ims`		(n+1)D array of PC scores. Shape: C×M×N (C=#PCs)
			- `names`	string array containing the PC names (i.e. PC0, …, PC<nclusters-1>)
			- `RVE`		float array containing relative value explained eigenvalues
			- `Loadings`	float array containing the eigenvectors for each PC as a function of el_id, to be used to construct loading plots.
	- `rel_dif`		provides info on quantified data, normalised for a given concentration (e.g. for the chondritic CI mean concentration)
		- `channel<XX>`
			- `cnc`		string cnc file path of the mean compositional matrix
			- `ims`		(n+1)D array of normalised peak intensities for each measurement point. Shape: C×M×N (C=#lines)
			- `names`	string array containing the element lines’ names (e.g. ‘Fe K’)

Currently, the data format assumes the application of 2D scans, with 1 or 2 fluorescence detectors monitoring the signal (named channel00 and channel02, as was originally the case at beamline P06). Clearly, this structure could be expanded to include more fluorescence detectors, and/or more motor positions etc. However, for backwards compatibility (and manageable data sizes) it is advised to retain the proposed format, and emulate a 3D volumetric scan as a stack of 2D scans.
Typically, float or integer array data are compressed with a gzip compression algorithm of strength 4. Strings are encoded as utf8 (n.decode(‘utf8’) when extract in other python scripts).

#### Preparing h5 file
In most cases the data collected at a beamline is spread over several documents, or contains more information than strictly required for further XRF analysis. Additionally, different beamlines often use (slightly) different data formats and/or structures. As such, a uniform data structure is required for convenient further processing.
In order to use the xrf_proc python functions, one has to include them in your main python script:
```
import sys
sys.path.insert(1, ‘</data_directory/contaning/xrf_fit>’)
import xrf_proc as Fit
```
  
Functions can then be used following, e.g.:
`Fit.ConvID15H5(arg1)`

Due to differences in beamline data formats, typically separate “conversion” functions are required for different beamlines. Currently, the following functions are supported:

>	`h5id15convert(h5id15, scanid, scan_dim, mot1_name='hry', mot2_name='hrz', ch0id='falconx_det0', ch1id='falconx2_det0', i0id='fpico2', i0corid=None, i1id='fpico3', i1corid=None, icrid='trigger_count_rate', ocrid='event_count_rate', atol=None, sort=True)`
>	* h5id15: (string) beamline stored h5 data file location, e.g. ‘id15/NIST_611_0001.h5’. A list of strings can be supplied to combine different measurements into 1 file, in which case the scanid argument should also be a list of identical length.
>	* scanid: (string) the identifier of the relevant scan in the h5 file, typically the scan number followed by .1, e.g. ‘4.1’. A list of strings can be supplied to combine different scans into 1 file, in which case all scanid’s are obtained from the same h5id15 file.
>	* scan_dim: (tuple) the scan dimensions, i.e. (10,10) for a 10×10 mapping
>	* mot1_name: [optional] (string) the name of mot1, the least moving motor, default: ‘hry’
>	* mot2_name: [optional] (string) the name of mot2, the most moving motor, default: ‘hrz’
>	* ch0id: [optional] (string) the detector identifier mnemonic for channel00
>	* ch1id: [optional] (string) the detector identifier mnemonic for channel01
>	* i0id: [optional] (string) the identifier mnemonic for I0
>	* i0corid: [optional] (string) If not None (default), the identifier mnemonic for the signal with which I0 should be corrected
>	* i1id: [optional] (string) the identifier mnemonic for I1
>	* i1corid: [optional] (string) If not None (default), the identifier mnemonic for the signal with which I1 should be corrected
>	* icrid: [optional] (string) the identifier mnemonic for the detector ICR data
>	* ocrid: [optional] (string) the identifier mnemonic for the detector OCR data
>	* atol: [optional] (float) the absolute tolerance parameter as used by the numpy.allclose() function to determine the motor incremental direction. Default: None, which corresponds to 1e-4.
>	* sort: [optional] (Boolean) if True (default) the data is sorted based on the corresponding motor positions
>	* Returns False on error, on success stores data in a new h5 file.

>	`ConvP06Nxs(scanid, sort=True, ch0=['xspress3_01','channel00'], ch2=['xspress3_01','channel02'])`
>	* Scanid: (string) general scan directory path, e.g. '/data4/202010_p06_brenker/data/orl0/scan_00142'. A list of strings can be supplied to combine multiple scans into a single file.
>	* Sort: [optional] (Boolean) if True, the data is ordered based on the sorting of mot1 and mot2. Sorting is required in order to obtain appropriate image results in case of e.g. snake scans. However, in some cases it is better to omit sorting in the merge step, and only do it after the fitting procedure (during data normalisation step), e.g. in the case of scans that are time triggered instead of motor position triggered.
>	* ch0: [optional] (list of strings) Contains the detector mnemonics for the detector identifier(s) of channel00. Each list item can be another list in the case when multiple detectors should be summed for channel00 in the merged file (e.g. ch0 = ['xspress3_01', ['channel00','channel02']])
>	* ch1: [optional] (list of strings) Contains the detector mnemonics for the detector identifier(s) of channel01. Each list item can be another list in the case when multiple detectors should be summed for channel02 in the merged file (e.g. ch2 = ['xspress3_01', ['channel00','channel02']])
>	* Returns False on error, on success stores data in a new h5 file with suffix ‘_merge.h5’

#### Fitting XRF data
Xrf_proc makes use of the PyMca5 python library in order to process XRF spectra. As such, when reporting data processed with xrf_proc in literature, it is strongly advised to at least also cite the PyMca software package.1
This process assumes a PyMca fit configuration file was created before running fit_xrf_batch(). This can be done by loading in the uniformalised h5 file in the PyMca gui (version>=5.3) and making a configuration file based on the /raw/maxpsec and /raw/sumspec data directories. Alternatively, one can launch xrf_config_gui.py for a GUI combining the PyMca fitting algorithm and configuration structure, with some useful functionalities such as the KLM line library as found in for instance the AXIL software package.

>	`Fit.fit_xrf_batch(h5file, cfgfile, standard=None, ncores=None, verbose=None)`
>	* h5file: (string) The uniform structured h5 file path containing the raw data to be fitted.
>	* cfgfile: (string) The PyMca configuration file path to be used for the fitting procedure. A list of paths with length equal to the amount of detector channels contained in the h5file, where each detector channel will then be fitted with the corresponding configuration file from the list. Note: if one wants to apply the fast linear PyMca fit (see standard option) a simple SNIP background should be used. In any case, this is often also sufficient for more detailed fits.
>	* standard: [optional] (string) if None (default) no detection limits or elemental yields will be determined. Additionally, a fast linear fit will be applied providing a very fast and fairly reliable fit result. (note: differences mainly occur in very heterogeneous and/or noisy datasets). For a more detailed fit (e.g. with more complicated spectral background and/or point-per-point fitting) a non-None value should be supplied (e.g. ‘something’). When fitting a standard reference material measurement it can be useful to calculate detection limits and/or elemental yields. To do so, supply the cnc-file path destination as a string.
>	* ncores: [optional] (int) The amount of cores to use. If None (default) then the maximum amount of cores-1 will be used for the fitting, as defined by multiprocessing.cpu_count(). A value of -1 or 0 has the same result. Any other integer defines the amount of cores to use. Multicore processing is only applied when standard is not None.
>	* verbose: if not None (default) the function will print much more information to the standard output window. Note that this can also slow down the fitting process considerably.
>	* Fitted data for each detector channel is written in h5file. Note that consecutive calls to fit_xrf_batch() will always overwrite the last data.

When calling this routine, make sure h5file is not accessed by any other program (e.g. still open in pymca or HDFView, …) to prevent file corruption.

#### Data normalisation
Corrections for detector dead time, primary beam intensity fluctuations and acquisition times can be made using 

>	`norm_xrf_batch(h5file, I0norm=None, snake=False, sort=False, timetriggered=False)`
>	* h5file: (string) The uniform structured h5 file path containing the fitted data to be normalised.
>	* I0norm: [optional] (float) I0 value to which each pixel must be normalised. If None (default) the data is normalised to the maximum I0 value in /raw/I0.
>	* snake: [optional] (Boolean) If True the data is treated for snake mesh scans, where each consecutive line in mot1 is approached from the opposite direction, by interpolating the data to a regular motor position grid. This function also correct for the half-pixel shift often accompanied with such processes. If False (default) the motor position interpolation etc is ignored.
>	* sort: [optional] (Boolean) If True data is sorted. It is advised to sort the data here, if it was not sorted by fit_xrf_batch(). Sorting is required for snake mesh scans. Default: False (as it is assumed sorting already occurred in fit_xrf_batch() )
>	* timetriggered: [optional] (Boolean) In case of time triggered detector readout, usually more datapoints are available than expected from the regular grid scan pattern. As such, data has to be interpolated to a regular grid. The code has some algorithms to detect this case automatically, but if certain this data is time triggered it is safest to state so by setting this parameter to True. Additionally, if timetriggered is True one should typically also define snake as True to make full use of the interpolation algorithms. Default: False. 
>	* tmnorm: [optional] (Boolean) Default is False. When set True, the data will be additionaly normalised for the measurement time (tm) specified for each scan point in the H5 file. Usually this data is already contained within the I0 value, so be wary to normalise for tm again.
>	* halfpixshift: [optional] (Boolean) Default is True. This parameter is only used when snake is set to True. When False, the half-pixel shift correction, which typically is required for appropriate data interpolation during snake scans, is omitted.
>	* Normalised data for each detector channel is written in h5file. Note that consecutive calls to norm_xrf_batch() will always overwrite the last data.
When calling this routine, make sure h5file is not accessed by any other program (e.g. still open in pymca or HDFView, …) to prevent file corruption.

#### Distribution images
Xrf_proc also has convenient, built-in functions to plot and save the (normalised) fitted data images. 
Data is plotted in a collated image, with intensity and scale bars. The viridis colour scale is used.

>	`hdf_overview_images(h5file, datadir, ncols, pix_size, scl_size, log=False, rotate=0, fliph=False, cb_opts=None, clim=None)`
>	* h5file: (string) The uniformly structured h5 file path containing the data to plot.
>	* datadir: (string) The data directory within the h5 file containing the information to plot, e.g. ‘norm’
>	* ncols: (int) The amount of columns to distribute the elemental distribution images in on the collated image.
>	* pix_size: (float) The size of a single pixel along the horizontal direction of the scan, in µm.
>	* scl_size: (float) The size of the scale bare to draw on the images along the horizontal direction of the scan, in µm.
>	* log: [optional] (Boolean) True to plot the intensity as a log10 scale. False (default) for a linear scale.
>	* rotate: [optional] (int) default is 0 (no rotation). An amount of degrees (rounded to the nearest 90) with which the image should be rotated, following the numpy.rot90() function.
>	* fliph: [optional] (Boolean) Default is False, set True to flip the image horizontally. This operation is performed after rotation, when requested.
>	* cb_opts: [optional] (plotims.Colorbar_opt class). When None (default) the default colorbar settings of plotims are used.
>	* clim: [optional] (list of 2 floats) Default is None. If not None, the image intensities will be limited between the fractions defined by clim as follows: [lower limit, upper limit]. It is advised to use values between 0 and 1, e.g. clim=[0.1, 0.9] to limit the image values between min+0.1*(max-min) and min+0.9*(max-min)

The function automatically plots channel00 detector channel. If present, also plots the channel02. Images are stored as .png files with suffix ‘_ch0_overview.png’ and ‘_ch2_overview.png’ respectively, optionally preceded by ‘_log’ in case of log scaling.

#### Concentration files
Following functions will often require information on the quantitative composition of a reference material or general sample matrix. For this purpose, a ‘concentration file’ was designed containing the relevant compositional information for a given sample/material. These ASCII text files have an extension of ‘.cnc’, and are structured as following:
  
It is important to note that the value under ‘Number of elements’ matches the amount of rows below ‘Z’ perfectly. Additionally, it is advised that the sum of the certified concentrations equals 1000000 ppm (i.e. 100%). 

>	`read_cnc(cncfile)`
>	* cncfile: (string) File path of the concentration file
>	* returns a dictionary-like Cnc class object, with following items: name (string), z (int array), conc (float array), err (float array), density (float), mass (float), density (float) and thickness (float). The units of each match the units used in the original concentration file.

#### Detection limits and elemental yields
Detection limits are calculated based on the net peak intensity (Ip,i¬) and background intensity (Ib,i) for a given element as obtained from an XRF sum spectrum, as well as well as the certified concentration of the corresponding element (ci) within the reference material: 
$〖DL〗_i=(3∙ \sqrt(I_(p,i) ))⁄I_(b,i) ∙c_i$
Error estimation is performed by standard error propagation rules, taking into account the certified concentration error if available (if not available, the concentration is considered a constant with no error).

>	`calc_detlim(h5file, cncfile)`
>	* h5file: (string) The uniformly structured h5 file path containing the normalised data corresponding to a reference material measurement.
>	* cncfile: (string) The concentration file path corresponding to the reference material under investigation.

As detection limits are calculated, the function also calculates the elemental yield for all elements present in both XRF fitting and concentration file. These yields are expressed as ppm/count/s, for straightforward semi-quantitative concentration estimates of the normalised scan data. Note that no correction for self-absorption or probed sample mass are taken into account here (which is available using the quant_with_ref() function).
In order to make plots of the detection limit values, for more straightforward data interpretation, the plot_detlim() function can be called. This function is also automatically called at the end of calc_detlim() for further improved convenience.

>	`plot_detlim(dl, el_names, tm=None, ref=None, dl_err=None, bar=False, save=None)`
>	* dl: (float) nD array of dimensions ([n_ref, ][n_tm, ]n_elements). In its simplest form (DLs for a single reference materials and measurement time) this is an array of N elements where N are the amount of elements for which detection limits were calculated. Providing multiple measurement times or reference materials plots all of these on the same graph. Note that plotting a large variety of datasets on a single figure can quickly make the image very crowded and confusing.
>	* el_names: (string) Array containing the fluorescence line names for which the detection limits were calculated.
>	* tm: [optional] (float) The different measurement times for which data is provided in dl. Default is None, denoting all values in dl correspond to the same measurement time.
>	* ref: [optional] (string) Array containing the names of the different references for which data is provided in dl. Default is None, denoting all values in dl correspond to the same reference material.
>	* dl_err: [optional] (float) Array of same shape as dl containing the 1-sigma absolute errors on dl. Plot_detlim() always displays the 3-sigma error bars. Default is None, when error bar plotting is omitted.
>	* bar: [optional] (Boolean) If True a histogram style plot is displayed. If False (default) a scatter plot is provided.
>	* save: [optional] (string) If None (default) the plot is not displayed, if not None the image is saved in the location provided here. 

On the scatter plots a best fit curve is added, to display the general trend of detection limit as a function of atomic number.

  #### Reference based quantification
Quantification based on comparison with reference materials can be a powerful and straightforward tool to obtain semi-quantitative information on the elemental composition of a sample. The xrf_proc package can provide this information. Do note that it is still up to the user’s discretion to judge the reliability of the obtained results.

>	`quant_with_ref(h5file, reffiles, channel='channel00', norm=None, absorb=None, snake=False)`
>	* h5file: (string) File path location of the H5 file containing normalised data.
>	* reffiles: (string) File path location of the H5 file containing elemental yield (el_yield) data as calculated by calc_detlim(), corresponding to the measurement of a reference material. A list of file paths can be provided, in which case the average elemental yield will be calculated for each element over all reference files. If an element in the h5file is not present in the listed refs, its yield is estimated through linear interpolation of the closest neighbouring atoms with the same line type. If Z is at the start or end of the reference elements, the yield will be extrapolated from the first or last 2 elements in the reference. If only 1 element in the reference has the same line type as the quantifiable element, but does not have the same Z, the same yield is used regardless as inter/extrapolation is impossible.
>	* channel: [optional] (string) The detector channel to quantify. Default is ‘channel00’
>	* norm: [optional] (string) The signal to which the elemental yields should be corrected (e.g. ‘Compt’, ‘Fe K’). The provided signal has to be present in all reffiles and h5file. If None (default) no normalisation is performed, and elemental yields are used as-is.
>	* absorb: [optional] (string tuple (['element'], 'cnc file')) The element that will be used to determine the self-absorption factor for each pixel (e.g. ‘Fe’) and the concentration file corresponding to the average sample (matrix) composition. If None (default) no self-absorption will be performed. The element will be used to find Ka and Kb line intensities and correct for their respective ratio using concentration values from the provided concentration file. Theoretical line intensity ratios are derived from the Xraylib library if available, and from PyMca5 internal libraries elsewise.
>	* snake: [optional] (Boolean) If this data corresponds to a snake mesh scan, set True. Otherwise set False (default). As the intensity ratio and corresponding self-absorption correction factor are determined from the raw data, interpolation to the appropriate motor position grid has to be performed in order to be able to match the correction factors to the normalised XRF data.

For some purposes it may be useful to present the quantified data, divided by a mean matrix composition, as is for instance regularly done when comparing quantified results of chondritic materials. For this purpose, one can use div_by_cnc():

>	`div_by_cnc(h5file, cncfile, channel=None)`
>	* h5file: (string) The H5 file path location, containing quantified data.
>	* cncfile: (string) The concentration file path location, corresponding to the matrix composition for which the quantified h5file data should be normalised.
>	* channel: [optional] (string) The detector channel to perform this division for. If None (default) the calculations are performed or all detector channels in the h5file.

#### Statistical analysis
A typical step in XRF data analysis, following fitting and normalisation procedures, is to perform data reduction methods such as K-means clustering and/or principal component analysis (PCA). The following functions allow one to do so with minimal effort:

>	`h5_kmeans(h5file, h5dir, nclusters=5, el_id=None)`
>	* h5file: (string) The H5 file path location, containing the data on which to perform the K-means clustering.
>	* h5dir: (string) The directory of the data to be clustered within the H5 file structure, e.g. ‘norm/channel02/ims’
>	* nclusters: [optional] (int) The amount of K-means cluster to divide the data into. By default, 5 clusters are requested.
>	* el_id: [optional] (int) List containing the element indices corresponding to the data segments that should be included in the K-means clustering process. If None (default) all elements within h5dir are included.
>	* Upon completion, h5file will contain the image indicating which cluster each pixel was attributed to, as well as sum spectra corresponding to each cluster group (sum spectra are not normalised for the amount of pixels contained within the cluster etc)

>	`h5_pca(h5file, h5dir, nclusters=5, el_id=None, kmeans=False) `
>	* h5file: (string) The H5 file path location, containing the data on which to perform the PCA analysis
>	* h5dir: (string) 
>	* nclusters: (string) The directory of the data to be clustered within the H5 file structure, e.g. ‘norm/channel02/ims’
>	* el_id: [optional] (int) List containing the element indices corresponding to the data segments that should be included in the K-means clustering process. If None (default) all elements within h5dir are included.
>	* kmeans: [optional] (Boolean) If True, K-means clustering will be performed on the PCA score images using the same amount of clusters as defined in nclusters. If False (True) K-means clustering will be omitted.
>	* Upon completion, PCA score images, eigenvectors (loadings) and principal component eigenvalues (RVE) are stored. If the kmeans option was True also Kmeans data are stored in the kmeans/ directory within the H5 file.

Although these functions are useful to perform clustering on readily available data within the H5 files, it can be useful to have access to general PCA and Kmeans functions to be used on any dataset:

>	`Kmeans(rawdata, nclusters=5, el_id=None)`
>	* rawdata: (float) 2D float array containing the data to be clustered of shape M×N with N the amount of elements and M the data points. Alternatively, a 3D float array can be supplied of shape N×M×L (M×L being the shape of the data image, multiplied by N elements) 
>	* nclusters: [optional] (int) The amount of K-means cluster to divide the data into. Default: 5.
>	* el_id: [optional] (int) List containing the element indices along axis N. If None (default) all elements are included.
>	* Returns two arguments: clusters, distortion. Clusters contains the attributed cluster index for each data point. Distortion is the distance of each point and its cluster centre as provided by the scipy.cluster.vq.vq() function.

>	`PCA(rawdata, nclusters=5, el_id=None)`	
>	* rawdata: (float) 2D float array containing the data to be clustered of shape M×N with N the amount of elements and M the data points. Alternatively, a 3D float array can be supplied of shape N×M×L (M×L being the shape of the data image, multiplied by N elements)
>	* nclusters: [optional] (int) The amount of principal components to divide the data into. Default: 5.
>	* el_id: [optional] (int) List containing the element indices along axis N. If None (default) all elements are included.
>	* Returns 3 arguments: scores, evals, evecs which are the principal component scores (images), eigenvalues (RVE, not yet normalised) and eigenvectors (loading values) respectively.

Additionally, although this function is not a part of xrf_proc.py but of plotims.py, a convenient tool for statistical analysis can be the studying of correlation plots. This can be fairly easily done using the `plotims.plot_correl()` function:

>	`plot_correl(imsdata, imsnames, el_id=None, save=None)`	
>	* imsdata: (float) N*M*Y array containing the signal intensities of N*M datapoints for Y elements .
>	* imsnames: (string) array of Y elements, containing the names of the corresponding elements
>	* el_id: [optional] (int) list containing the indices of the elements to include in the plot
>	* save: [optional] (string) File path to save the created image

The generated image will display the correlation scatter plots in the top right half, including a best linear fit (black dashed line) and 95% confidence interval as well as R² values for the linear fit correlation. The main diagonal contains the intensity distribution histogram plots of each variable. The lower left half contains kernel density distribution plots, as well as the Pearson correlation coefficients with a marking of the confidence interval (***: 0.1%CI, **:1%CI, *:5%CI, no stars: <5%CI).

In order to use this function, make sure to import plotims:
```
import sys
sys.path.insert(1, ‘</data_directory/contaning/plotims>’)
import plotims as Ims
```
  
## Example file
Below is an example python processing file:
```python
import sys
sys.path.insert(1, 'D:/School/PhD/python_pro/plotims')
sys.path.insert(1, 'D:/School/PhD/python_pro/xrf_proc')
import xrf_fit as Fit
import plotims as Ims


def prepare_p06():
    Fit.ConvP06Nxs('dir/ref/scan_00001')
    Fit.ConvP06Nxs(['dir/ct/scan_00002','dir/ct/scan_00003'])


def prepare_id15():
    Fit.ConvID15H5('dir/srmscan.h5', '1.1', (10, 10), mot1_name='hrz', mot2_name='hry')
    Fit.ConvID15H5('dir/ctscan.h5', '1.1', (100, 101), mot1_name='hrrz', mot2_name='hry')
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
    
    # calculate tomographic reconstruction for 'channel01' signal of 'norm' data
    #       estimate centre of rotation value based from 'Sr-K' signal
    #       display final reconstruction images in collimated overview style with 8 colums
    tomo_proc.h5_tomo_proc('dir/preppedfile.h5', rot_mot='mot2', channel='channel01', signal='Sr-K', rot_centre=None, ncol=8)
#%%
def correlation_plots():
    import h5py
    
    f = h5py.File('dir/preppedfile.h5','r')
    data = np.moveaxis(np.array(f['tomo/channel01/slices']),0,-1)
    data[np.isnan(data)] = 0.
    names = [n.decode('utf8') for n in f['tomo/channel01/names']]
    f.close()
    print([names[i] for i in [1,2,3,4,5,6,8,9,14]])
    Ims.plot_correl(data, names, el_id=[1,2,3,4,5,6,8,9,14], save='dir/preppedfile_correlation.png')
 ```       

 
## References
(1) Solé, V. A.; Papillon, E.; Cotte, M.; Walter, P.; Susini, J. Spectrochimica Acta Part B: Atomic Spectroscopy 2007, 62, 63-68.

