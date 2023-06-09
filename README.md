# XProc
Program with data stream for XRF analysis, including fitting, normalisation, quantification, absorption corrections and imaging.
The repository can be easily installed using the following command in a (unix) terminal:

`git clone https://www.github.com/PieterTack/XProc`

## Data processing using xrf_proc and plotims
### Using XProc
XProc is a python based API that should be used for XRF spectral integration (making use of PyMca5 routines), and subsequent data processing.
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
For convenient data processing, it is advised to use a uniform data structure. Until such time as all synchrotrons and subsequent beamlines unanimously decide to follow the same data structure, it is up to the multiple-beamline-frequenting user to define their own structure. This is the structure that is expected for the data routines provided by XProc.py and plotims.py:

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

```
def ConvID15H5(h5id15, scanid, scan_dim, mot1_name='hry', mot2_name='hrz', ch0id='falconx_det0', ch1id='falconx2_det0', i0id='fpico2', i0corid=None, i1id='fpico3', i1corid=None, icrid='trigger_count_rate', ocrid='event_count_rate', atol=None, sort=True):
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

Returns
-------
bool
    Returns False upon failure.

```

```
def ConvP06Nxs(scanid, sort=True, ch0=['xspress3_01','channel00'], ch1=None, readas1d=False):
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

Returns
-------
None.

```

```
def ConvDeltaCsv(csvfile):
Read the Delta Premium handheld CSV files and restructure as H5 for further processing


Parameters
----------
csvfile : String
    File path to the CSV file to be converted.

Returns
-------
None.

```

```
def ConvEdaxSpc(spcprefix, outfile, scandim, coords=[0,0,1,1]):
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

```




#### Fitting XRF data
Xrf_proc makes use of the PyMca5 python library in order to process XRF spectra. As such, when reporting data processed with xrf_proc in literature, it is strongly advised to at least also cite the PyMca software package.1
This process assumes a PyMca fit configuration file was created before running fit_xrf_batch(). This can be done by loading in the uniformalised h5 file in the PyMca gui (version>=5.3) and making a configuration file based on the /raw/maxpsec and /raw/sumspec data directories. Alternatively, one can launch xrf_config_gui.py for a GUI combining the PyMca fitting algorithm and configuration structure, with some useful functionalities such as the KLM line library as found in for instance the AXIL software package.

```
def  fit_xrf_batch(h5file, cfgfile, standard=None, ncores=None, verbose=None):
Fit a batch of xrf spectra using the PyMca fitting routines. A PyMca config file should be supplied.
The cfg file should use the SNIP background subtraction method. Others will fail as considered 'too slow' by the PyMca fast linear fit routine itself.
Additionally, using the standard option also integrates the individual spectra, not only the sum spectrum, without fast linear fit. This can take a long time!!
   

Parameters
----------
h5file : string
    File directory path to the H5 file containing the data.
cfgfile : string
    File path to the PyMca-type CFG configuration file containing the fitting parameters.
standard : NoneType, optional
    If not a NoneType (e.g. string) then all spectra are integrated separately without using the fast linear fit procedure. The default is None.
ncores : Integer, optional
    The amount of cores over which the multiprocessing package should split the task. Values of -1, 0 and None allow the system to use all available cores, minus 1. The default is None.
verbose : Boolean, optional
    If not None, the PyMca fit returns errors encountered during the procedure. The default is None.

Returns
-------
None.

```

When calling this routine, make sure h5file is not accessed by any other program (e.g. still open in pymca or HDFView, …) to prevent file corruption.

#### Data normalisation
Corrections for detector dead time, primary beam intensity fluctuations and acquisition times can be made using 

```
def norm_xrf_batch(h5file, I0norm=None, snake=False, sort=False, timetriggered=False, tmnorm=False, halfpixshift=True, mot2nosort=False):
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

Returns
-------
None.

```
When calling this routine, make sure h5file is not accessed by any other program (e.g. still open in pymca or HDFView, …) to prevent file corruption.

#### Distribution images
Xrf_proc also has convenient, built-in functions to plot and save the (normalised) fitted data images. 
Data is plotted in a collated image, with intensity and scale bars. The viridis colour scale is used.

```
def hdf_overview_images(h5file, datadir, ncols, pix_size, scl_size, log=False, rotate=0, fliph=False, cb_opts=None, clim=None):
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
log : Boolean, optional
    If True, the Briggs logarithm of the data is displayed. The default is False.
rotate : integer, optional
    Amount of degrees, rounded to nearest 90, over which images should be rotated. The default is 0.
fliph : Boolean, optional
    If True, data is flipped over the horizontal image axes (after optional rotation). The default is False.
cb_opts : Plotims.Colorbar_opt class, optional
    User supplied intensity scale colorbar options for imaging. The default is None.
clim : list, optional
    List containing the lower and upper intensity limit to apply to the plots. The default is None, indicating no specific limits.

Returns
-------
None.

```

The function automatically plots channel00 detector channel. If present, also plots the channel02. Images are stored as .png files with suffix ‘_ch0_overview.png’ and ‘_ch2_overview.png’ respectively, optionally preceded by ‘_log’ in case of log scaling.

#### Concentration files
Following functions will often require information on the quantitative composition of a reference material or general sample matrix. For this purpose, a ‘concentration file’ was designed containing the relevant compositional information for a given sample/material. These ASCII text files have an extension of ‘.cnc’, and are structured as following:
  
It is important to note that the value under ‘Number of elements’ matches the amount of rows below ‘Z’ perfectly. Additionally, it is advised that the sum of the certified concentrations equals 1000000 ppm (i.e. 100%). 

```
def read_cnc(cncfile):
Read in the data of a concentration (.cnc) file.

Parameters
----------
cncfile : String
    .cnc file path.

Returns
-------
rv : Cnc() Class
    Cnc class containing the data contained within the .cnc file.

```

#### Detection limits and elemental yields
Detection limits are calculated based on the net peak intensity (Ip,i¬) and background intensity (Ib,i) for a given element as obtained from an XRF sum spectrum, as well as well as the certified concentration of the corresponding element (ci) within the reference material: 
$〖DL〗_i=(3∙ \sqrt(I_(p,i) ))⁄I_(b,i) ∙c_i$
Error estimation is performed by standard error propagation rules, taking into account the certified concentration error if available (if not available, the concentration is considered a constant with no error).

```
def calc_detlim(h5file, cncfile, plotytitle="Detection Limit (ppm)", sampletilt=90):
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

```

As detection limits are calculated, the function also calculates the elemental yield for all elements present in both XRF fitting and concentration file. These yields are expressed as ppm/count/s, for straightforward semi-quantitative concentration estimates of the normalised scan data. Note that no correction for self-absorption or probed sample mass are taken into account here (which is available using the quant_with_ref() function).
In order to make plots of the detection limit values, for more straightforward data interpretation, the plot_detlim() function can be called. This function is also automatically called at the end of calc_detlim() for further improved convenience.

```
def plot_detlim(dl, el_names, tm=None, ref=None, dl_err=None, bar=False, save=None, ytitle="Detection Limit (ppm)"):
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

```

On the scatter plots a best fit curve is added, to display the general trend of detection limit as a function of atomic number.

#### Reference based quantification
Quantification based on comparison with reference materials can be a powerful and straightforward tool to obtain semi-quantitative information on the elemental composition of a sample. The xrf_proc package can provide this information. Do note that it is still up to the user’s discretion to judge the reliability of the obtained results.

```
def quant_with_ref(h5file, reffiles, channel='channel00', norm=None, absorb=None, snake=False, div_by_rhot=None, mask=None):
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
div_by_rhot : float or None, optional
    If keyword div_by_rhot is not None, the calculated aerial concentration is divided by a user-supplied div_by_rhot [cm²/g] value. The default is None.
mask : String, list of strings, 2D binary integer array or None, optional
    A data mask can be provided. This can either be a reference to a kmeans cluster ID supplied as a string or list of strings, e.g. 'kmeans/CLR2' or ['CLR2','CLR4'],
        or a string data path within the H5 file containing a 2D array of size equal to the H5 file image size, where 0 values represent pixels to omit
        from the quantification and 1 values are pixels to be included. Alternatively, a 2D array can be directly supplied as argument. The default is None.

Yields
------
bool
    Returns False on error.

```

For some purposes it may be useful to present the quantified data, divided by a mean matrix composition, as is for instance regularly done when comparing quantified results of chondritic materials. For this purpose, one can use div_by_cnc():

```
def div_by_cnc(h5file, cncfile, channel=None):
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

```

#### Statistical analysis
A typical step in XRF data analysis, following fitting and normalisation procedures, is to perform data reduction methods such as K-means clustering and/or principal component analysis (PCA). The following functions allow one to do so with minimal effort:

```
def h5_kmeans(h5file, h5dir, nclusters=5, el_id=None, nosumspec=False):
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

```

```
def h5_pca(h5file, h5dir, nclusters=5, el_id=None, kmeans=False):
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

```

Although these functions are useful to perform clustering on readily available data within the H5 files, it can be useful to have access to general PCA and Kmeans functions to be used on any dataset:

```
def Kmeans(rawdata, nclusters=5, el_id=None, whiten=True):
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

```

```
def PCA(rawdata, nclusters=5, el_id=None):
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

```

Additionally, although this function is not a part of xrf_proc.py but of plotims.py, a convenient tool for statistical analysis can be the studying of correlation plots. This can be fairly easily done using the `plotims.plot_correl()` function:

```
def plot_correl(imsdata, imsnames, el_id=None, save=None):
Display correlation plots.

Parameters
----------
imsdata : float array
    imsdata is a N*M*Y float array containing the signal intensities of N*M datapoints for Y elements.
imsnames : string
    imsnames is a string array of Y elements, containing the names of the corresponding elements.
el_id : integer list, optional
    el_id should be a integer list containing the indices of the elements to include in the plot. The default is None.
save : string, optional
    File path as which the image should be saved. The default is None.

Returns
-------
None.

```

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
import Xproc as Fit
import Xims as Ims


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

