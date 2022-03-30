# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:22:43 2021

@author: prrta

xrf_config: graphical user interface to generate PyMca5 config files (including XRF spectrum calibration etc.)
adding functionality of displaying relative XRF emission line intensities during peak allocation. 
"""

import sys
# sys.path.insert(1, 'D:/School/PhD/python_pro/plotims')
# import plotims as IMS

import matplotlib
matplotlib.use('Qt5Agg') #Render to Pyside/PyQt canvas
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure


import numpy as np
from scipy.signal import find_peaks
import h5py
from PyMca5.PyMcaPhysics.xrf import ClassMcaTheory
from PyMca5.PyMcaPhysics.xrf import Elements


from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QApplication, QWidget, QDialog
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout
from PyQt5.QtWidgets import QCheckBox, QPushButton, QLabel, QCheckBox, \
    QLineEdit, QTabWidget, QFileDialog, QComboBox, QTreeWidget, QTreeWidgetItem, \
    QScrollArea
    

class Poll_h5dir():
    def __init__(self, h5file, parent=None):
        self.paths=[]
        self.h5file = h5file
        # extract all Dataset paths from the H5 file
        f = h5py.File(self.h5file, 'r')
        self.paths = self.descend(f, paths=None)

        # in this case, we only want spectra
        self.paths = [path for path in self.paths if 'sumspec' in path or 'maxspec' in path]    
        self.specs = [np.array(f[path]) for path in self.paths]
        
        f.close()

    def spe(self, path):
        return self.specs[self.paths.index(path)]

    def dirs(self):
        return self.paths
        
    def descend(self, obj, paths=None):
        if paths is None:
            paths = []
            
        if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
            for key in obj.keys():
                self.descend(obj[key], paths=paths)
        elif type(obj)==h5py._hl.dataset.Dataset:
            paths.append(obj.name)
        return paths

def compile_pymca_dict(ele, energy=None):
    # extract a dict containing the lines and radrates for a given element
    dict = Elements._getUnfilteredElementDict(ele, energy)
    myrays = ['Ka xrays', 'Kb xrays', 'L1 xrays', 'L2 xrays', 'L3 xrays', 'M xrays'] # we only want these rays, the others are just repetitions
    rays = [ray for ray in dict['rays'] if ray in myrays]
    
    newdict = []
    for i in range(len(rays)):
        for line in dict[rays[i]]:
            if dict[line]['rate'] > 0.0:
                lineID = line
                if 'Ka' in rays[i]:
                    group = 'Ka'
                    lineID = line[:-1] #from K lines remove the a or b extension
                elif 'Kb' in rays[i]:
                    group = 'Kb'
                    lineID = line[:-1] #from K lines remove the a or b extension
                elif 'L1' in rays[i]:
                    group = 'L1'
                elif 'L2' in rays[i]:
                    group = 'L2'
                elif 'L3' in rays[i]:
                    group = 'L3'
                elif 'M' in rays[i]:
                    group = 'M'
                newdict.append({'linegroup': group, 'line': lineID, 'energy': dict[line]['energy'], 'rate': dict[line]['rate'] })
    return newdict


class CalibrateWindow(QDialog):
    def __init__(self, mainobj, parent=None):
        super(CalibrateWindow, self).__init__(parent)
        self.newcte = None
        self.newgain = None
        self.mainobj = mainobj
        self.peakpos = find_peaks(self.mainobj.rawspe, height=np.percentile(self.mainobj.rawspe, 40.), distance=10)[0] #TODO: better methods may be available to find local maxima
        self.cte = self.mainobj.ConfigDict['detector']['zero']
        self.gain = self.mainobj.ConfigDict['detector']['gain']
        self.chnls = []
        self.energies = []

        layout_main = QVBoxLayout()
        layout_lineid = QHBoxLayout()

        # QComboBox with all elements
        self.elements = QComboBox()
        el_list = ['Energy']
        for i in range(len(Elements.ElementList)):
            el_list.append("{:d}".format(i+1)+" "+Elements.ElementList[i])
        self.elements.addItems(el_list)
        self.elements.setCurrentIndex(26) #select Fe by default
        layout_lineid.addWidget(self.elements)
        # QComboBox with all lines for a given element
        self.linedict = compile_pymca_dict('Fe')
        items = [line['line']+": "+"{:.3f}".format(line['energy'])+"keV ("+"{:.5f}".format(line['rate'])+")" for line in self.linedict]
        rates = [float(line['rate']) for line in self.linedict if 'K' in line['linegroup']] #TODO: issues if no K lines would be in list, or if not amongst the first in list
        self.lines = QComboBox()
        self.lines.addItems(items)
        self.lines.setCurrentIndex(rates.index(np.max(rates)))
        layout_lineid.addWidget(self.lines)
        # LineEdit to provide energy float if requested
        self.energy = QLineEdit("")
        self.energy.setMaximumWidth(50)
        self.energy.setValidator(QDoubleValidator(0., 1E3,3))
        layout_lineid.addWidget(self.energy)
        # button to add line
        self.add = QPushButton("Add2Calib")
        self.add.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.add.setMaximumWidth(75)
        layout_lineid.addWidget(self.add)
        layout_main.addLayout(layout_lineid)
        
        # QLineEdit (noneditable) with channelnr --> energyvalue (el-linename)
        self.scroll_win = QScrollArea()
        self.caliblist_txt = "Lines in calibration:\n---\nchannelnr  -->  energy"
        self.caliblist = QLabel(self.caliblist_txt)
        self.caliblist.setAlignment(Qt.AlignTop)
        self.scroll_win.setWidget(self.caliblist)
        self.scroll_win.setWidgetResizable(True)
        self.scroll_win.setMinimumHeight(180)
        layout_main.addWidget(self.scroll_win)

        # Initiate calculation of calibration
        self.calc = QPushButton("Calculate")
        self.calc.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.calc.setMaximumWidth(60)
        layout_main.addWidget(self.calc)
        
        # show window
        self.setLayout(layout_main)
        self.setWindowTitle('Energy Calibration')
        self.setWindowModality(Qt.ApplicationModal)
        self.setFocusPolicy(Qt.StrongFocus)
        self.show()
 
        # draw vertical line on max peak in spectrum
        self.currentid = np.argmax(mainobj.rawspe)
        self.update_calibline(self.currentid)

 
        # event handling
        self.calc.clicked.connect(self.calculate) # calculate button
        self.add.clicked.connect(self.addlist) # add current line to calibration list
        self.elements.currentIndexChanged.connect(self.change_linelist) #if a different element is selected, adjust possible lines.
        

    def keyPressEvent(self, event): #handle arrow keys
        if event.key() == Qt.Key_Up:
            tempid = np.where(self.peakpos > self.currentid)
            if tempid[0].size == 0: #no values found
                self.currentid = self.peakpos[0]
            else:
                self.currentid = self.peakpos[np.min(tempid)]
            self.update_calibline(self.currentid)
        elif event.key() == Qt.Key_Down:
            tempid = np.where(self.peakpos < self.currentid)
            if tempid[0].size == 0: #no values found
                self.currentid = self.peakpos[-1]
            else:
                self.currentid = self.peakpos[np.max(tempid)]
            self.update_calibline(self.currentid)
        elif event.key() == Qt.Key_Left:
            self.currentid -= 1
            self.update_calibline(self.currentid)
        elif event.key() == Qt.Key_Right:
            self.currentid += 1
            self.update_calibline(self.currentid)
        else:
            QDialog.keyPressEvent(self, event)

    def change_linelist(self):
        if self.elements.currentIndex() == 0: #Energy tab: set lines to ---
            items = ['----']
            currentid = 0
        else:
            current_element = Elements.ElementList[self.elements.currentIndex()-1]
            self.linedict = compile_pymca_dict(current_element)
            items = [line['line']+": "+"{:.3f}".format(line['energy'])+"keV ("+"{:.5f}".format(line['rate'])+")" for line in self.linedict]
            rates = [float(line['rate']) for line in self.linedict if 'K' in line['linegroup']]
            currentid = rates.index(np.max(rates))
        self.lines.clear()
        self.lines.addItems(items)
        self.lines.setCurrentIndex(currentid)

    def update_calibline(self, currentid):
        if len(self.mainobj.mpl.axes.lines) > 1: # something makes that on first keypress the originally added axvline is not in axes.lines
            self.mainobj.mpl.axes.lines[-1].remove()
        self.mainobj.mpl.axes.axvline(self.cte+currentid*self.gain, color='red')
        self.mainobj.mpl.canvas.draw()
            
    def addlist(self):
        if self.elements.currentIndex() == 0:
            if self.energy.text() != "":
                linename = self.energy.text()+' keV'
                self.chnls.append(self.currentid)
                self.energies.append(float(self.energy.text()))
            else:
                return
        else:
            linename = "{:.3f}".format(self.linedict[self.lines.currentIndex()]['energy']) + \
                " keV ("+Elements.ElementList[self.elements.currentIndex()-1] + \
                "-" + self.linedict[self.lines.currentIndex()]['line']+")"
            self.chnls.append(self.currentid)
            self.energies.append(float(self.linedict[self.lines.currentIndex()]['energy']))

        self.caliblist_txt += "\n"+"{:d}".format(self.currentid)+"  -->  "+linename
        self.caliblist.setText(self.caliblist_txt)

    def calculate(self):
        # calculate new cte and gain
        if self.energies != []:
            params = np.polyfit(self.chnls, self.energies, 1) #1st degree linear fit
            self.newgain = params[0]
            self.newcte = params[1]
        # close spawned window and return selected elements...
        self.hide()
        if self.newcte is not None and self.newgain is not None:
            super().accept()
        else:
            super().reject()


class MatplotlibWidget(QWidget):
    
    def __init__(self, parent = None):
        
        QWidget.__init__(self, parent)
        
        self.fig = Figure(figsize=(7,3.5), dpi=100, tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(NavigationToolbar(self.canvas, self))
        vertical_layout.addWidget(self.canvas)
        
        # self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.axes = self.fig.add_subplot(111)
        
        self.setLayout(vertical_layout)


class Config_GUI(QWidget):
    
    def __init__(self, parent=None):
        super(Config_GUI, self).__init__(parent)

        # a default config dict to build on
        self.ConfigDict = ClassMcaTheory.McaTheory().configure(None)
        # some general adjustments we usually want to make to the configuration
        #   some of those listed here are default values, but just listed for convenience
        self.ConfigDict['fit']['use_limit'] = 1
        self.ConfigDict['fit']['sumflag'] = 1
        self.ConfigDict['fit']['escapeflag'] = 1
        self.ConfigDict['fit']['scatterflag'] = 1
        self.ConfigDict['fit']['energyscatter'] = [1]
        self.ConfigDict['fit']['energyflagg'] = [1]
        self.ConfigDict['fit']['energyweight'] = [1.0]
        if self.ConfigDict['fit']['energy'] is None:
            self.ConfigDict['fit']['energy'] = [20.]
        self.ConfigDict['attenuators']['Matrix'] = [0, 'MULTILAYER', 0.0, 0.0, 45.0, 45.0, 1, 90.0]

        self.ConfigDict['fit']['maxiter'] = 50
        self.ConfigDict['fit']['stripflag'] = 1 #0: no bkgr stripping, 1: bkgr stripping
        self.ConfigDict['fit']['stripalgorithm'] = 1 # 0:strip, 1:SNIP
        self.ConfigDict['fit']['continuum'] = 0 #0: no continuum, 4: linear polynomial, 5: exp. polynomial
        self.ConfigDict['fit']['fitweight'] = 1
        self.ConfigDict['fit']['fitfunction'] = 0
        # self.ConfigDict['fit']['linpolorder'] = 5 #linpol order
        # self.ConfigDict['fit']['deltaonepeak'] = 0.01
        # self.ConfigDict['fit']['exppolorder'] = 5 #exppol order
                   
            
        # as well as some other variables we'll need to store data etc.
        self.new_window = None
        self.filename = ""
        self.file = None #contains the file class, preventing continuously having to reread files when changing internal paths to display
        self.subdirs = []
        self.rawspe = None
        self.fitres = None
        
        
        # create widgets

        # create main layout for widgets
        layout_main = QVBoxLayout()
        layout_browseh5 = QHBoxLayout()
        layout_subdir = QHBoxLayout()
        layout_body = QHBoxLayout()
        layout_canvas = QVBoxLayout()
        layout_opts = QVBoxLayout()
        layout_fitopts = QVBoxLayout()
        layout_fitrange = QHBoxLayout()
        layout_raylE = QHBoxLayout()
        layout_scatangle = QHBoxLayout()
        layout_fitspecs = QHBoxLayout()
        layout_escsum = QHBoxLayout()
        layout_libopts = QVBoxLayout()
        layout_calibres = QHBoxLayout()
        layout_buts = QHBoxLayout()

        
        # browse buttons
        self.file_lbl = QLabel("File:")
        self.filedir = QLineEdit("")
        self.browse = QPushButton("...")
        self.browse.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.browse.setMaximumWidth(25)
        layout_browseh5.addWidget(self.file_lbl)
        layout_browseh5.addWidget(self.filedir)
        layout_browseh5.addWidget(self.browse)
        layout_main.addLayout(layout_browseh5)
        
        # dropdown box to select file subdir
        self.subdir_lbl = QLabel('     Sub directory:')
        self.subdir = QComboBox()
        self.subdir.addItems([''])
        self.subdir.setMinimumWidth(200)
        layout_subdir.addWidget(self.subdir_lbl)
        layout_subdir.addWidget(self.subdir)
        layout_subdir.addStretch()
        layout_main.addLayout(layout_subdir)
        
        # curves display
        self.mpl = MatplotlibWidget()
        self.mpl.axes.set_xlabel('Energy [keV]')
        self.mpl.axes.set_ylabel('Intensity [counts]')
        layout_canvas.addWidget(self.mpl)
        
        
        # calib results
        self.dettype = QComboBox()
        self.dettype.addItems(['Si', 'Ge'])
        self.cte_lbl = QLabel('zero:')
        self.cte = QLineEdit("{:.3f}".format(self.ConfigDict['detector']['zero']))
        self.cte.setMaximumWidth(50)
        self.cte.setValidator(QDoubleValidator(-1E6, 1E6,3))
        self.gain_lbl = QLabel('gain:')
        self.gain = QLineEdit("{:.3f}".format(self.ConfigDict['detector']['gain']))
        self.gain.setMaximumWidth(50)
        self.gain.setValidator(QDoubleValidator(-1E6, 1E6,3))
        self.fano_lbl = QLabel('fano:')
        self.fano = QLineEdit("{:.3f}".format(self.ConfigDict['detector']['fano']))
        self.fano.setMaximumWidth(50)
        self.fano.setValidator(QDoubleValidator(-1E6, 1E6,3))
        self.fano.setEnabled(False)
        self.noise_lbl = QLabel('noise:')
        self.noise = QLineEdit("{:.3f}".format(self.ConfigDict['detector']['noise']))
        self.noise.setMaximumWidth(50)
        self.noise.setValidator(QDoubleValidator(-1E6, 1E6,3))
        self.noise.setEnabled(False)
        layout_calibres.addWidget(self.dettype)
        layout_calibres.addWidget(self.cte_lbl)
        layout_calibres.addWidget(self.cte)
        layout_calibres.addWidget(self.gain_lbl)
        layout_calibres.addWidget(self.gain)
        layout_calibres.addWidget(self.fano_lbl)
        layout_calibres.addWidget(self.fano)
        layout_calibres.addWidget(self.noise_lbl)
        layout_calibres.addWidget(self.noise)
        layout_calibres.addStretch()
        layout_canvas.addLayout(layout_calibres)
        # general action buttons
        self.save = QPushButton("SAVE")
        self.save.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.save.setMaximumWidth(75)
        self.load = QPushButton("LOAD")
        self.load.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.load.setMaximumWidth(75)
        self.refit = QPushButton("REFIT")
        self.refit.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.refit.setMaximumWidth(75)
        self.calib = QPushButton("CALIB")
        self.calib.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.calib.setMaximumWidth(75)
        layout_buts.addWidget(self.save)
        layout_buts.addWidget(self.load)
        layout_buts.addWidget(self.refit)
        layout_buts.addWidget(self.calib)
        layout_canvas.addLayout(layout_buts)
        layout_body.addLayout(layout_canvas)


        
        # fitting options
        #   fit range
        #   fit scatterpeaks?
        #       rayleigh energy
        #       scatter angle
        #   strip bkgr?
        #       bkgr function, power
        #   fit pile-up( pile-up factor?), escape
        self.fitopts_lbl = QLabel("Fit options:")
        layout_fitopts.addWidget(self.fitopts_lbl)
        self.fitrange_lbl0 = QLabel("Fit range:")
        self.fitmin = QLineEdit("{:.3f}".format(self.ConfigDict['fit']['xmin']*self.ConfigDict['detector']['gain']+self.ConfigDict['detector']['zero']))
        self.fitmin.setMaximumWidth(50)
        self.fitmin.setValidator(QDoubleValidator(0, 1E6, 0))
        self.fitrange_lbl1 = QLabel(" - ")
        self.fitmax = QLineEdit("{:.3f}".format(self.ConfigDict['fit']['xmax']*self.ConfigDict['detector']['gain']+self.ConfigDict['detector']['zero']))
        self.fitmax.setMaximumWidth(50)
        self.fitmax.setValidator(QDoubleValidator(0, 1E6, 0))
        layout_fitrange.addWidget(self.fitrange_lbl0)
        layout_fitrange.addWidget(self.fitmin)
        layout_fitrange.addWidget(self.fitrange_lbl1)
        layout_fitrange.addWidget(self.fitmax)
        layout_fitrange.addStretch()
        layout_fitopts.addLayout(layout_fitrange)
        self.fit_scatter = QCheckBox("Fit scatter peaks")
        self.fit_scatter.setChecked(True)
        layout_fitopts.addWidget(self.fit_scatter)
        self.raylE_lbl = QLabel("     Rayleigh energy [keV]:")
        self.raylE = QLineEdit("{:.3f}".format(self.ConfigDict['fit']['energy'][0]))
        self.raylE.setMaximumWidth(50)
        self.raylE.setValidator(QDoubleValidator(0, 1E6, 3))
        layout_raylE.addWidget(self.raylE_lbl)
        layout_raylE.addWidget(self.raylE)
        layout_raylE.addStretch()
        layout_fitopts.addLayout(layout_raylE)
        self.scatangle_lbl = QLabel("     Scatter angle [Deg]:    ")
        self.scatangle = QLineEdit("{:.3f}".format(self.ConfigDict['attenuators']['Matrix'][7]))
        self.scatangle.setMaximumWidth(50)
        self.scatangle.setValidator(QDoubleValidator(-1E6, 1E6, 3))
        layout_scatangle.addWidget(self.scatangle_lbl)
        layout_scatangle.addWidget(self.scatangle)
        layout_scatangle.addStretch()
        layout_fitopts.addLayout(layout_scatangle)
        self.fit_bkgr = QCheckBox("Strip Background")
        self.fit_bkgr.setChecked(True)
        layout_fitopts.addWidget(self.fit_bkgr)
        self.fittype = QComboBox()
        self.fittype.addItems(['SNIP', 'Linear polynomial', 'Exp. polynomial'])
        self.fitpower = QLineEdit("{:d}".format(self.ConfigDict['fit']['snipwidth']))
        self.fitpower.setMaximumWidth(50)
        self.fitpower.setValidator(QDoubleValidator(-1E6, 1E6, 0))
        layout_fitspecs.addWidget(self.fittype)
        layout_fitspecs.addWidget(self.fitpower)
        layout_fitspecs.addStretch()
        layout_fitopts.addLayout(layout_fitspecs)
        self.fitsum = QCheckBox("Fit pile-up")
        self.fitsum.setChecked(True)
        self.fitesc = QCheckBox("Fit escape")
        self.fitesc.setChecked(True)
        layout_escsum.addWidget(self.fitsum)
        layout_escsum.addWidget(self.fitesc)
        layout_escsum.addStretch()
        layout_fitopts.addLayout(layout_escsum)
        layout_opts.addLayout(layout_fitopts)
        
        # library options
        # 3 tabs:
        #   peakID
        #   Fitresults
        #   Plottingopts
        self.whitespace = QLabel("")
        self.libopts_lbl = QLabel("Fitting:")
        layout_libopts.addWidget(self.whitespace)
        layout_libopts.addWidget(self.libopts_lbl)
        self.lib_tabs = QTabWidget()
        self.tab_peakid = QWidget() #peakid
        tab_peakid_layout = QVBoxLayout()
        tab_zselect_layout = QHBoxLayout()
        self.zselect_lbl = QLabel("Z:")
        self.zselect = QLineEdit("{:d}".format(26))
        self.zselect.setMaximumWidth(30)
        self.zselect.setValidator(QDoubleValidator(1, 110, 0))
        self.elselect_lbl = QLabel("el:")
        self.elselect = QLineEdit("Fe")
        self.elselect.setMaximumWidth(30)
        tab_zselect_layout.addWidget(self.zselect_lbl)
        tab_zselect_layout.addWidget(self.zselect)
        tab_zselect_layout.addWidget(self.elselect_lbl)
        tab_zselect_layout.addWidget(self.elselect)
        tab_zselect_layout.addStretch()
        tab_peakid_layout.addLayout(tab_zselect_layout)        
        tab_addline_layout = QHBoxLayout()
        self.addline_lbl = QLabel("line:")
        self.linefield = QLineEdit("")
        self.linefield.setMaximumWidth(50)
        self.line_add = QPushButton("Add")
        self.line_add.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.line_add.setMaximumWidth(30)
        self.line_rem = QPushButton("Remove")
        self.line_rem.setAutoDefault(False) # set False as otherwise this button is called on each return
        self.line_rem.setMaximumWidth(50)
        tab_addline_layout.addWidget(self.addline_lbl)
        tab_addline_layout.addWidget(self.linefield)
        tab_addline_layout.addWidget(self.line_add)
        tab_addline_layout.addWidget(self.line_rem)
        tab_addline_layout.addStretch()
        tab_peakid_layout.addLayout(tab_addline_layout)
        self.linetree = QTreeWidget()
        self.linetree.setColumnCount(1)
        self.linetree.setHeaderLabels(['Emission Lines'])
        self.linetree.setColumnWidth(0,50)
        self.adjust_linetree(self.linetree, self.elselect.text())
        tab_peakid_layout.addWidget(self.linetree)
        self.tab_peakid.setLayout(tab_peakid_layout)
        self.lib_tabs.addTab(self.tab_peakid, "Peak ID")        
        self.tab_fitres = QWidget() #fitresults
        tab_fitres_layout = QVBoxLayout()
        # scrollable library of added lines and integrated peak+bkgr intensity
        self.fittree = QTreeWidget()
        self.fittree.setColumnCount(1)
        self.fittree.setHeaderLabels(['Lines in fit\n-----\n          '+'{:10}'.format('Line')+'      Int.     Bkgr.'])
        self.fittree.setColumnWidth(0,50)
        elements = QTreeWidgetItem(self.fittree)
        elements.setText(0,'----')
        self.fittree.addTopLevelItem(elements)
        tab_fitres_layout.addWidget(self.fittree)
        self.tab_fitres.setLayout(tab_fitres_layout)        
        self.lib_tabs.addTab(self.tab_fitres, "Fit Output")
        self.tab_plotopts = QWidget() #plotopts
        tab_plotopts_layout = QVBoxLayout()
        self.plotopt_fit = QCheckBox("Display total fit curve")
        self.plotopt_fit.setChecked(True)
        self.plotopt_sum = QCheckBox("Display pileup fit curve")
        self.plotopt_sum.setChecked(True)
        self.plotopt_bkg = QCheckBox("Display fit background")
        self.plotopt_bkg.setChecked(True)
        tab_plotopts_layout.addWidget(self.plotopt_fit)
        tab_plotopts_layout.addWidget(self.plotopt_sum)
        tab_plotopts_layout.addWidget(self.plotopt_bkg)
        self.tab_plotopts.setLayout(tab_plotopts_layout)
        self.lib_tabs.addTab(self.tab_plotopts, "Plot Options")
        layout_libopts.addWidget(self.lib_tabs)        
        layout_opts.addLayout(layout_libopts)
        layout_body.addLayout(layout_opts)
        layout_main.addLayout(layout_body)
        self.lib_tabs.setCurrentWidget(self.tab_peakid)
        
        # show window
        self.setLayout(layout_main)
        self.setWindowTitle('XRF Config GUI')
        self.tab_peakid.setFocusPolicy(Qt.StrongFocus)
        self.show()
 
        # event handling
        self.browse.clicked.connect(self.browse_app) # browse button
        self.filedir.returnPressed.connect(self.browse_app)
        self.subdir.currentIndexChanged.connect(self.subdir_change) #different data directory selected
        self.cte.editingFinished.connect(self.update_ctegain) # cte changed
        self.gain.editingFinished.connect(self.update_ctegain) # gain changed
        self.calib.clicked.connect(self.calibrate) # initiate calibration
        self.zselect.returnPressed.connect(self.adjust_zselect) # KLM library Z changed
        self.elselect.returnPressed.connect(self.adjust_elselect) # KLM library el changed
        self.fitmin.returnPressed.connect(self.set_fitminmax) # selected new fitmin
        self.fitmax.returnPressed.connect(self.set_fitminmax) #s elected new fitmax
        self.fit_scatter.stateChanged.connect(self.fitscatter_params) # fit_scatter region (un)selected
        self.raylE.returnPressed.connect(self.fitscatter_params) # new rayleigh energy
        self.scatangle.returnPressed.connect(self.fitscatter_params) # new scatter angle
        self.dettype.currentIndexChanged.connect(self.fitscatter_params) # detector type
        self.fit_bkgr.stateChanged.connect(self.set_bkgr_sum_esc) # strip bkgr (un)selected
        self.fitsum.stateChanged.connect(self.set_bkgr_sum_esc) # fit pileup (un)selected
        self.fitesc.stateChanged.connect(self.set_bkgr_sum_esc) # fit escape (un)selected
        self.fittype.currentIndexChanged.connect(self.set_fittype) # different bkgr fit curve
        self.fitpower.returnPressed.connect(self.set_bkgr_sum_esc) # different fit power
        self.line_add.clicked.connect(self.add_line) # add line to peak library
        self.line_rem.clicked.connect(self.rem_line) # remove line from peak library
        self.linetree.itemClicked.connect(self.linetree_onClick) # Item clicked in peakid linetree
        self.linetree.itemDoubleClicked.connect(self.linetree_onDoubleClick) #item double clicked in peakid linetree
        self.save.clicked.connect(self.save_config) # save button
        self.load.clicked.connect(self.load_config) # load button
        self.refit.clicked.connect(self.do_refit) # refit button
        self.plotopt_fit.stateChanged.connect(self.update_plot) # plot options fit
        self.plotopt_sum.stateChanged.connect(self.update_plot) # plot options pileup
        self.plotopt_bkg.stateChanged.connect(self.update_plot) # plot options background
        

    def keyPressEvent(self, event): #handle arrow keys for KLM library
        if event.key() == Qt.Key_Up:
            newZ = int(self.zselect.text())+10
            if newZ > 110:
                newZ = newZ-110
            elif newZ == 110:
                newZ = 10
            self.zselect.setText("{:d}".format(newZ))
            self.elselect.setText(Elements.ElementList[newZ-1])
            #after key events, update the linetree and display KLM lines
            self.adjust_linetree(self.linetree, self.elselect.text()) # update the linetree
            self.update_plot(update=True) #update image
        elif event.key() == Qt.Key_Down:
            newZ = int(self.zselect.text())-10
            if newZ < 0:
                newZ = newZ+110
            elif newZ == 0:
                newZ = 100
            self.zselect.setText("{:d}".format(newZ))
            self.elselect.setText(Elements.ElementList[newZ-1])
            #after key events, update the linetree and display KLM lines
            self.adjust_linetree(self.linetree, self.elselect.text()) # update the linetree
            self.update_plot(update=True) #update image
        elif event.key() == Qt.Key_Left:
            newZ = int(self.zselect.text())-1
            if newZ < 1:
                newZ = len(Elements.ElementList)
            self.zselect.setText("{:d}".format(newZ))
            self.elselect.setText(Elements.ElementList[newZ-1])
            #after key events, update the linetree and display KLM lines
            self.adjust_linetree(self.linetree, self.elselect.text()) # update the linetree
            self.update_plot(update=True) #update image
        elif event.key() == Qt.Key_Right:
            newZ = int(self.zselect.text())+1
            if newZ > len(Elements.ElementList):
                newZ = 1
            self.zselect.setText("{:d}".format(newZ))
            self.elselect.setText(Elements.ElementList[newZ-1])
            #after key events, update the linetree and display KLM lines
            self.adjust_linetree(self.linetree, self.elselect.text()) # update the linetree
            self.update_plot(update=True) #update image
        else:
            QWidget.keyPressEvent(self, event)

    def adjust_linetree(self, obj, element):
        linedict = compile_pymca_dict(element)
        obj.clear()
        if [line for line in linedict if 'K' in line['linegroup']] != []:
            Klines = QTreeWidgetItem(obj)
            Klines.setText(0,'K lines')
            Klines_Ka = QTreeWidgetItem(Klines)
            Klines_Ka.setText(0, 'Ka')
            Klines_Ka_sub = []
            Klines_Kb = QTreeWidgetItem(Klines)
            Klines_Kb.setText(0, 'Kb')
            Klines_Kb_sub = []
        if [line for line in linedict if 'L' in line['linegroup']] != []:
            Llines = QTreeWidgetItem(obj)
            Llines.setText(0,'L lines')
            Llines_L1 = QTreeWidgetItem(Llines)
            Llines_L1.setText(0, 'L1')
            Llines_L1_sub = []
            Llines_L2 = QTreeWidgetItem(Llines)
            Llines_L2.setText(0, 'L2')
            Llines_L2_sub = []
            Llines_L3 = QTreeWidgetItem(Llines)
            Llines_L3.setText(0, 'L3')
            Llines_L3_sub = []
        if [line for line in linedict if 'M' in line['linegroup']] != []:
            Mlines = QTreeWidgetItem(obj)
            Mlines.setText(0,'M lines')
            Mlines_sub = []          
        for line in linedict:
            text = line['line']+"    "+"{:.3f}".format(line['energy'])+"keV ("+"{:.5f}".format(line['rate'])+")"
            if line['linegroup'] == 'Ka':
                Klines_Ka_sub.append(QTreeWidgetItem(Klines_Ka))
                Klines_Ka_sub[-1].setText(0, text)
            elif line['linegroup'] == 'Kb':
                Klines_Kb_sub.append(QTreeWidgetItem(Klines_Kb))
                Klines_Kb_sub[-1].setText(0, text)
            elif line['linegroup'] == 'L1':
                Llines_L1_sub.append(QTreeWidgetItem(Llines_L1))
                Llines_L1_sub[-1].setText(0, text)
            elif line['linegroup'] == 'L2':
                Llines_L2_sub.append(QTreeWidgetItem(Llines_L2))
                Llines_L2_sub[-1].setText(0, text)
            elif line['linegroup'] == 'L3':
                Llines_L3_sub.append(QTreeWidgetItem(Llines_L3))
                Llines_L3_sub[-1].setText(0, text)
            elif 'M' in line['linegroup']:
                Mlines_sub.append(QTreeWidgetItem(Mlines))
                Mlines_sub[-1].setText(0, text)            

    def adjust_fittree(self, obj):
        obj.clear()
        if self.ConfigDict['peaks'] != {}:
            for elements in self.ConfigDict['peaks']:
                element = QTreeWidgetItem(obj)
                element.setText(0, elements)
                element_sub = []
                for lines in self.ConfigDict['peaks'][elements]:
                    if self.fitres is None:
                        peakint = '----'
                        bkgrint = '----'
                    else:
                        peakint = "{:.3f}".format(self.fitres['linesarea'][self.fitres['lines'].index(elements+' '+lines)])
                        bkgrint = "{:.3f}".format(self.fitres['linesbkgr'][self.fitres['lines'].index(elements+' '+lines)])
                    text = '{:10}'.format(lines)+'     '+peakint+'     '+bkgrint
                    element_sub.append(QTreeWidgetItem(element))
                    element_sub[-1].setText(0, text)
        else:
            elements = QTreeWidgetItem(obj)
            elements.setText(0,'----')            
        # add Compton and Rayl if option is checked
        if self.fit_scatter.isChecked():
            element = QTreeWidgetItem(obj)
            element.setText(0, 'Scatter')
            if self.fitres is None:
                peakint = '----'
                bkgrint = '----'
            else:
                peakint = "{:.3f}".format(self.fitres['linesarea'][self.fitres['lines'].index('Rayl')])
                bkgrint = "{:.3f}".format(self.fitres['linesbkgr'][self.fitres['lines'].index('Rayl')])
            rayltext = '{:10}'.format('Rayl')+'    '+peakint+'     '+bkgrint
            if self.fitres is None:
                peakint = '----'
                bkgrint = '----'
            else:
                peakint = "{:.3f}".format(self.fitres['linesarea'][self.fitres['lines'].index('Compt')])
                bkgrint = "{:.3f}".format(self.fitres['linesbkgr'][self.fitres['lines'].index('Compt')])
            comptext = '{:10}'.format('Compt')+peakint+'     '+bkgrint
            element_sub = QTreeWidgetItem(element)
            element_sub.setText(0, comptext)
            element_sub = QTreeWidgetItem(element)
            element_sub.setText(0, rayltext)

    def update_plot(self, update=True):
        # plot default spectrum
        if self.rawspe is not None:
            xdata = np.arange(len(self.rawspe))*self.ConfigDict['detector']['gain']+self.ConfigDict['detector']['zero']
            if update is False:
                self.mpl.axes.cla()
                self.mpl.axes.plot(xdata, self.rawspe, color='black', label='rawspe')
                self.mpl.axes.set_xlabel('Energy [keV]')
                self.mpl.axes.set_ylabel('Intensity [counts]')
                self.mpl.axes.set_yscale('log')
                self.mpl.axes.set_xlim((np.min(xdata),np.max(xdata)))
                self.mpl.axes.set_ymargin(0.05)

            if update is not False:
                for i in reversed(range(len(self.mpl.axes.lines))):
                    if self.mpl.axes.lines[i].get_label() in ['fit','bkg','sum']:
                        self.mpl.axes.lines[i].remove()
                if self.fitres is not None:
                    if self.plotopt_bkg.isChecked():
                        self.mpl.axes.plot(self.fitres['energy'], self.fitres['bkgr'], color='green', linewidth=1, label='bkg')
                    if self.plotopt_fit.isChecked():
                        self.mpl.axes.plot(self.fitres['energy'], self.fitres['yfit'], color='red', linewidth=1, label='fit')
                    if self.plotopt_sum.isChecked():
                        self.mpl.axes.plot(self.fitres['energy'], self.fitres['pileup'], color='magenta', linewidth=1, label='sum')
            
            if update is not False:
                linedict = compile_pymca_dict(self.elselect.text())
                while self.mpl.axes.collections:
                    self.mpl.axes.collections[-1].remove()
                if [line for line in linedict if 'K' in line['linegroup']] != []:
                    energies = [line['energy'] for line in linedict if 'K' in line['linegroup']]
                    ratios = [line['rate'] for line in linedict if 'K' in line['linegroup']]
                    # check if lines are in spectral range
                    xlines = [energy for energy in energies if energy >= np.min(xdata) and energy <= np.max(xdata)]
                    if xlines != []:
                        ratios = [ratios[energies.index(line)] for line in xlines]
                        maxrateid = ratios.index(np.max(ratios))
                        # find spectrum channel corresponding to most intense line energy
                        maxint = self.rawspe[np.max(np.where(xdata <= xlines[maxrateid]))]
                        ymin = [np.min(self.rawspe) - (np.max(self.rawspe)-np.min(self.rawspe))*0.05]*len(xlines)
                        ymax = np.array(ratios)/ratios[maxrateid]*maxint 
                        self.mpl.axes.vlines(xlines, ymin, ymax, colors='blue')
                if [line for line in linedict if 'L' in line['linegroup']] != []:
                    energies = [line['energy'] for line in linedict if 'L' in line['linegroup']]
                    ratios = [line['rate'] for line in linedict if 'L' in line['linegroup']]
                    # check if lines are in spectral range
                    xlines = [energy for energy in energies if energy >= np.min(xdata) and energy <= np.max(xdata)]
                    if xlines != []:
                        ratios = [ratios[energies.index(line)] for line in xlines]
                        maxrateid = ratios.index(np.max(ratios))
                        # find spectrum channel corresponding to most intense line energy
                        maxint = self.rawspe[np.max(np.where(xdata <= xlines[maxrateid]))]
                        ymin = [np.min(self.rawspe) - (np.max(self.rawspe)-np.min(self.rawspe))*0.05]*len(xlines)
                        ymax = np.array(ratios)/ratios[maxrateid]*maxint 
                        self.mpl.axes.vlines(xlines, ymin, ymax, colors='blue')
                if [line for line in linedict if 'M' in line['linegroup']] != []:
                    energies = [line['energy'] for line in linedict if 'M' in line['linegroup']]
                    ratios = [line['rate'] for line in linedict if 'M' in line['linegroup']]
                    # check if lines are in spectral range
                    xlines = [energy for energy in energies if energy >= np.min(xdata) and energy <= np.max(xdata)]
                    if xlines != []:
                        ratios = [ratios[energies.index(line)] for line in xlines]
                        maxrateid = ratios.index(np.max(ratios))
                        # find spectrum channel corresponding to most intense line energy
                        maxint = self.rawspe[np.max(np.where(xdata <= xlines[maxrateid]))]
                        ymin = [np.min(self.rawspe) - (np.max(self.rawspe)-np.min(self.rawspe))*0.05]*len(xlines)
                        ymax = np.array(ratios)/ratios[maxrateid]*maxint 
                        self.mpl.axes.vlines(xlines, ymin, ymax, colors='blue')
            self.mpl.canvas.draw()


    def browse_app(self):
        self.filename = QFileDialog.getOpenFileName(self, caption="Open spectrum file", filter="H5 (*.h5);;SPE (*.spe);;CSV (*.csv)")[0]
        self.filedir.setText("'"+str(self.filename)+"'")
        # read in first ims file, to obtain data on elements and dimensions
        if(self.filename != "''"):
            if self.filename.split('.')[-1] == 'spe':
                pass #TODO
            elif self.filename.split('.')[-1] == 'csv':
                pass #TODO
            elif self.filename.split('.')[-1] == 'h5':
                self.file = Poll_h5dir(self.filename)
                self.subdirs = []
                self.subdirs = self.file.dirs()
                self.rawspe = self.file.spe([dirs for dirs in self.subdirs if 'raw' in dirs and 'sumspec' in dirs][0])
                # change dropdown widget to display appropriate subdirs
                self.subdir.clear()
                self.subdir.addItems(self.subdirs)
                self.subdir.setCurrentIndex(self.subdirs.index([dirs for dirs in self.subdirs if 'raw' in dirs and 'sumspec' in dirs][0]))
        # now adjust plot window (if new file or dir chosen, the fit results should clear and only self.rawspe is displayed)
        self.fitres = None
        self.update_plot(update=False)
        self.adjust_fittree(self.fittree)

    def subdir_change(self, index):
        if self.subdirs != []:
            self.rawspe = self.file.spe(self.subdirs[index])
            self.fitres = None
            self.update_plot(update=False)
            self.adjust_fittree(self.fittree)

    def update_ctegain(self):
        self.ConfigDict['detector']['gain'] = float(self.gain.text())
        self.ConfigDict['detector']['zero'] = float(self.cte.text())
        self.fitmin.setText("{:.3f}".format(self.ConfigDict['fit']['xmin']*self.ConfigDict['detector']['gain']+self.ConfigDict['detector']['zero']))
        self.fitmax.setText("{:.3f}".format(self.ConfigDict['fit']['xmax']*self.ConfigDict['detector']['gain']+self.ConfigDict['detector']['zero']))
        self.fitres = None
        self.update_plot(update=False)

    def adjust_zselect(self):
        newZ = int(self.zselect.text())
        self.elselect.setText(Elements.ElementList[newZ-1])
        #after key events, update the linetree and display KLM lines
        self.adjust_linetree(self.linetree, self.elselect.text()) # update the linetree
        self.update_plot(update=True) #update image

    def adjust_elselect(self):
        if Elements.getz(self.elselect.text()) is not None:
            newZ = Elements.getz(self.elselect.text())
            self.zselect.setText("{:d}".format(newZ))
            #after key events, update the linetree and display KLM lines
            self.adjust_linetree(self.linetree, self.elselect.text()) # update the linetree
            self.update_plot(update=True) #update image
        else:
            self.elselect.setText('Fe') #if a non-existing element was set, just default to Fe
            self.adjust_elselect()

    def set_fitminmax(self):
        self.ConfigDict['fit']['xmin'] = np.round((float(self.fitmin.text())-self.ConfigDict['detector']['zero'])/self.ConfigDict['detector']['gain'])
        self.ConfigDict['fit']['xmax'] = np.round((float(self.fitmax.text())-self.ConfigDict['detector']['zero'])/self.ConfigDict['detector']['gain'])
        
    def fitscatter_params(self):
        if self.fit_scatter.isChecked():
            self.ConfigDict['fit']['scatterflag'] = 1
            self.ConfigDict['fit']['energyscatter'] = [1]
            self.ConfigDict['fit']['energyflag'] = [1]
            self.ConfigDict['fit']['energyweight'] = [1.0]
            self.ConfigDict['fit']['energy'] = [float(self.raylE.text())]
            self.ConfigDict['attenuators']['Matrix'] = [0, 'MULTILAYER', 0.0, 0.0, float(self.scatangle.text())/2, float(self.scatangle.text())/2, 1, float(self.scatangle.text())]
        else:
            self.ConfigDict['fit']['scatterflag'] = 0
            self.ConfigDict['fit']['energyscatter'] = [0]
            self.ConfigDict['fit']['energyflag'] = [0]
            self.ConfigDict['fit']['energyweight'] = [1.0]
            self.ConfigDict['fit']['energy'] = [float(self.raylE.text())]
            self.ConfigDict['attenuators']['Matrix'] = [0, 'MULTILAYER', 0.0, 0.0, float(self.scatangle.text())/2, float(self.scatangle.text())/2, 1, float(self.scatangle.text())]

        self.ConfigDict['detector']['detele'] = self.dettype.currentText() #Si or Ge; 'detene' tab appears not used directly...

    def set_bkgr_sum_esc(self):
        if self.fit_bkgr.isChecked():
            self.ConfigDict['fit']['stripflag'] = 1 #0: no bkgr stripping, 1: bkgr stripping
            self.ConfigDict['fit']['stripalgorithm'] = 1 # 0:strip, 1:SNIP
            self.ConfigDict['fit']['fitweight'] = 1
            self.ConfigDict['fit']['fitfunction'] = 0
        else:
            self.ConfigDict['fit']['stripflag'] = 0 #0: no bkgr stripping, 1: bkgr stripping
            self.ConfigDict['fit']['stripalgorithm'] = 1 # 0:strip, 1:SNIP
            self.ConfigDict['fit']['fitweight'] = 1
            self.ConfigDict['fit']['fitfunction'] = 0
            
        if self.fittype.currentText() == 'SNIP':
            self.ConfigDict['fit']['maxiter'] = 50
            self.ConfigDict['fit']['stripalgorithm'] = 1 # 0:strip, 1:SNIP
            self.ConfigDict['fit']['continuum'] = 0 #0: no continuum, 4: linear polynomial, 5: exp. polynomial
            self.ConfigDict['fit']['snipwidth'] = int(self.fitpower.text()) #snipwidth
            self.ConfigDict['fit']['deltaonepeak'] = 0.01
        elif self.fittype.currentText() == 'Linear polynomial':
            self.ConfigDict['fit']['maxiter'] = 50
            self.ConfigDict['fit']['stripalgorithm'] = 0 # 0:strip, 1:SNIP
            self.ConfigDict['fit']['continuum'] = 4 #0: no continuum, 4: linear polynomial, 5: exp. polynomial
            self.ConfigDict['fit']['linpolorder'] = int(self.fitpower.text()) #linpol order
            self.ConfigDict['fit']['deltaonepeak'] = 0.01
        elif self.fittype.currentText() == 'Exp. polynomial':
            self.ConfigDict['fit']['maxiter'] = 50
            self.ConfigDict['fit']['stripalgorithm'] = 0 # 0:strip, 1:SNIP
            self.ConfigDict['fit']['continuum'] = 5 #0: no continuum, 4: linear polynomial, 5: exp. polynomial
            self.ConfigDict['fit']['exppolorder'] = int(self.fitpower.text()) #exppol order
            self.ConfigDict['fit']['deltaonepeak'] = 0.01
            
        if self.fitesc.isChecked():
            self.ConfigDict['fit']['escapeflag'] = 1
        else:
            self.ConfigDict['fit']['escapeflag'] = 0
            
        if self.fitsum.isChecked():
            self.ConfigDict['fit']['sumflag'] = 1
        else:
            self.ConfigDict['fit']['sumflag'] = 0
            
    def set_fittype(self):
        if self.fittype.currentText() == 'SNIP':
            self.ConfigDict['fit']['maxiter'] = 50
            self.ConfigDict['fit']['stripalgorithm'] = 1 # 0:strip, 1:SNIP
            self.ConfigDict['fit']['continuum'] = 0 #0: no continuum, 4: linear polynomial, 5: exp. polynomial
            self.ConfigDict['fit']['snipwidth'] = 30
            self.fitpower.setText("{:d}".format(self.ConfigDict['fit']['snipwidth'])) #snipwidth
            self.ConfigDict['fit']['deltaonepeak'] = 0.01
        elif self.fittype.currentText() == 'Linear polynomial':
            self.ConfigDict['fit']['maxiter'] = 50
            self.ConfigDict['fit']['stripalgorithm'] = 0 # 0:strip, 1:SNIP
            self.ConfigDict['fit']['continuum'] = 4 #0: no continuum, 4: linear polynomial, 5: exp. polynomial
            # self.ConfigDict['fit']['linpolorder'] = 5
            self.fitpower.setText("{:d}".format(self.ConfigDict['fit']['linpolorder'])) #linpol order
            self.ConfigDict['fit']['deltaonepeak'] = 0.01
        elif self.fittype.currentText() == 'Exp. polynomial':
            self.ConfigDict['fit']['maxiter'] = 50
            self.ConfigDict['fit']['stripalgorithm'] = 0 # 0:strip, 1:SNIP
            self.ConfigDict['fit']['continuum'] = 5 #0: no continuum, 4: linear polynomial, 5: exp. polynomial
            # self.ConfigDict['fit']['exppolorder'] = 6
            self.fitpower.setText("{:d}".format(self.ConfigDict['fit']['exppolorder'])) #exppol order
            self.ConfigDict['fit']['deltaonepeak'] = 0.01

    def linetree_onClick(self, item, column):
        it = item.text(0).split(' ')[0]
        # recognise selected line type
        element = self.elselect.text()
        if len(it) == 1: # K, L or M
            if 'K' in it:    
                line = 'K'
            elif 'L' in it:
                line = 'L'
            elif 'M' in it:
                line = 'M'
            else:
                print('///'+it) #should not occur
        elif len(it) == 2: # Ka, Kb, L1, L2 or L3
            if 'a' in it:    
                line = 'Ka'
            elif 'b' in it:
                line = 'Kb'
            elif it == 'L1':
                line = 'L1'
            elif it == 'L2':
                line = 'L2'
            elif it == 'L3':
                line = 'L3'
            elif it[0] == 'K':
                line = 'Kb'
            elif it[0] == 'K':
                line = 'Kb'
            else:
                print('---'+it) #should be an M line then, but would be strange to pop up here...
        else: # the separate lines were selected
            if it[0] == 'K' and it[1] == 'L':
                line = 'Ka'
            elif it[0] == 'K' and it[1] != 'L':
                line = 'Kb'
            elif it[0:2] == 'L1':
                line = 'L1'
            elif it[0:2] == 'L2':
                line = 'L2'
            elif it[0:2] == 'L3':
                line = 'L3'
            elif it[0] == 'M':
                line = 'M'
            else:
                print('***'+it) #should not occur           
        # set lineedit field
        text = element+'-'+line
        self.linefield.setText(text)
        
    def linetree_onDoubleClick(self, item, column):
        self.add_line()

    def add_line(self):
        edit = self.linefield.text()
        keys = list(self.ConfigDict['peaks'].keys())
        # first split for each ; in case we added multiple lines simultaneously
        edit = edit.split(';')
        # now go through all lines, check if they are in model; if not: add; if are: check if line was added...
        for ed in edit:
            if '-' not in ed: #add all K lines from this element
                element = ed.replace(' ','')
                if element in keys:
                    if 'K' not in self.ConfigDict['peaks'][element]:
                        self.ConfigDict['peaks'][element].append('K')
                else:
                    self.ConfigDict['peaks'][element] = 'K'
            else:
                ed = ed.split('-')
                element = ed[0]
                line = ed[1].replace('[','').replace(']','').replace(' ','') #could be this is a list, e.g. '[K, L]'
                lines = line.split(',')
                if element in keys:
                    for i in lines:
                        if i not in self.ConfigDict['peaks'][element]:
                            self.ConfigDict['peaks'][element] = list(self.ConfigDict['peaks'][element])
                            self.ConfigDict['peaks'][element].append(i)
                else:
                    if type(lines) is list and len(lines) == 1:
                        self.ConfigDict['peaks'][element] = lines[0]
                    else:
                        self.ConfigDict['peaks'][element] = lines
        # check self.ConfigDict['peaks'] health...
        keys = list(self.ConfigDict['peaks'].keys())
        for key in keys:
            # lines = self.ConfigDict['peaks'][key]
            if type(self.ConfigDict['peaks'][key]) is list:
                if 'K' in self.ConfigDict['peaks'][key]:
                    if 'Ka' in self.ConfigDict['peaks'][key]:
                        del(self.ConfigDict['peaks'][key][self.ConfigDict['peaks'][key].index('Ka')])
                    if 'Kb' in lines:
                        del(self.ConfigDict['peaks'][key][self.ConfigDict['peaks'][key].index('Kb')])
                if 'L' in self.ConfigDict['peaks'][key]:
                    if 'L1' in self.ConfigDict['peaks'][key]:
                        del(self.ConfigDict['peaks'][key][self.ConfigDict['peaks'][key].index('L1')])
                    if 'L2' in self.ConfigDict['peaks'][key]:
                        del(self.ConfigDict['peaks'][key][self.ConfigDict['peaks'][key].index('L2')])
                    if 'L3' in self.ConfigDict['peaks'][key]:
                        del(self.ConfigDict['peaks'][key][self.ConfigDict['peaks'][key].index('L3')])
                # self.ConfigDict['peaks'][key] = lines
            if self.ConfigDict['peaks'][key] == []:
                del(self.ConfigDict['peaks'][key])
        # order the library in order of increasing atomic number...
        sortedpeaks = {}
        for el in sorted(self.ConfigDict['peaks'], key=Elements.getz):
            if type(self.ConfigDict['peaks'][el]) is list:
                sortedpeaks[el] = sorted(self.ConfigDict['peaks'][el])
            else:
                sortedpeaks[el] = self.ConfigDict['peaks'][el]
        self.ConfigDict['peaks'] = sortedpeaks
        self.fitres = None
        self.adjust_fittree(self.fittree)
    
    def rem_line(self):
        import re
        edit = self.linefield.text()
        keys = list(self.ConfigDict['peaks'].keys())
        # first split for each ; in case we added multiple lines simultaneously
        edit = edit.split(';')
        # now go through all lines, check if they are in model
        for ed in edit:
            if ed == 'all': # remove all lines
                for key in keys:
                    del(self.ConfigDict['peaks'][key])
                del(self.ConfigDict['peaks'])
                self.ConfigDict['peaks'] = {}
            elif '-' not in ed: #remove all lines from this element
                element = ed.replace(' ','')
                if element in keys:
                    del(self.ConfigDict['peaks'][element])
            else:
                ed = ed.split('-')
                element = ed[0]
                line = ed[1].replace('[','').replace(']','').replace(' ','') #could be this is a list, e.g. '[K, L]'
                lines = line.split(',')
                if element in keys:
                    for i in lines:
                        matches = list(filter(re.compile("\A"+i).match, self.ConfigDict['peaks'][element]))
                        if matches != [] :
                            for m in matches:
                                if type(self.ConfigDict['peaks'][element]) is str:
                                    del(self.ConfigDict['peaks'][element])
                                else:
                                    del(self.ConfigDict['peaks'][element][self.ConfigDict['peaks'][element].index(m)])
                else: #element not in ConfigDict, so nothing to remove...
                    pass
        # check self.ConfigDict['peaks'] health...
        keys = list(self.ConfigDict['peaks'].keys())
        for key in keys:
            if type(self.ConfigDict['peaks'][key]) is list:
                if 'K' in self.ConfigDict['peaks'][key]:
                    if 'Ka' in self.ConfigDict['peaks'][key]:
                        del(self.ConfigDict['peaks'][key][self.ConfigDict['peaks'][key].index('Ka')])
                    if 'Kb' in lines:
                        del(self.ConfigDict['peaks'][key][self.ConfigDict['peaks'][key].index('Kb')])
                if 'L' in self.ConfigDict['peaks'][key]:
                    if 'L1' in self.ConfigDict['peaks'][key]:
                        del(self.ConfigDict['peaks'][key][self.ConfigDict['peaks'][key].index('L1')])
                    if 'L2' in self.ConfigDict['peaks'][key]:
                        del(self.ConfigDict['peaks'][key][self.ConfigDict['peaks'][key].index('L2')])
                    if 'L3' in self.ConfigDict['peaks'][key]:
                        del(self.ConfigDict['peaks'][key][self.ConfigDict['peaks'][key].index('L3')])
            if self.ConfigDict['peaks'][key] == []:
                del(self.ConfigDict['peaks'][key])
        self.fitres = None
        self.adjust_fittree(self.fittree)

    def save_config(self):
        filename = QFileDialog.getSaveFileName(self, caption="Save config in:", filter="CFG (*.cfg)")[0]
        self.ConfigDict.write(filename)

    def load_config(self):
        #TODO: set Ge or Si button...
        filename = QFileDialog.getOpenFileName(self, caption="Save config in:", filter="CFG (*.cfg)")[0]
        self.ConfigDict.read(filename)
        # go over ConfigDict and adjust GUI accordingly
        if self.ConfigDict['fit']['scatterflag'] == 1:
            self.fit_scatter.setChecked(True)
            self.raylE.setText("{:.3f}".format(self.ConfigDict['fit']['energy']))
            if len(self.ConfigDict['attenuators']['Matrix']) == 8:
                self.scatangle.setText("{:.3f}".format(self.ConfigDict['attenuators']['Matrix'][7]))
            else:
                self.scatangle.setText("{:.3f}".format(self.ConfigDict['attenuators']['Matrix'][4]+self.ConfigDict['attenuators']['Matrix'][5]))
                self.ConfigDict['attenuators']['Matrix'].append(1)
                self.ConfigDict['attenuators']['Matrix'].append(self.ConfigDict['attenuators']['Matrix'][4]+self.ConfigDict['attenuators']['Matrix'][5])
        else:
            self.fit_scatter.setChecked(False)
            if self.ConfigDict['fit']['energy'][:] is not None:
                self.raylE.setText("{:.3f}".format(self.ConfigDict['fit']['energy']))
            else:
                self.raylE.setText("{:.3f}".format(20.))
                self.ConfigDict['fit']['energy'] = [20.]
            if len(self.ConfigDict['attenuators']['Matrix']) == 8:
                self.scatangle.setText("{:.3f}".format(self.ConfigDict['attenuators']['Matrix'][7]))
            else:
                self.scatangle.setText("{:.3f}".format(self.ConfigDict['attenuators']['Matrix'][4]+self.ConfigDict['attenuators']['Matrix'][5]))
                self.ConfigDict['attenuators']['Matrix'].append(1)
                self.ConfigDict['attenuators']['Matrix'].append(self.ConfigDict['attenuators']['Matrix'][4]+self.ConfigDict['attenuators']['Matrix'][5])
        if self.ConfigDict['fit']['escapeflag'] == 1:
            self.fitesc.setChecked(True)
        else:
            self.fitesc.setChecked(False)
        if self.ConfigDict['fit']['sumflag'] == 1:
            self.fitsum.setChecked(True)
        else:
            self.fitsum.setChecked(False)
        if self.ConfigDict['fit']['stripflag'] == 0:
            self.fit_bkgr.setChecked(False)
        else:
            self.fit_bkgr.setChecked(True)
        if self.ConfigDict['fit']['continuum'] == 0:
            self.fittype.setCurrentText('SNIP')
            self.fitpower.setText("{:d}".format(self.ConfigDict['fit']['snipwidth']))
        elif self.ConfigDict['fit']['continuum'] == 4:
            self.fittype.setCurrentText('Linear polynomial')
            self.fitpower.setText("{:d}".format(self.ConfigDict['fit']['linpolorder']))
        elif self.ConfigDict['fit']['continuum'] == 5:
            self.fittype.setCurrentText('Exp. polynomial')
            self.fitpower.setText("{:d}".format(self.ConfigDict['fit']['exppolorder']))
        self.gain.setText("{:.3f}".format(self.ConfigDict['detector']['gain']))
        self.cte.setText("{:.3f}".format(self.ConfigDict['detector']['zero']))
        self.fitmin.setText("{:.3f}".format(self.ConfigDict['fit']['xmin']*self.ConfigDict['detector']['gain']+self.ConfigDict['detector']['zero']))
        self.fitmax.setText("{:.3f}".format(self.ConfigDict['fit']['xmax']*self.ConfigDict['detector']['gain']+self.ConfigDict['detector']['zero']))
        self.dettype.setCurrentText(self.ConfigDict['detector']['detele']) #Si or Ge; 'detene' tab appears not used directly...
        if self.ConfigDict['peaks'] != {}:
            # order the library in order of increasing atomic number...
            sortedpeaks = {}
            for el in sorted(self.ConfigDict['peaks'], key=Elements.getz):
                if type(self.ConfigDict['peaks'][el]) is list:
                    sortedpeaks[el] = sorted(self.ConfigDict['peaks'][el])
                else:
                    sortedpeaks[el] = self.ConfigDict['peaks'][el]
            self.ConfigDict['peaks'] = sortedpeaks
        # self.update_plot()
        self.update_plot(update=False) #redraw the original plot, with KLM lines
        self.fitres = None
        self.adjust_fittree(self.fittree)

    def do_refit(self):                    
        mcafit = ClassMcaTheory.ClassMcaTheory()
        mcafit.configure(self.ConfigDict)
        mcafit.setData(range(0,len(self.rawspe)), self.rawspe)
        mcafit.estimate()
        fitresult, result = mcafit.startfit(digest=1)
        names = [n.replace('Scatter Peak000', 'Rayl') for n in result["groups"]]
        names = [n.replace('Scatter Compton000', 'Compt') for n in names]
        fitarea = [result[peak]["fitarea"] for peak in result["groups"]]
        bkgr = [result[peak]["statistics"]-result[peak]["fitarea"] for peak in result["groups"]]

        self.fitres = {
            'energy':result['energy'],
            'yfit':result['yfit'],
            'bkgr':result['continuum'],
            'pileup':result['continuum']+result['pileup'],
            'lines':names,
            'linesarea':fitarea,
            'linesbkgr':bkgr
            }
        # self.update_plot()
        self.update_plot(update=True) #redraw the original plot, with KLM lines
        self.adjust_fittree(self.fittree)
        
    def calibrate(self):
        if self.rawspe is not None and self.new_window is None: #must have a spectrum loaded...
            self.new_window = CalibrateWindow(self)
            self.new_window.setFocus()
            if self.new_window.exec_() == QDialog.Accepted:
                self.ConfigDict['detector']['gain'] = float(self.new_window.newgain)
                self.ConfigDict['detector']['zero'] = float(self.new_window.newcte)
                self.gain.setText("{:.3f}".format(self.ConfigDict['detector']['gain']))
                self.cte.setText("{:.3f}".format(self.ConfigDict['detector']['zero']))
                self.fitmin.setText("{:.3f}".format(self.ConfigDict['fit']['xmin']*self.ConfigDict['detector']['gain']+self.ConfigDict['detector']['zero']))
                self.fitmax.setText("{:.3f}".format(self.ConfigDict['fit']['xmax']*self.ConfigDict['detector']['gain']+self.ConfigDict['detector']['zero']))
            self.new_window.close()
            self.new_window = None
            self.fitres = None
            self.update_plot(update=False) #redraw the original plot, without calib line



if __name__ == "__main__":
    app = QApplication(sys.argv)
    xrf_config = Config_GUI()
    xrf_config.show()
    sys.exit(app.exec_())