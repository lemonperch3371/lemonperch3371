import sys
from pathlib import Path
import exdir

# import cupy as np
import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from pyqtgraph.Qt import QtWidgets
from pathlib import Path
import tiffile
import datetime
import matplotlib.pyplot as plt
# try:
# 	import cupy as cp
# 	from _interpnd import LinearNDInterpolator as LinearNDInterpolator_cp
# except Exception as ex:
# 	print(ex)
import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator

from scipy.interpolate import interp1d, NearestNDInterpolator, LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.ndimage import median_filter, maximum_filter, minimum_filter, gaussian_filter
from scipy.optimize import curve_fit
from scipy.constants import speed_of_light

sys.path.append(str(Path("F:/20241113/")))
from data_utils import import_images, make_dashboard


import pandas as pd
from matplotlib.colors import hsv_to_rgb


# calibr_curve = pd.read_csv("Lamp_mirrorx2_lens_spectrometer_grating_300Lpmm.csv", index_col=0)
# FESH0650 = pd.read_csv("FESH0650.csv", index_col=0)
# FESH0650.loc[FESH0650["transmission"]<0.01, "transmission"]=np.nan
# FESH0650 = interp1d(FESH0650.index,FESH0650.transmission, bounds_error=False,)(calibr_curve.index)
# calibr_curve = interp1d(calibr_curve.index, calibr_curve.calibr_curve/FESH0650, bounds_error=False,fill_value=np.nan)




exdir_path = sys.argv[0].replace(".py",".exdir")

app = pg.mkQApp()
win = QtWidgets.QMainWindow()
area = DockArea()
win.setCentralWidget(area)


DASHBOARD = {}

colormap = pg.colormap.get("jet", source='matplotlib')
myLUT = colormap.getLookupTable()

colors = [
            (0, 0, 0, 0),
            (4, 5, 61, 100),
            (50, 42, 220, 255),
            (236, 87, 134, 255),
            (246, 246, 0, 255),
            (255, 255, 255, 255)
        ]

        # color map
cmap = pg.ColorMap(pos=np.array([0, 0.1, 0.2, 0.5, 0.8, 1]), color=colors)

DASHBOARD[f"Mosaic"] = {"type": "mosaic", "dock_pos": ("left",)}
DASHBOARD[f"Mosaic_diff"] = {"type": "mosaic", "dock_pos": ("below","Mosaic")}
DASHBOARD["Raw"] = {"type": "raw", "dock_pos": ("bottom", )}



DOCKS = make_dashboard(DASHBOARD, area)



curve = DASHBOARD["Raw"]["plot"].plot([],[],pen="r")

def update_raw(roi):
	pos = np.array(roi.pos())

	dist = np.linalg.norm(DASHBOARD[f"Raw"]["pos"]-pos,axis=-1)
	idx = dist<= max(0.2,dist.min()+0.2)


	curve.setData(DASHBOARD["Raw"]["rshift"][idx].mean(axis=0),DASHBOARD["Raw"]["intens"][idx].mean(axis=0))





def update_image(roi):
	MIN,MAX = roi.getRegion()


	pos = DASHBOARD["Raw"]["pos"]


	rshift= DASHBOARD["Raw"]["rshift"]
	intens = DASHBOARD["Raw"]["intens"]
	X = DASHBOARD["Raw"]["X"]#[mask]
	Y = DASHBOARD["Raw"]["Y"]#[mask]

	mask = (rshift>MIN)&(rshift<MAX)
	print(mask.sum())
	sig = np.nanmean(np.where(mask, intens, np.nan),axis=-1)

	points = pos
	interp = NearestNDInterpolator(points[~np.isnan(sig)], sig[~np.isnan(sig)], rescale=True)

	img = interp(X,Y)
	img_ = img
	# img_ = np.log10(img)
	# img_ = np.where(np.isinf(img_), np.nan, img_)

	# for j,state in enumerate(POL_STATE):
	# name = f"Mosaic{i}_{state}"
	# im = gaussian_filter(img_[...,j],(0.5,0.5,0.0))
	im = gaussian_filter(img_,(0.5,0.5))
	DASHBOARD["Mosaic"]["imv"].setImage(im, axes={"x":1, "y": 0}, autoRange=False, autoLevels=True)
	DASHBOARD["Mosaic"]["imv"].imageItem.setRect([X.min(), Y.min(), np.ptp(X), np.ptp(Y)])




DASHBOARD["Mosaic"]["imv"].setColorMap(cmap)
DASHBOARD["Mosaic_diff"]["imv"].setColorMap(cmap)

target = pg.TargetItem([50,50])
DASHBOARD["Mosaic"]["imv"].addItem(target)
target.setZValue(100)
DASHBOARD["Mosaic"]["target"] = target
target.sigPositionChanged.connect(update_raw)



roi = pg.LinearRegionItem(orientation="vertical")
DASHBOARD["Raw"]["plot"].addItem(roi)
DASHBOARD["Raw"]["roi"] = roi
roi.sigRegionChangeFinished.connect(update_image)

cm=pg.colormap.getFromMatplotlib("hsv")

hist = pg.HistogramLUTWidget(levelMode="rgba")
DASHBOARD[f"Mosaic"]["hist"] = hist
DASHBOARD[f"Mosaic"]["imv"].ui.gridLayout.addWidget(hist,0,1)

def updateImage(index, mapID=0):
	i = mapID
	name = DASHBOARD[f"Mosaic"]["combo"].currentText()
	DASHBOARD[f"Mosaic"]["hist"].setImageItem(DASHBOARD[f"Mosaic"]["images"][name])

def _import_images(ExpDir,imv_dashboard, pos_):
	imv_dashboard["images"] = import_images(ExpDir, imv_dashboard["imv"], center_of_interest=pos_.mean(axis=0), radius_of_interest=1.5)

	combo = QtWidgets.QComboBox()
	combo.addItems(list(imv_dashboard["images"]))
	def _updateImage(index, mapID=0):
		updateImage(index, mapID=mapID)

	imv_dashboard["combo"] = combo
	imv_dashboard["combo"].currentIndexChanged[int].connect(_updateImage)
	imv_dashboard["imv"].ui.gridLayout.addWidget(combo,1,0)



with exdir.File(exdir_path,'r') as ExpDir:

	intens_ = []
	rshift_ = []
	pos_ = []


	for timestamp in list(ExpDir["scanND"])[-1:]:
		data = ExpDir["scanND"][timestamp]["data"]

		print(data.shape, data.dtype)
		mask = data["time"]>0
		if mask.sum()<10:
			continue

		power = data["metadata"]["CW:power"][mask]

		intens = data["data"]["LabRAM"]["raw"]["intensity"][mask]/data["data"]["LabRAM"]["raw"]["exposure"][mask][:,np.newaxis]/power [:,np.newaxis]
		rshift= data["data"]["LabRAM"]["raw"]["freq_shift"][mask]

		pos = data["position"]["Stage3D:XYZ"][mask,:2] #- np.array([-0.8,1])

		intens_.append(intens)
		rshift_.append(rshift)
		pos_.append(pos)
	print(intens_)
	intens = np.vstack(intens_)
	rshift = np.vstack(rshift_)
	pos = np.vstack(pos_)

	_import_images(ExpDir, DASHBOARD[f"Mosaic"], pos)
	_import_images(ExpDir, DASHBOARD[f"Mosaic_diff"], pos)


	step = 0.2
	X,Y = np.meshgrid(np.arange(pos[:,0].min(),pos[:,0].max(),step), np.arange(pos[:,1].min(),pos[:,1].max(),step))


	mask = (rshift>360)&(rshift<390)
	print(mask.sum())
	sig = np.nanmean(np.where(mask, intens, np.nan),axis=-1)

	points = pos
	interp = NearestNDInterpolator(points[~np.isnan(sig)], sig[~np.isnan(sig)], rescale=True)

	img = interp(X,Y)
	img_ = img
	# img_ = np.log10(img)
	# img_ = np.where(np.isinf(img_), np.nan, img_)

	# for j,state in enumerate(POL_STATE):
	# name = f"Mosaic{i}_{state}"
	# im = gaussian_filter(img_[...,j],(0.5,0.5,0.0))
	im = gaussian_filter(img_,(0.5,0.5))
	DASHBOARD["Mosaic"]["imv"].setImage(im, axes={"x":1, "y": 0})
	DASHBOARD["Mosaic"]["imv"].imageItem.setRect([X.min(), Y.min(), np.ptp(X), np.ptp(Y)])



	DASHBOARD["Raw"]["intens"] = intens
	DASHBOARD["Raw"]["rshift"] = rshift
	DASHBOARD["Raw"]["X"] = X
	DASHBOARD["Raw"]["Y"] = Y
	DASHBOARD["Raw"]["pos"] = pos


	DASHBOARD["Raw"]["points"] = np.array([X.flatten(),Y.flatten()]).T


# for i in range(3):
# 	for j,s in enumerate(POL_STATE):
# 		name = f"Mosaic{i}_{s}"
#
# 		DASHBOARD[f"Mosaic{i}_parallel"]["roi"].setRegion([510-i*30,530-i*30])
# 		DASHBOARD[f"Mosaic{i}_parallel"]["imv"].autoLevels()
#
# 		DASHBOARD[f"Mosaic{i}_parallel"]["imv"].autoRange()

DASHBOARD[f"Mosaic"]["target"].setPos(pos.mean(axis=0))

DASHBOARD[f"Raw"]["roi"].setRegion([360,390])


win.show()
if __name__ == '__main__':
	pg.exec()

