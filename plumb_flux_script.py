from netCDF4 import Dataset
import sys
from scipy.interpolate import interp1d
import time
import os
import imp
try:
    imp.find_module('bottleneck')
    import bottleneck as bn
    import numpy as np
except ImportError:
    import numpy as bn

def c_diff(arr, h, dim, cyclic = False):
    ndim = arr.ndim
    lst = [i for i in xrange(ndim)]

    lst[dim], lst[0] = lst[0], lst[dim]
    rank = lst 
    arr = np.transpose(arr, tuple(rank))

    if ndim == 3:
        shp = (arr.shape[0]-2,1,1)
    elif ndim == 4:
        shp = (arr.shape[0]-2,1,1,1)
    
    d_arr = np.copy(arr)
    if not cyclic:  
	d_arr[0,...] = (arr[1,...]-arr[0,...])/(h[1]-h[0])
	d_arr[-1,...] = (arr[-1,...]-arr[-2,...])/(h[-1]-h[-2])
        d_arr[1:-1,...] = (arr[2:,...]-arr[0:-2,...])/np.reshape(h[2:]-h[0:-2], shp)

    elif cyclic:
	d_arr[0,...] = (arr[1,...]-arr[-1,...])/(h[1]-h[-1])
	d_arr[-1,...] = (arr[0,...]-arr[-2,...])/(h[0]-h[-2])
	d_arr[1:-1,...] = (arr[2:,...]-arr[0:-2,...])/np.reshape(h[2:]-h[0:-2], shp)

    d_arr = np.transpose(d_arr, tuple(rank))

    return d_arr

def rmv_mean(arr):
	return arr-bn.nanmean(arr,axis=3)[...,np.newaxis]

def interp(lev, data, lev_int):
	f = interp1d(lev[::-1],data[:,::-1,...],axis=1)
	return f(lev_int[::-1])[:,::-1,...]

in_file = sys.argv[1] #input file
scale_by_sqrt_p = True #scaled by square root of (1000/p)
inter_bool = True #interpolate to regular vertical grid

print 'openning files'
dataset = Dataset(in_file, 'r')
t = dataset.variables['te'][:]
g = dataset.variables['geopoth'][:]
lon = dataset.variables['lon'][:]
lat = dataset.variables['lat'][:]
lev = dataset.variables['lev'][:]
tim = dataset.variables['time'][:]
units = dataset.variables['time'].units
dataset.close()

nlev = lev.shape[0]
nlat = lat.shape[0]
nlon = lon.shape[0]
ntime = tim.shape[0]


print 'calculation'
#constants
ga = 9.80665
sclhgt=8000.
loglevel = np.log(lev/1000.)
gc = 290.
a = 6.37122e06 
pi = np.pi
phi = lat*pi/180.0     
omega = 7.2921e-5      
f = 2*omega*np.sin(phi) 
lati = np.abs(lat) <= 10
f[lati] = np.nan
lev_tmp = np.reshape(lev,(1,nlev,1,1))
cos_tmp = np.reshape(np.cos(phi),(1,1,nlat,1))
f_tmp = np.reshape(f,(1,1,nlat,1))

theta = bn.nanmean(t, axis=3)[...,np.newaxis]*(1000./lev_tmp)**(0.286) #potential temperature

dthetadz =  c_diff(theta, -sclhgt*loglevel, 1, False)
NN = (gc*(lev_tmp/1000.)**0.286)/sclhgt * dthetadz #Brunt-Vaisala

psidev = rmv_mean(g)*ga/f_tmp #QG streamfunction 
dpsidevdlon = c_diff(psidev, lon, 3, True)
dpsidevdlonlon = c_diff(dpsidevdlon, lon, 3, True) 
dpsidevdlat = c_diff(psidev, lat, 2, False)
dpsidevdlonlat = c_diff(dpsidevdlon, lat, 2, False)
dpsidevdz = c_diff(psidev, -sclhgt*loglevel, 1, False)
dpsidevdlonz = c_diff(dpsidevdlon, -sclhgt*loglevel, 1, False)

#eq. 5.7 in Plumb (1985)
Fx = ((lev_tmp/1000.)/(2*a*a*cos_tmp))*(dpsidevdlon*dpsidevdlon - psidev*dpsidevdlonlon)
Fy = ((lev_tmp/1000.)/(2*a*a))*(dpsidevdlon*dpsidevdlat - psidev*dpsidevdlonlat)
Fz = ((f_tmp*f_tmp*(lev_tmp/1000.))/(2*NN*a))*(dpsidevdlon*dpsidevdz - psidev*dpsidevdlonz)

if inter_bool:
	print 'interpolation'
	lev_int = np.concatenate((10**np.linspace(3,2.1,10),10**np.linspace(2,-0.95,15)))#10**np.linspace(2,-0.8,15 
	Fx_int = interp(lev, Fx, lev_int)
	Fy_int = interp(lev, Fy, lev_int)  
	Fz_int = interp(lev, Fz, lev_int)
	nlev2 = lev_int.shape[0]       
else:
	nlev2 = nlev
	lev_int = lev
	Fx_int = np.copy(Fx)
	Fy_int = np.copy(Fy) 
	Fz_int = np.copy(Fz)

print 'NetCDF files writing'
os.system('mkdir -p ./Plumb_flux/')
f = Dataset('./Plumb_flux/'+in_file, 'w')#, format='NETCDF3_CLASSIC')
f.createDimension('lat', nlat)
f.createDimension('lev', nlev2)
f.createDimension('lon', nlon)
f.createDimension('time', None)

latitude = f.createVariable('lat', np.float32, ('lat',))
longitude = f.createVariable('lon', np.float32, ('lon',))
levels = f.createVariable('lev', np.float32, ('lev',))
t = f.createVariable('time', np.float64, ('time',))

fx = f.createVariable('Fx', np.float32, ('time','lev','lat','lon',))
fy = f.createVariable('Fy', np.float32, ('time','lev','lat','lon',))
fz = f.createVariable('Fz', np.float32, ('time','lev','lat','lon',))

f.description = 'Plumb 3D flux (6-hourly)'
f.history = 'Created ' + time.ctime(time.time())
f.source = 'netCDF4 python module'
t.units = units#'days since 1979-01-01 00:00:00.0'
latitude.units = 'degrees north'

levels.units = 'hPa'
fx.units = 'm^2/s^2'
fy.units = 'm^2/s^2'
fz.units = 'm^2/s^2'

t[:] = tim
latitude[:] = lat
levels[:] = lev_int
longitude[:] = lon

fx[:] = Fx_int
fy[:] = Fy_int
fz[:] = Fz_int

f.close()			
print 'done'
 
