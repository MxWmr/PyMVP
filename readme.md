# PyMVP

## A Python package to correct and analyze Moving Vessel Profiler data

This package was done to be used with MVP300 acquired by ENS (École Normale Supérieure) Paris
The package was developed for the WHIRLS oceanographic cruise (June-July 2026)
It should not be updated after. But you can raise an issue to contact me or use: [maximilien.wemaere@ens-paris-saclay.fr](mailto:maximilien.wemaere@ens-paris-saclay.fr)

## Installation

This package is available on PyPI
`pip install PyMVP`

(if you use a conda environment, be sure to use a pip of your environment with ` conda install pip` and `which pip`)

## How to use

### Main functions

The package is built around an object called Analyzer:

```
import PyMVP as pmvp 

mvpa = pmvp.Analyzer()
```

Then you can load the MVP data (in .raw which are ASCII raw files of MVP or in .nc)

```
path = "path/to/mvp/data"
mvpa.load_mvp_data(path,delp=[ , , , ],subdirs=False) 
```

delp is the list of profiles you want to delete
subdirs is an option to look for MVP raw files also in subdirectories
In the description of the function you can find other optional arguments like outlier filter parameters, offset for temperature, conductivity or oxygen ...

You can also load CTD data for comparison:

```
mvpa.load_ctd_data(path_ctd)
```

(in .nc or .cnv, use the optional arg format= to set it)

/!\ Loading MVP or CTD will work only with the sensor configuration similar to those used for the WHIRLS cruise. But the code can be easily updated to adapt to your configuration. You just have to dive into main.py and mvp_routines.py

The data is formed of different NumPy matrices with one line per profile (upcast or downcast, generally even for downcasts and odd for upcasts)
There is one matrix per variable, for example:

```
mvpa.TEMP_mvp
```

is a matrix of n_profiles x max_points_per_profile (each profile under this value is filled with nan)

As there is no GPS position in the MVP raw file, we can use this function to load it from a .nc or .nav file from the boat; with the datetime we find the position of the MVP at each time. The .nav or .nc file must have 'time' 'lat' 'long' variables.
If the time range of the MVP files is not covered by the nav files in the folder, the function will return an error.

```
mvpa.load_gps_from_ncdf(folder_with_navfiles)
```

There is also a function to load GPS data from CSV:

```
load_GPS_from_csv(navfile.csv)
```

Correction on MVP data can be done with:

```
mvpa.mvp_correction(high_cutoff=0.2,dp=0.1)
```

high_cutoff is the cutoff frequency for filtering
dp is the step of pressure you want for the bin average

Corrected data is presented as dictionaries: one dict per variable
The keys are the IDs of the profiles and they point to a list of the profile (without nan)

Convert the SUNA spectrum data into NO3 concentrations using a calibration file

```
mvpa.process_NO3(path_cal)
```

Interpolate CTD and MVP data on pressure of length n_pres (from min MVP pressure to max MVP pressure)

```
mvpa.interpolate_CTD_and_MVPcorrected(n_pres)
```

Interp data are now back into matrices as they are all the same length

We can compute the distance along the transect using GPS, with the following function:

```
mvpa.compute_dist()
```

(needed to plot section)

### Visualization

There are multiple functions to visualize data:

```
plot_vertical_speed(self,id)
```

Plot a geographical map with all the locations of casts

```
plot_profile_map(self)
```

Plot T and S profiles for MVP data and also CTD if needed

```
plot_TSprofile(self, id_mvp)
```

Plot Fluo and Oxy profiles for MVP data and also CTD if needed

```
plot_BGCprofile(self, id_mvp)
```

Plot T-S diagram for MVP data and also CTD if needed

```
plot_diagramTS(self,id_mvp)
```

Statistically compare MVP and CTD profiles (T, S and O and C)
Generate different plots that can help find the offset for T,S,C and O and the slope of error depending on the depth for O.

```
stat_compar(self,id_mvp=[...],id_ctd)
```

Plot a section of 2D MVP data (corrected and interpolated)
choose var from: 'TEMP', 'COND', 'SAL', 'DO', 'FLUO', 'TURB', 'PH', 'SUNA', 'SPEED'
you can save the plot, giving a name to the save arguments, and you can save the section as a .nc giving the name of the file to save_ncdf

```
plot_MVP_transect(self,var,depth_max,depth_min,vmax,vmin,cmap=None,save=None,save_ncdf=None)
```

Plot a simple transect of corrected and interpolated MVP data for a specified variable. Each profile is offset vertically by a specified delta to visualize multiple profiles on the same plot.

```
plot_MVP_transect_simple(VAR='TEMP',delta=None,l_id=None,depth_max=None,depth_min=None)
```

### Other functions

A summary of loaded data is available via:

```
mvpa.print_profile_metadata()
```

More MVP data from another repo can be loaded:

```
mvpa.load_mvp_data_again(path_to_another_mvp_repo,delp=[])
```

You can apply oxygen correction to all MVP profiles using a specified slope. (that can be computed with mvpa.stat_compar())

```
mvpa.correct_oxygen_with_slope(slope)
```

Not stable functions :

To set MVP data on nearby CTD data, we can delete the offset with the following function

```
mvpa.corrige_MVP_offset_on_ctd_simple(id_mvp,id_ctd,min_depth)
```

We advise choosing a min_depth that avoids taking into account the surface layer which can introduce errors.

For a far more precise detection of offset, if CTD and MVP casts are done at the same place and time we can use the following function that also looks for a pressure offset.

```
mvpa.corrige_MVP_offset_on_ctd_exact(id_mvp,id_ctd,min_depth)
```

All MVP profiles can be corrected via the nearest CTD (via time or geo reference with mode ='Time' or 'Dist')

```
mvpa.corrige_MVP_offset_on_ctd_all(min_depth,mode)
```

An oxygen profile of MVP can be corrected with one CTD profile via:

```
mvpa.correct_oxygen(id_mvp,id_ctd,plotting=True)
```

All oxygen profiles can be corrected with the nearest CTD profile of each one. mode='Temp' for temporal or 'Dist' for geographical distance to find the nearest profile

```
mvpa.correct_oxygen_all(self,mode)
```

The waterflow (the magnitude of the speed of the MVP in the water) can be computed with:

```
mvpa.compute_waterflow(horizontal_speed=2,corr=False)
```

Export MVP data to a NetCDF file using xarray.

```
mvpa.to_netcdf(output_ncdf_path,corrected=False)
```

Export MVP data to CSV files

```
mvpa.to_csv( filepath, corrected=False, per_profile_files=False)
```

set corrected=True to have the corrected data
and set per_profile_files=True to create one CSV file per profile (down or up)

Print all methods of the class with their header:

```
mvpa.help()
```

## Example

This is a common example: others are in the folder example

```
import PyMVP as mvp
import os
import glob

path = "..."
savepath = "..."
savepath_ncdf= "..."

# create the analyzer object
mvpa = mvp.Analyzer()

# load the data (if there is a bad file, put its number in delp, so it can be ignored)
# here subdirs=False, means that the MVP files .raw will be searched in the path directly otherwise it would be searched in subdirectories of the path (useful to concatenate multiple sections).
mvpa.load_mvp_data(path,subdirs=False,delp=[12])

# check if there are the correct profiles, and make sure that there is an even number of profiles (otherwise it misses an upcast or downcast, and the correction will be affected)
mvpa.print_profile_metadata()


# load the GPS data
mvpa.load_gps_from_ncdf('...')

# correct the MVP profiles
mvpa.mvp_correction()

# process the NO3 spectrums
path_cal = "/..."
mvpa.process_NO3(path_cal)

# interpolate the MVP data (here on 10000 levels of pressure)
mvpa.interpolate_CTD_and_MVPcorrected(10000)

# compute the distance along the section
mvpa.compute_dist()


## Visualization

# plot the vertical speed
mvpa.plot_vertical_speed()

# plot ox,turb,fluo,NO3 profiles (number id must be even as it is a downcast, and the upcast will also be plotted with id: id+1)  
mvpa.plot_BGCprofile(6) 

# plot T,S profiles  (number id must be even)
mvpa.plot_TSprofile(4)


# plot TS diagram
mvpa.plot_diagramTS(correction=True,save=savepath+'section3_TSdiag.png')

# plot transect of temperature
mvpa.plot_MVP_transect(VAR='TEMP',depth_max=1000,vmin=8,save=savepath+"section3_temp.png")

# plot transect of salinity
mvpa.plot_MVP_transect(VAR='SAL',depth_max=1000,save=savepath+"section3_sal.png")
```


