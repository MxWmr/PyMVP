# PyMVP
## A python package to correct and analyze moving vessel profiler data

This package was done to be used with MVP300 acquired by ENS (Ecole Normale Supérieure) Paris
It is stille in progress until WHIRLS mission in  june-july 2026


## Installation

This package is avalaible on PyPi
``` pip install PyMVP ```

(if you use conda environment, be sure to use a pip of your environment with ``` conda install pip``` and ```which pip```)


For V0.1.0, it will be available on conda



## How to use 


### Main functions

The package is build around an object called Analyzer:

``` 
import PyMVP as pmvp 

mvpa = pmvp.Analyzer()
``` 

Then you can load the MVP data (in .raw which are ascii raw files of MVP or in .nc)

```
path = "path/to/mvp/data"
mvpa.load_mvp_data(path,delp=[ , , , ]) 
```
delp is the list of profiles you want to delete

You can also load CTD data (only in .nc as seabird package is no longuer up to date so .csv have to be converted into .nc) for comparison:

```
mvpa.load_ctd_data(path_ctd)
```

The data is formed of different numpy matrix with one line per profile (upcast or downcast, generally even for downcasts and odd for upcasts)
There is one matrix per variables, for example:
```
mvpa.TEMP_mvp
```
is a a matrix of n_profiles x max_points_per_profile (each profile under this value is filled with nan)

As there is no GPS position in the MVP raw file, we can use this funciton, to load it from a .nc or .nav file from the boat, with the datetime we find the position of the MVP at each time. The .nav or .nc file must have 'time' 'lat' 'long' variables.
If the time range of thze MVP files is not covered by the nav files in the folder, the function will return an error.

mvpa.load_gps_from_ncdf(folder_with_navfiles)




Correction on MVP data can be done with:

```
mvpa.mvp_correction(high_cutoff=0.2,dp=0.1)
```
high_cutoff is the cutoff frequency for filtering
dp is the step of pressure you want for the bin average


Corrected data is presented as dictionaries: one dic per variables 
The keys are the id of the profiles and they point to a list of the profile (without nan)



Convert the SUNA spectrum data into NO3 concentrations using a calibration file
```
mvpa.process_NO3(path_cal)
```


Interpolate CTD and MVP data on pressure of length n_pres (from min MVP pressure to max MVP pressure)
```
mvpa.interpolate_CTD_and_MVPcorrected(n_pres)
```
Interp data are now back into matrix as they are all the same length


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

Plot a geographical map with all the locations of cast
```
plot_profile_map(self)
```

plot T and S profile for mvp data and also ctd if needed
```
plot_TSprofile(self, id_mvp)
```

lot Fluo and Oxy profile for mvp data and also ctd if needed
```
plot_BGCprofile(self, id_mvp)
```

plot T-S diagram for mvp data and also ctd if needed
```
plot_diagramTS(self,id_mvp)
```

statistically compara MVP adn CTD profiles (T and S)
```
stat_compar(self,id_mvp=[...],id_ctd)
```

Plot a section of 2D MVP data (corrected and interpolated)
choose var from: 'TEMP', 'COND', 'SAL', 'DO', 'FLUO', 'TURB', 'PH', 'SUNA', 'SPEED'
you can save the plot, giving a name to the save arguments, and you can save teh section as a .nc giving the name of the file to save_ncdf

```
plot_MVP_transect(self,var,depth_max,depth_min,vmax,vmin,cmap=None,save=None,save_ncdf=None)
```


### Other functions

A sum up of loaded data is available via:
```
mvpa.print_profile_metadata()
```

More MVP data from another repo can be load:
```
mvpa.load_mvp_data_again(path_to_another_mvp_repo,delp=[])
```

To set MVP data on nearby CTD data, we can delete the offset with the following function
```
mvpa.corrige_MVP_offset_on_ctd_simple(id_mvp,id_ctd,min_depth)
```
We advise to choose a min_depth that avoid to take into acount the surface layer which can introduce errors.

For a far more precise detection of offset, if CTD and MVP cast are done at the same place and same time we can use the following function that also look for a pressure offset.

```
mvpa.corrige_MVP_offset_on_ctd_exact(id_mvp,id_ctd,min_depth)
```

All MVP profiles can be corrected via the nearest CTD (via time or geo reference with mode ='Time' or 'Dist')
```
mvpa.corrige_MVP_offset_on_ctd_all(min_depth,mode)
```

An oxygen profile of mvp can corrected with one ctd profile via:
```
mvpa.correct_oxygen(id_mvp,id_ctd,plotting=True)
```

All oxgen profiles can be corrected with nearest CTD profile of each one. mode='Temp' for temporal or 'Dist' for geographical distance to find the nearest profile
```
mvpa.correct_oxygen_all(self,mode)
```

The waterflow (the magnitude of the speed of the MVP in the water) can be computed with:
```
mvpa.compute_waterflow(horizontal_speed=2,corr=False)
```


Export MVP data to a NetCDF file using xarray.
```
mvpa.to_netcdf(output_ncdf_path)
```

Print all methods of the class with their header:
```
mvpa.help()
```


## Example


```
import PyMVP as mvp
import os
import glob

path = "..."
savepath = "..."
savepath_ncdf= "..."

# create the analyzer object
mvpa = mvp.Analyzer()

# load the data (if there is a bad file, put it's number in delp, so it can be ignored)
# here subdirs=False, means that the MVP files .raw will be searched in the path directly otherwise it would be searched in sunbdirectories of the path (useful to concatenate multiples sections).
mvpa.load_mvp_data(path,subdirs=False,delp=[12])

# check if there is the good profiles, and make sure that there is an even number of profile (otherwise it miss a upcast or downcast, and the correction will be affected)
mvpa.print_profile_metadata()


# load the gps data
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


## Vizualisation

# plot the vertical speed
mvpa.plot_vertical_speed()

# plot ox,turb,fluo,NO3 profiles (number id must be even as it is a downcast, and the upcast will also be plot of id: id+1)  
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
