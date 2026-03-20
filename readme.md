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


Correction on MVP data can be done with:

```
mvpa.mvp_correction(high_cutoff=0.2,dp=0.1)
```
high_cutoff is the cutoff frequency for filtering
dp is the step of pressure you want for the bin average


Corrected data is presented as dictionaries: one dic per variables 
The keys are the id of the profiles and they point to a list of the profile (without nan)


Interpolate CTD and MVP data on pressure of length n_pres (from min MVP pressure to max MVP pressure)
```
mvpa.interpolate_CTD_and_MVPcorrected(n_pres)
```
Interp data are now back into matrix as they are all the same length


To set MVP data on nearby CTD data, we can delete the offset with the following function
```
mvpa.corrige_MVP_offset_on_ctd_simple(id_mvp,id_ctd,min_depth)
```
We advise to choose a min_depth that avoid to take into acount the surface layer which can introduce errors.

For a far more precise detection of offset, if CTD and MVP cast are done at the same place and same time we can use the following function that also look for a pressure offset.

```
mvpa.corrige_MVP_offset_on_ctd_exact(id_mvp,id_ctd,min_depth)
```

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
```
plot_MVP_transect(self,var,dpepth_max,depth_min,vmax,vmin,cmap)
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

Oxygen data from RINKO3 can be corrected via:
```
mvpa.correct_oxygen()
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
