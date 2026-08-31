##########################################################################
# PyMVP/main.py
# Author: Maximilien Wemaere (LMD/CNRS)
# Date: March 2026
#
#
# Simple routines to load, analyze and correct data from a Moving Vessel Profiler (MVP) 300
# Requires numpy, matplotlib, gsw, seabird, tqdm, cartopy
#
#
# Routines to read mvp data are adapted from routines provided by Pierre l'Hegaret (UBO)
# (mvp_routines.py, temporal_lag_correction.py, thermal_mass_correction.py)
#
#
# STILL IN DEVELOPMENT !
#
#
##########################################################################



from re import L

import numpy as np 
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import gsw
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from . import mvp_routines as mvp
from . import NO3_process as no3_p
from scipy.ndimage import median_filter
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from geopy.distance import geodesic
import pandas as pd
from scipy.interpolate import interp1d

  
class Analyzer:
    def __init__(self, Yorig=1950):
        """
        Initialize the analyzer with the reference year.
        Args:
            Yorig (int): Reference year for dates (default 1950).
        """
        self.Yorig = Yorig
        self.date_ref = datetime(Yorig, 1, 1)
        self.mvp = False
        self.ctd = False
        self.speed = False
        self.corrected = False
        self.GPS = False
        self.SUNA = False
        self.dist = False

    def ___version___(self):
        return "1.0.0"


    def load_mvp_data(self,data_path, delp=[], subdirs=False,format='raw',only_new=False, output_path=None,outlier_temp=35,outlier_cond=55,offset_cond=0,offset_temp=0,offset_oxy=0):
        """
        Load MVP data from .raw and .log files in the data_path folder.
        Fills the object attributes with data matrices and associated metadata.
        Args:
            data_path (str): Path to the folder containing MVP files.
            delp (list): Indices of profiles to remove from the list (optional).
            subdirs (bool): Whether to search in subdirectories for MVP files (default False).
            format (str): Format of the input files, either 'raw' for .raw and .log files or 'ncdf' for .nc files (default 'raw').
            only_new (bool): Whether to load only new files (default False).
            output_path (str): Path to the folder where corrected files will be saved (optional).
        """
        self.data_path = data_path
        self.subdirs = subdirs
        self.output_path = output_path
        self.oxy_offset = offset_oxy

        if format=='raw':
            if self.subdirs:
                files = sorted(filter(os.path.isfile,glob.glob(self.data_path + '**/*.raw', recursive=True)))
            else:
                files = sorted(filter(os.path.isfile,glob.glob(self.data_path + '*.raw', recursive=self.subdirs)))


            if only_new:
                list_output = [f for f in os.listdir(self.output_path) if f.endswith(".nc")]
                files = [f for f in files if not "MVP_"+os.path.basename(f).replace('.raw', '.nc') in list_output]

        elif format=='ncdf':
            if self.subdirs:
                files = sorted(filter(os.path.isfile,glob.glob(self.data_path + '**/MVP*.nc', recursive=True)))
            else:
                files = sorted(filter(os.path.isfile,glob.glob(self.data_path + 'MVP*.nc', recursive=self.subdirs)))

            if only_new:
                list_output = [f for f in os.listdir(self.output_path) if f.endswith(".nc")]
                files = [f for f in files if not "MVP_"+os.path.basename(f) in list_output]


        print('Found ' + str(len(files)) + ' MVP files in the directory: ' + self.data_path)



        if format=='ncdf':
            
            for f in files:
                nc = xr.open_dataset(f)
                self.PRES_mvp = nc['PRES'].values
                self.TEMP_mvp = nc['TEMP'].values
                self.COND_mvp = nc['COND'].values
                self.SOUNDVEL_mvp = nc['SOUNDVEL'].values
                self.DO_mvp = nc['DO'].values
                self.TEMP2_mvp = nc['TEMP2'].values
                self.SUNA_mvp = nc['SUNA'].values
                self.FLUO_mvp = nc['FLUO'].values
                self.TURB_mvp = nc['TURB'].values
                self.PH_mvp = nc['PH'].values
                self.SALT_mvp = nc['SAL'].values
                self.TIME_mvp = nc['TIME_s'].values
                self.Lat_mvp = nc['LATITUDE'].values
                self.Lon_mvp = nc['LONGITUDE'].values
                self.DATETIME_mvp = nc['profile_time'].values
                self.label_mvp = nc['profile'].values
                self.freq_echant = nc.attrs['sampling frequency_hz']

                nc.close()
                print('MVP data loaded successfully.')
                self.mvp = True

                return

        PRES_temp = []
        TEMP_temp = []
        COND_temp = []
        SOUNDVEL_temp = []
        DO_temp = []
        TEMP2_temp = [] # temp from DO sensor
        SUNA_temp = []
        FLUO_temp = [] 
        TURB_temp = []
        PH_temp = [] 
        SALT_temp = []
        TIME_mvp_temp = []
        LAT_temp = []
        LON_temp= []
        DATETIME_mvp = []
        DIR = []
        Label_mvp = []

        delp.sort(reverse=True)
        self.delp =delp
        for i in delp:
            del files[i]

        for f_i,mvp_dat_name in enumerate(files[0:]):

            mvp_log_name=mvp_dat_name[:-4]+'.log'

            # Get start and end time of the cycle

            if format=='raw':
                try:
                    (mvp_tstart,mvp_tend,cycle_dur, lat, lon, dt_station) = mvp.get_log(mvp_log_name,self.Yorig)
                except:
                    delp.append(f_i)
                    continue

            if cycle_dur>1:

                # Read one cycle MVP data  
                (pres,soundvel,cond,temp,do_raw,temp2_raw,suna_raw,fluo_raw,turb_raw,ph_raw) = mvp.read_mvp_cycle_raw(mvp_dat_name)
                if len(pres)<=20:
                    self.delp.append(f_i)
                    continue
                (pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph) = mvp.raw_data_conversion(pres,soundvel,cond,temp,do_raw,temp2_raw,suna_raw,fluo_raw,turb_raw,ph_raw)

                freq_echant = float(len(pres)/cycle_dur)

                
                if np.nanmax(pres)-np.nanmin(pres)>2:

                    # Allocate time to samples and select the ascending part 
                    (pres_up,soundvel_up,cond_up,temp_up,do_up,temp2_up,suna_up,fluo_up,turb_up,ph_up,time_up) = mvp.time_mvp_cycle_up([pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph],mvp_tstart,mvp_tend)
                    (pres_down,soundvel_down,cond_down,temp_down,do_down,temp2_down,suna_down,fluo_down,turb_down,ph_down,time_down) = mvp.time_mvp_cycle_down([pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph],mvp_tstart,mvp_tend)

                    if len(pres_down)>0 and np.nanmax(pres_down)-np.nanmin(pres_down)>2:
                        print(f_i,' ',mvp_dat_name.split('/')[-1],' length of down profile:', len(pres_down))
                        PRES_temp.append(pres_down)
                        SOUNDVEL_temp.append(soundvel_down)
                        COND_temp.append(cond_down)
                        TEMP_temp.append(temp_down)
                        DO_temp.append(do_down)
                        TEMP2_temp.append(temp2_down)
                        SUNA_temp.append(suna_down)
                        FLUO_temp.append(fluo_down)
                        TURB_temp.append(turb_down)
                        PH_temp.append(ph_down)
                        SALT_temp.append(gsw.SP_from_C(cond_down, temp_down,pres_down))
                        TIME_mvp_temp.append(time_down)
                        LAT_temp.append(lat)
                        LON_temp.append(lon)

                        DIR.append('down')
                        Label_mvp.append(mvp_dat_name.split('/')[-1])

                    else:
                        print('ohohoh no down profile found for file: ' + mvp_dat_name)
                        # for l in [PRES_temp,SOUNDVEL_temp,COND_temp,TEMP_temp,DO_temp,TEMP2_temp,SUNA_temp,FLUO_temp,TURB_temp,PH_temp,SALT_temp,TIME_mvp_temp,LAT_temp,LON_temp,DIR,Label_mvp]:
                        #     l.append([np.nan])
                        self.delp.append(f_i)
                        continue
                            
                    if len(pres_up)>0 and np.nanmax(pres_up)-np.nanmin(pres_up)>2:
                        print(f_i,' ',mvp_dat_name.split('/')[-1],' length of up profile:',len(pres_up))
                    
                        PRES_temp.append(pres_up)
                        SOUNDVEL_temp.append(soundvel_up)
                        COND_temp.append(cond_up)
                        TEMP_temp.append(temp_up)
                        DO_temp.append(do_up)
                        TEMP2_temp.append(temp2_up)
                        SUNA_temp.append(suna_up)
                        FLUO_temp.append(fluo_up)
                        TURB_temp.append(turb_up)
                        PH_temp.append(ph_up)
                        SALT_temp.append(gsw.SP_from_C(cond_up, temp_up,pres_up))
                        TIME_mvp_temp.append(time_up)
                        LAT_temp.append(lat)
                        LON_temp.append(lon)
                        DIR.append('up')
                        Label_mvp.append(mvp_dat_name.split('/')[-1])
                    else:
                        print('ohohoh no up profile found for file: ' + mvp_dat_name)
                        for l in [PRES_temp,SOUNDVEL_temp,COND_temp,TEMP_temp,DO_temp,TEMP2_temp,SUNA_temp,FLUO_temp,TURB_temp,PH_temp,SALT_temp,TIME_mvp_temp,LAT_temp,LON_temp,DIR,Label_mvp]:
                            # l.append([np.nan])
                            l.pop()
                        self.delp.append(f_i)
                        continue
                else:
                    print('ohohoh no profile found for file: ' + mvp_dat_name)
                    self.delp.append(f_i)
                    continue

                DATETIME_mvp.append(dt_station)

            else:
                print('ohohoh no profile found for file: ' + mvp_dat_name)
                self.delp.append(f_i)
                continue               
                
                    

        # Re-arange files into matrices
        M_size = 0
        for i in range(len(PRES_temp)):
            M_size = max(M_size, len(PRES_temp[i]))
            
        PRES_mvp = np.zeros(( len(PRES_temp), M_size))
        SOUNDVEL_mvp = np.zeros(( len(PRES_temp), M_size))
        COND_mvp = np.zeros(( len(PRES_temp), M_size))
        TEMP_mvp = np.zeros(( len(PRES_temp), M_size))
        DO_mvp = np.zeros(( len(PRES_temp), M_size))
        TEMP_mvp2 = np.zeros(( len(PRES_temp), M_size))
        SUNA_mvp = np.zeros(( len(PRES_temp), M_size))
        FLUO_mvp = np.zeros(( len(PRES_temp), M_size))
        TURB_mvp = np.zeros(( len(PRES_temp), M_size))
        PH_mvp = np.zeros(( len(PRES_temp), M_size))
        SALT_mvp = np.zeros(( len(PRES_temp), M_size))
        TIME_mvp = np.zeros(( len(PRES_temp), M_size))
        LAT_mvp = np.zeros(( len(PRES_temp), M_size))
        LON_mvp = np.zeros(( len(PRES_temp), M_size))
        PRES_mvp[:] = np.nan
        SOUNDVEL_mvp[:] = np.nan
        COND_mvp[:] = np.nan
        TEMP_mvp[:] = np.nan
        DO_mvp[:] = np.nan
        TEMP_mvp2[:] = np.nan
        SUNA_mvp[:] = np.nan
        FLUO_mvp[:] = np.nan
        TURB_mvp[:] = np.nan
        PH_mvp[:] = np.nan
        SALT_mvp[:] = np.nan
        TIME_mvp[:] = np.nan
        LAT_mvp[:] = np.nan
        LON_mvp[:] = np.nan

        del M_size

        for i in range(len(PRES_temp)):
            PRES_mvp[i,0:len(PRES_temp[i])] = PRES_temp[i]
            SOUNDVEL_mvp[i,0:len(SOUNDVEL_temp[i])] = SOUNDVEL_temp[i]
            COND_mvp[i,0:len(COND_temp[i])] = COND_temp[i]
            TEMP_mvp[i,0:len(TEMP_temp[i])] = TEMP_temp[i]
            DO_mvp[i,0:len(DO_temp[i])] = DO_temp[i]
            TEMP_mvp2[i,0:len(TEMP2_temp[i])] = TEMP2_temp[i]
            SUNA_mvp[i,0:len(SUNA_temp[i])] = SUNA_temp[i]
            FLUO_mvp[i,0:len(FLUO_temp[i])] = FLUO_temp[i]
            TURB_mvp[i,0:len(TURB_temp[i])] = TURB_temp[i]
            PH_mvp[i,0:len(PH_temp[i])] = PH_temp[i]
            SALT_mvp[i,0:len(SALT_temp[i])] = SALT_temp[i]
            TIME_mvp[i,0:len(TIME_mvp_temp[i])] = TIME_mvp_temp[i]
            LAT_mvp[i,0:len(PRES_temp[i])] = LAT_temp[i]
            LON_mvp[i,0:len(PRES_temp[i])] = LON_temp[i]




        # filter outlier for temp and cond
        TEMP_mvp[TEMP_mvp>outlier_temp] = np.nan
        TEMP_mvp[TEMP_mvp<-1] = np.nan
        COND_mvp[COND_mvp>outlier_cond] = np.nan
        COND_mvp[COND_mvp<10] = np.nan
        SALT_mvp[SALT_mvp>45] = np.nan
        SALT_mvp[SALT_mvp<30] = np.nan
        
         
        self.PRES_mvp = PRES_mvp
        self.SOUNDVEL_mvp = SOUNDVEL_mvp
        self.COND_mvp = COND_mvp+offset_cond
        self.TEMP_mvp = TEMP_mvp+offset_temp
        self.DO_mvp = DO_mvp
        self.TEMP2_mvp = TEMP_mvp2
        self.SUNA_mvp = SUNA_mvp
        self.FLUO_mvp = FLUO_mvp
        self.TURB_mvp = TURB_mvp
        self.PH_mvp = PH_mvp
        self.SALT_mvp = SALT_mvp
        self.TIME_mvp = TIME_mvp
        self.LAT_mvp = LAT_mvp
        self.LON_mvp = LON_mvp
        self.DATETIME_mvp = DATETIME_mvp
        self.DIR = DIR
        self.label_mvp = Label_mvp
        self.freq_echant = freq_echant
    
        del PRES_temp, SOUNDVEL_temp, DO_temp, TEMP2_temp, SUNA_temp, FLUO_temp, TURB_temp, PH_temp, COND_temp, TEMP_temp, SALT_temp, TIME_mvp_temp, LAT_temp, LON_temp        

        print('MVP data loaded successfully.')
        self.mvp = True

        self.convert_DO_to_umolL()




    def load_mvp_data_again(self,data_path,format='raw',delp=[]):
        """
        Load MVP data from .raw and .log files in the data_path folder.
        Fills the object attributes with data matrices and associated metadata.
        Args:
            data_path (str): Path to the folder containing MVP files.
            delp (list): Indices of profiles to remove from the list (optional).
        """


        if format=='raw':
            files = sorted(filter(os.path.isfile,glob.glob(data_path + '*.raw', recursive=True)))
        elif format=='ncdf':
            files = sorted(filter(os.path.isfile,glob.glob(data_path + '**/MVP*.nc', recursive=True)))
        print('Found ' + str(len(files)) + ' MVP files in the directory: ' + data_path)



        if format=='ncdf':
            for f in files:
                nc = xr.open_dataset(f)
                self.PRES_mvp = nc['PRES'].values
                self.TEMP_mvp = nc['TEMP'].values
                self.COND_mvp = nc['COND'].values
                self.SOUNDVEL_mvp = nc['SOUNDVEL'].values
                self.DO_mvp = nc['DO'].values
                self.TEMP2_mvp = nc['TEMP2'].values
                self.SUNA_mvp = nc['SUNA'].values
                self.FLUO_mvp = nc['FLUO'].values
                self.TURB_mvp = nc['TURB'].values
                self.PH_mvp = nc['PH'].values
                self.SALT_mvp = nc['SAL'].values
                self.TIME_mvp = nc['TIME'].values
                self.Lat_mvp = nc['LATITUDE'].values
                self.Lon_mvp = nc['LONGITUDE'].values
                self.DATETIME_mvp = nc['profile_time'].values
                self.Label_mvp = nc['profile'].values
                self.freq_echant = nc.attrs['sampling frequency_hz']

                nc.close()
                print('MVP data loaded successfully.')
                self.mvp = True

                return




        PRES_temp = []
        TEMP_temp = []
        COND_temp = []
        SOUNDVEL_temp = []
        DO_temp = []
        TEMP2_temp = [] # temp from DO sensor
        SUNA_temp = []
        FLUO_temp = [] 
        TURB_temp = []
        PH_temp = [] 
        SALT_temp = []
        TIME_mvp_temp = []
        LAT_temp = []
        LON_temp= []
        DATETIME_mvp = []
        DIR = []
        Label_mvp = []

        delp.sort(reverse=True)
        for i in delp:
            del files[i]

        for mvp_dat_name in files[0:]:

            mvp_log_name=mvp_dat_name[:-4]+'.log'

            # Get start and end time of the cycle
            (mvp_tstart,mvp_tend,cycle_dur, lat, lon, dt_station) = mvp.get_log(mvp_log_name,self.Yorig)

            
            if cycle_dur>1:

                # Read one cycle MVP data  

                (pres,soundvel,cond,temp,do_raw,temp2_raw,suna_raw,fluo_raw,turb_raw,ph_raw) = mvp.read_mvp_cycle_raw(mvp_dat_name)
                (pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph) = mvp.raw_data_conversion(pres,soundvel,cond,temp,do_raw,temp2_raw,suna_raw,fluo_raw,turb_raw,ph_raw)
                   

                freq_echant = float(len(pres)/cycle_dur)

                DATETIME_mvp.append(dt_station)
                
                if np.nanmax(pres)-np.nanmin(pres)>2:

                    # Allocate time to samples and select the ascending part 
                    (pres_up,soundvel_up,cond_up,temp_up,do_up,temp2_up,suna_up,fluo_up,turb_up,ph_up,time_up) = mvp.time_mvp_cycle_up([pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph],mvp_tstart,mvp_tend)
                    (pres_down,soundvel_down,cond_down,temp_down,do_down,temp2_down,suna_down,fluo_down,turb_down,ph_down,time_down) = mvp.time_mvp_cycle_down([pres,soundvel,cond,temp,do,temp2,suna,fluo,turb,ph],mvp_tstart,mvp_tend)


                    if len(pres_down)>0:
                        if np.nanmax(pres_down)-np.nanmin(pres_down)>2:
                            PRES_temp.append(pres_down)
                            SOUNDVEL_temp.append(soundvel_down)
                            COND_temp.append(cond_down)
                            TEMP_temp.append(temp_down)
                            DO_temp.append(do_down)
                            TEMP2_temp.append(temp2_down)
                            SUNA_temp.append(suna_down)
                            FLUO_temp.append(fluo_down)
                            TURB_temp.append(turb_down)
                            PH_temp.append(ph_down)
                            SALT_temp.append(gsw.SP_from_C(cond_down, temp_down,pres_down))
                            TIME_mvp_temp.append(time_down)
                            LAT_temp.append(lat)
                            LON_temp.append(lon)

                            DIR.append('down')
                            Label_mvp.append(mvp_dat_name.replace('\\','/').split('/')[-2])

                    else:
                        print('ohohoh no down profile found for file: ' + mvp_dat_name)

                            
                    if len(pres_up)>0:
                        if np.nanmax(pres_up)-np.nanmin(pres_up)>2:
                            PRES_temp.append(pres_up)
                            SOUNDVEL_temp.append(soundvel_up)
                            COND_temp.append(cond_up)
                            TEMP_temp.append(temp_up)
                            DO_temp.append(do_up)
                            TEMP2_temp.append(temp2_up)
                            SUNA_temp.append(suna_up)
                            FLUO_temp.append(fluo_up)
                            TURB_temp.append(turb_up)
                            PH_temp.append(ph_up)
                            SALT_temp.append(gsw.SP_from_C(cond_up, temp_up,pres_up))
                            TIME_mvp_temp.append(time_up)
                            LAT_temp.append(lat)
                            LON_temp.append(lon)

                            DIR.append('up')
                            Label_mvp.append(mvp_dat_name.replace('\\','/').split('/')[-2])


                    else:
                        print('ohohoh no up profile found for file: ' + mvp_dat_name)

                else:
                    print('ohohoh no profile found for file: ' + mvp_dat_name)

                    
                    

        # Re-arange files into matrices
        M_size = 0
        for i in range(len(PRES_temp)):
            M_size = max(M_size, len(PRES_temp[i]))
            
        if M_size < self.PRES_mvp.shape[1]:
            M_size = self.PRES_mvp.shape[1]
        else:
            nan_cols = np.full((self.PRES_mvp.shape[0], M_size - self.PRES_mvp.shape[1]), np.nan)
            self.PRES_mvp = np.hstack((self.PRES_mvp, nan_cols))
            self.SOUNDVEL_mvp = np.hstack((self.SOUNDVEL_mvp, nan_cols))
            self.COND_mvp = np.hstack((self.COND_mvp, nan_cols))
            self.TEMP_mvp = np.hstack((self.TEMP_mvp, nan_cols))
            self.DO_mvp = np.hstack((self.DO_mvp, nan_cols))
            self.TEMP2_mvp = np.hstack((self.TEMP2_mvp, nan_cols))
            self.SUNA_mvp = np.hstack((self.SUNA_mvp, nan_cols))
            self.FLUO_mvp = np.hstack((self.FLUO_mvp, nan_cols))
            self.TURB_mvp = np.hstack((self.TURB_mvp, nan_cols))
            self.PH_mvp = np.hstack((self.PH_mvp, nan_cols))
            self.SALT_mvp = np.hstack((self.SALT_mvp, nan_cols))
            self.TIME_mvp = np.hstack((self.TIME_mvp, nan_cols))
            self.Lat_mvp = np.hstack((self.Lat_mvp, nan_cols))
            self.Lon_mvp = np.hstack((self.Lon_mvp, nan_cols))




        PRES_mvp = np.zeros(( len(PRES_temp), M_size))
        SOUNDVEL_mvp = np.zeros(( len(PRES_temp), M_size))
        COND_mvp = np.zeros(( len(PRES_temp), M_size))
        TEMP_mvp = np.zeros(( len(PRES_temp), M_size))
        DO_mvp = np.zeros(( len(PRES_temp), M_size))
        TEMP_mvp2 = np.zeros(( len(PRES_temp), M_size))
        SUNA_mvp = np.zeros(( len(PRES_temp), M_size))
        FLUO_mvp = np.zeros(( len(PRES_temp), M_size))
        TURB_mvp = np.zeros(( len(PRES_temp), M_size))
        PH_mvp = np.zeros(( len(PRES_temp), M_size))
        SALT_mvp = np.zeros(( len(PRES_temp), M_size))
        TIME_mvp = np.zeros(( len(PRES_temp), M_size))
        LAT_mvp = np.zeros(( len(PRES_temp), M_size))
        LON_mvp = np.zeros(( len(PRES_temp), M_size))
        PRES_mvp[:] = np.nan
        SOUNDVEL_mvp[:] = np.nan
        COND_mvp[:] = np.nan
        TEMP_mvp[:] = np.nan
        DO_mvp[:] = np.nan
        TEMP_mvp2[:] = np.nan
        SUNA_mvp[:] = np.nan
        FLUO_mvp[:] = np.nan
        TURB_mvp[:] = np.nan
        PH_mvp[:] = np.nan
        SALT_mvp[:] = np.nan
        TIME_mvp[:] = np.nan
        LAT_mvp[:] = np.nan
        LON_mvp[:] = np.nan

        del M_size

        for i in range(len(PRES_temp)):
            PRES_mvp[i,0:len(PRES_temp[i])] = PRES_temp[i]
            SOUNDVEL_mvp[i,0:len(SOUNDVEL_temp[i])] = SOUNDVEL_temp[i]
            COND_mvp[i,0:len(COND_temp[i])] = COND_temp[i]
            TEMP_mvp[i,0:len(TEMP_temp[i])] = TEMP_temp[i]
            DO_mvp[i,0:len(DO_temp[i])] = DO_temp[i]
            TEMP_mvp2[i,0:len(TEMP2_temp[i])] = TEMP2_temp[i]
            SUNA_mvp[i,0:len(SUNA_temp[i])] = SUNA_temp[i]
            FLUO_mvp[i,0:len(FLUO_temp[i])] = FLUO_temp[i]
            TURB_mvp[i,0:len(TURB_temp[i])] = TURB_temp[i]
            PH_mvp[i,0:len(PH_temp[i])] = PH_temp[i]
            SALT_mvp[i,0:len(SALT_temp[i])] = SALT_temp[i]
            TIME_mvp[i,0:len(TIME_mvp_temp[i])] = TIME_mvp_temp[i]
            LAT_mvp[i,0:len(PRES_temp[i])] = LAT_temp[i]
            LON_mvp[i,0:len(PRES_temp[i])] = LON_temp[i]


        self.PRES_mvp = np.concatenate((self.PRES_mvp, PRES_mvp), axis=0)
        self.SOUNDVEL_mvp = np.concatenate((self.SOUNDVEL_mvp, SOUNDVEL_mvp), axis=0)
        self.COND_mvp = np.concatenate((self.COND_mvp, COND_mvp), axis=0)
        self.TEMP_mvp = np.concatenate((self.TEMP_mvp, TEMP_mvp), axis=0)
        self.DO_mvp = np.concatenate((self.DO_mvp, DO_mvp), axis=0)
        self.TEMP2_mvp = np.concatenate((self.TEMP2_mvp, TEMP_mvp2), axis=0)
        self.SUNA_mvp = np.concatenate((self.SUNA_mvp, SUNA_mvp), axis=0)
        self.FLUO_mvp = np.concatenate((self.FLUO_mvp, FLUO_mvp), axis=0)
        self.TURB_mvp = np.concatenate((self.TURB_mvp, TURB_mvp), axis=0)
        self.PH_mvp = np.concatenate((self.PH_mvp, PH_mvp), axis=0)
        self.SALT_mvp = np.concatenate((self.SALT_mvp, SALT_mvp), axis=0)
        self.TIME_mvp = np.concatenate((self.TIME_mvp, TIME_mvp), axis=0)
        self.Lat_mvp = np.concatenate((self.Lat_mvp, LAT_mvp), axis=0)
        self.Lon_mvp = np.concatenate((self.Lon_mvp, LON_mvp), axis=0)

        self.DATETIME_mvp.extend(DATETIME_mvp)
        self.DIR.extend(DIR)
        self.label_mvp.extend(Label_mvp)
    
        del PRES_temp, SOUNDVEL_temp, DO_temp, TEMP2_temp, SUNA_temp, FLUO_temp, TURB_temp, PH_temp, COND_temp, TEMP_temp, SALT_temp, TIME_mvp_temp, LAT_temp, LON_temp        

        print('MVP data loaded successfully.')
        self.mvp = True
        self.convert_DO_to_umolL()


    def load_ctd_data(self,data_path_ctd,format='ncdf'):
        """
        Load CTD data from .nc files in the data_path_ctd folder.
        Fills the object attributes with data matrices and associated metadata.
        Args:
            data_path_ctd (str): Path to the folder containing CTD files.
        """

        if format=='ncdf':
            list_of_ctd_files = sorted(filter(os.path.isfile,\
                            glob.glob(data_path_ctd +'*.nc')))
            
            print('Found ' + str(len(list_of_ctd_files)) + ' CTD files in the directory: ' + data_path_ctd)





            # keys: ['scan', 'timeJ', 'timeQ', 'LATITUDE', 'LONGITUDE', 'PRES', 'TEMP', 'CNDC', 'descentrate', 'flECO-AFL', 'v1', 'wetCDOM', 'v0', 'turbWETntu0', 'v5', 'CStarTr0', 'CStarAt0', 'oxygen_ml_L', 'oxsolML/L', 'v2', 'flag', 'timeS']
            LAT_ctd_temp = []
            LON_ctd_temp = []
            PRES_ctd_temp = []
            TEMP_ctd_temp = []
            COND_ctd_temp = []
            TURB_ctd_temp = []
            oxy_ctd_temp = []
            FLUO_ctd_temp = []
            CDOM_ctd_temp = []
            DATETIME_ctd = []
            SALT_ctd_temp = []

            for f in list_of_ctd_files:
                nc = xr.open_dataset(f)
                PRES_ctd_temp.append(nc['PRES'].values[0])
                PRES_ctd_temp.append(nc['PRES'].values[1])
                TEMP_ctd_temp.append(nc['TEMP'].values[0])
                TEMP_ctd_temp.append(nc['TEMP'].values[1])
                COND_ctd_temp.append(nc['COND'].values[0])
                COND_ctd_temp.append(nc['COND'].values[1])
                SALT_ctd_temp.append(nc['SAL'].values[0])
                SALT_ctd_temp.append(nc['SAL'].values[1])
                TURB_ctd_temp.append(nc['TURB'].values[0])
                TURB_ctd_temp.append(nc['TURB'].values[1])
                oxy_ctd_temp.append(nc['OXY'].values[0])
                oxy_ctd_temp.append(nc['OXY'].values[1])
                FLUO_ctd_temp.append(nc['FLUO'].values[0])
                FLUO_ctd_temp.append(nc['FLUO'].values[1])
                CDOM_ctd_temp.append(nc['CDOM'].values[0])
                CDOM_ctd_temp.append(nc['CDOM'].values[1])
                LAT_ctd_temp.append(nc['LATITUDE'].values[0])
                LAT_ctd_temp.append(nc['LATITUDE'].values[1])
                LON_ctd_temp.append(nc['LONGITUDE'].values[0])
                LON_ctd_temp.append(nc['LONGITUDE'].values[1])
                DATETIME_ctd.append(nc['profile_time'].values[0])

                nc.close()

        elif format=='cnv':
            list_of_ctd_files = sorted(filter(os.path.isfile,\
                            glob.glob(data_path_ctd +'*.cnv')))
            
            print('Found ' + str(len(list_of_ctd_files)) + ' CTD files in the directory: ' + data_path_ctd)

            LAT_ctd_temp = []
            LON_ctd_temp = []
            PRES_ctd_temp = []
            TEMP_ctd_temp = []
            COND_ctd_temp = []
            TURB_ctd_temp = []
            oxy_ctd_temp = []
            FLUO_ctd_temp = []
            CDOM_ctd_temp = []
            DATETIME_ctd = []
            SALT_ctd_temp = []

            for f in list_of_ctd_files:
                ctd_data,nmea_datetime = mvp.read_cnv(f)
                PRES_ctd_temp.append(ctd_data['prDM'].values)
                TEMP_ctd_temp.append(ctd_data['t090C'].values)
                COND_ctd_temp.append(ctd_data['c0mS/cm'].values)
                SALT_ctd_temp.append(ctd_data['sal00'].values)
                # TURB_ctd_temp.append(ctd_data['TURB'].values)
                oxy_ctd_temp.append(ctd_data['sbeox0ML/L'].values)
                FLUO_ctd_temp.append(ctd_data['flECO-AFL'].values)
                # CDOM_ctd_temp.append(ctd_data['CDOM'].values)
                LAT_ctd_temp.append(ctd_data['latitude'].values)
                LON_ctd_temp.append(ctd_data['longitude'].values)
                DATETIME_ctd.append(nmea_datetime)


            # complete shorter arrays with NaN to match the length of the longest array
            max_length = max(len(arr) for arr in PRES_ctd_temp)
            for i, arr in enumerate(PRES_ctd_temp):
                if len(arr) < max_length:
                    PRES_ctd_temp[i] = np.pad(arr, (0, max_length - len(arr)), constant_values=np.nan)
                    TEMP_ctd_temp[i] = np.pad(TEMP_ctd_temp[i], (0, max_length - len(TEMP_ctd_temp[i])), constant_values=np.nan)
                    COND_ctd_temp[i] = np.pad(COND_ctd_temp[i], (0, max_length - len(COND_ctd_temp[i])), constant_values=np.nan)
                    SALT_ctd_temp[i] = np.pad(SALT_ctd_temp[i], (0, max_length - len(SALT_ctd_temp[i])), constant_values=np.nan)
                    # TURB_ctd_temp[i] = np.pad(TURB_ctd_temp[i], (0, max_length - len(TURB_ctd_temp[i])), constant_values=np.nan)
                    oxy_ctd_temp[i] = np.pad(oxy_ctd_temp[i], (0, max_length - len(oxy_ctd_temp[i])), constant_values=np.nan)
                    FLUO_ctd_temp[i] = np.pad(FLUO_ctd_temp[i], (0, max_length - len(FLUO_ctd_temp[i])), constant_values=np.nan)
                    # CDOM_ctd_temp[i] = np.pad(CDOM_ctd_temp[i], (0, max_length - len(CDOM_ctd_temp[i])), constant_values=np.nan)
                    LAT_ctd_temp[i] = np.pad(LAT_ctd_temp[i], (0, max_length - len(LAT_ctd_temp[i])), constant_values=np.nan)
                    LON_ctd_temp[i] = np.pad(LON_ctd_temp[i], (0, max_length - len(LON_ctd_temp[i])), constant_values=np.nan)

        self.PRES_ctd = np.array(PRES_ctd_temp)
        self.TEMP_ctd = np.array(TEMP_ctd_temp)
        self.COND_ctd = np.array(COND_ctd_temp)
        self.SALT_ctd = np.array(SALT_ctd_temp)
        self.TURB_ctd = np.array(TURB_ctd_temp)
        self.oxy_ctd = np.array(oxy_ctd_temp)*44.661 # convert from ml/L to umol/L
        self.FLUO_ctd = np.array(FLUO_ctd_temp)
        self.CDOM_ctd = np.array(CDOM_ctd_temp)
        self.LAT_ctd = np.array(LAT_ctd_temp)
        self.LON_ctd = np.array(LON_ctd_temp)
        self.DATETIME_ctd = np.array(DATETIME_ctd)
        self.TIME_ctd = np.array([(np.datetime64(dt) - np.datetime64(self.date_ref)) / np.timedelta64(1, 'D') for dt in self.DATETIME_ctd])


        print('CTD data loaded successfully.')
        self.ctd = True


    def load_gps_from_ncdf(self, gps_path): 
        """
        Load GPS data from a .nc (or .nav) file in the gps_path.
        Fills the object attributes with GPS data and associated metadata.
        Args:
            gps_path (str): Path to the .nc or .nav file containing GPS data. Must have the variables 'time' 'lat' 'long'
        """

        day0 = self.DATETIME_mvp[0].strftime('%Y%m%d') 
        day1 = self.DATETIME_mvp[-1].strftime('%Y%m%d') 
        file = glob.glob(gps_path +day0 + '*.nav') 
        ds0 = xr.open_dataset(file[0]) 

        gpstime = ds0['time'].values 
        lat_gps = ds0['lat'].values 
        lon_gps = ds0['long'].values 

        if day0 != day1: 

            for day in range(int(day0)+1, int(day1)+1):
                file1 = glob.glob(gps_path +str(day) + '*.nav') 
                ds1 = xr.open_dataset(file1[0]) 

                gpstime = np.concatenate((gpstime, ds1['time'].values)) 
                lat_gps = np.concatenate((lat_gps, ds1['lat'].values)) 
                lon_gps = np.concatenate((lon_gps, ds1['long'].values)) 

        self.lat_gps = lat_gps
        self.lon_gps = lon_gps
        self.gps_time = gpstime

        # transform gpstime from datetime to julian days since 1/1/1950 
        gpstime = (gpstime - np.datetime64('1950-01-01T00:00:00')) / np.timedelta64(1, 'D') 


        # change self.Lat_mvp and self.Lon_mvp to the GPS coordinates using gpstime to match 
        self.Lon_mvp = np.zeros(( self.PRES_mvp.shape[0], self.PRES_mvp.shape[1])) 
        self.Lat_mvp = np.zeros(( self.PRES_mvp.shape[0], self.PRES_mvp.shape[1])) 


        # Tous les temps MVP sous forme d'un vecteur
        time_flat = self.TIME_mvp.ravel()

        # Position d'insertion dans gpstime (trié)
        idx = np.searchsorted(gpstime, time_flat)

        # Gestion des bords
        idx = np.clip(idx, 1, len(gpstime)-1)

        left = idx - 1
        right = idx

        # Choix du voisin le plus proche
        nearest = np.where(
            np.abs(time_flat - gpstime[left]) <= np.abs(gpstime[right] - time_flat),
            left,
            right
        )
        mask_nan = np.isnan(time_flat)
        lat_ = lat_gps[nearest]
        lat_[mask_nan] = np.nan
        lon_ = lon_gps[nearest]
        lon_[mask_nan] = np.nan



        # Reconstruction des tableaux
        self.Lat_mvp = lat_.reshape(self.PRES_mvp.shape)
        self.Lon_mvp = lon_.reshape(self.PRES_mvp.shape)

        print('GPS data loaded successfully.')
        self.GPS = True

    def load_GPS_from_csv(self, gps_path):
        """
        Load GPS data from a .csv file in the gps_path.
        Fills the object attributes with GPS data and associated metadata.
        Args:
            gps_path (str): Path to the .csv file containing GPS data.
        """
        self.gps_path = gps_path
        gps_data = pd.read_csv(gps_path)
        self.GPS_TIME = gps_data['time'].values
        self.GPS_LAT = gps_data['latitude'].values
        self.GPS_LON = gps_data['longitude'].values
        print('GPS data loaded successfully.')
        self.gps = True

        self.Lon_mvp = np.zeros(( self.PRES_mvp.shape[0], self.PRES_mvp.shape[1]))
        self.Lat_mvp = np.zeros(( self.PRES_mvp.shape[0], self.PRES_mvp.shape[1]))

        for i in range(self.PRES_mvp.shape[0]):
            self.Lon_mvp[i,:] = np.interp(self.TIME_mvp[i,:], self.GPS_TIME, self.GPS_LON.astype(float))
            self.Lat_mvp[i,:] = np.interp(self.TIME_mvp[i,:], self.GPS_TIME, self.GPS_LAT.astype(float))     

        self.GPS = True     

    def compute_waterflow(self,horizontal_speed=2,corr=False):
        """
        Compute the water flow speed (u,v) from the speed of the profiles.
        Args:
            horizontal_speed (float): Horizontal speed of the boat in m/s.
        """
        
        if corr:
            SPEED_MVP = []
            for i in range(len(self.PRES_mvp_corr)):
                SPEED_MVP.append(np.sqrt(np.gradient(self.PRES_mvp_corr[i], 1/self.freq_echant)**2+ horizontal_speed**2))
            self.SPEED_mvp_corr = {i: SPEED_MVP[i] for i in range(len(SPEED_MVP))}
        else:
            SPEED_MVP = np.zeros((self.PRES_mvp.shape[0], self.PRES_mvp.shape[1]))
            for i in range(self.PRES_mvp.shape[0]):
                SPEED_MVP[i,:] = np.sqrt(np.gradient(self.PRES_mvp[i,:], 1/self.freq_echant)**2+ horizontal_speed**2)

            self.SPEED_mvp = SPEED_MVP
        print('Water flow speed computed successfully.')
        self.speed = True

    def print_profile_metadata(self):
        """
        Print main metadata (date, position, number of samples) for each loaded MVP and CTD profile.
        """

        if self.mvp:
            print('MVP data:')
            print('Number of profiles: ' + str(len(self.DATETIME_mvp)))
            for i in range(0,len(self.DATETIME_mvp)):
                print(f"  Profil down {2*i} - Profil up {2*i+1} - Latitude: {self.LAT_mvp[2*i,0]:.5f}, Longitude: {self.LON_mvp[2*i,0]:.5f}, Date/Heure: {self.DATETIME_mvp[i]}")

        if self.ctd:
            print('CTD data:')
            print('Number of profiles: ' + str(len(self.DATETIME_ctd)))
            for i in range(0,len(self.DATETIME_ctd)):
                print(f"  Profil {i} - Latitude: {self.LAT_ctd[i,0]:.5f}, Longitude: {self.LON_ctd[i,0]:.5f}, Date/Heure: {self.DATETIME_ctd[i]}")


    def keep_selected_profiles(self, id_mvp, id_ctd=None):
        """
        Keep only the selected MVP and CTD profiles in the object attributes.
        Args:
            id_mvp (list): Indices of MVP profiles to keep.
            id_ctd (list): Indices of CTD profiles to keep (optional).
        """
        

        # Make a list of all id to keep for MVP profiles
        l_id = []
        l_id2 = []
        for i in id_mvp:
            l_id.append(i)
            if i%2==0:
                l_id2.append(i//2) 


  
        # Keep only the selected profiles

        if self.mvp:

            self.PRES_mvp = self.PRES_mvp[l_id,:]
            self.SOUNDVEL_mvp = self.SOUNDVEL_mvp[l_id,:]
            self.COND_mvp = self.COND_mvp[l_id,:]
            self.TEMP_mvp = self.TEMP_mvp[l_id,:]
            self.DO_mvp = self.DO_mvp[l_id,:]
            self.TEMP2_mvp = self.TEMP2_mvp[l_id,:]
            self.SUNA_mvp = self.SUNA_mvp[l_id,:]
            self.FLUO_mvp = self.FLUO_mvp[l_id,:]
            self.TURB_mvp = self.TURB_mvp[l_id,:]
            self.PH_mvp = self.PH_mvp[l_id,:]
            self.SALT_mvp = self.SALT_mvp[l_id,:]
            self.TIME_mvp = self.TIME_mvp[l_id,:]
            try:
                self.Lat_mvp = self.Lat_mvp[l_id,:]
                self.Lon_mvp = self.Lon_mvp[l_id,:]
            except:
                self.LAT_mvp = self.LAT_mvp[l_id]
                self.LON_mvp = self.LON_mvp[l_id]
            self.DATETIME_mvp = np.array(self.DATETIME_mvp)[l_id2]
            self.DIR = np.array(self.DIR)[l_id]
            self.label_mvp = np.array(self.label_mvp)[l_id]

        if self.ctd and id_ctd != None:

            l_id = id_ctd

            self.PRES_ctd = self.PRES_ctd[l_id,:]
            self.TEMP_ctd = self.TEMP_ctd[l_id,:]
            self.SALT_ctd = self.SALT_ctd[l_id,:]
            self.COND_ctd = self.COND_ctd[l_id,:]
            # self.TURB_ctd = self.TURB_ctd[l_id]
            self.oxy_ctd = self.oxy_ctd[l_id,:]
            self.FLUO_ctd = self.FLUO_ctd[l_id,:]
            # self.CDOM_ctd = self.CDOM_ctd[l_id]
            self.LAT_ctd = self.LAT_ctd[l_id,:]
            self.LON_ctd = self.LON_ctd[l_id,:]
            self.DATETIME_ctd = np.array(self.DATETIME_ctd)[l_id]


    def plot_vertical_speed(self,id=None,window=20):
        """
        plot profile of vertical speed
        Args:
            id (int): index of the profile to plot, if None, plot the mean profile

        """
            
        if self.mvp==False:
            print('No MVP data loaded.')
            return
    
        if id==None:
            v_z_down = np.gradient(self.PRES_mvp[0::2], 1/self.freq_echant,axis=1)
            v_z_up = np.gradient(self.PRES_mvp[1::2], 1/self.freq_echant,axis=1)

            # smooth speed
            for i in range(v_z_down.shape[0]):
                v_z_down[i,:] = np.convolve(v_z_down[i,:], np.ones(2*window+1)/(2*window+1), mode='same')
                v_z_up[i,:] = np.convolve(v_z_up[i,:], np.ones(2*window+1)/(2*window+1), mode='same')
            
            # take mean profile
            p_grid = np.arange(0, 1001, 1)  # 0 à 1000 dbar, pas de 1

            n_down = v_z_down.shape[0]
            n_up   = v_z_up.shape[0]

            vz_down_interp = np.full((n_down, len(p_grid)), np.nan)
            vz_up_interp   = np.full((n_up,   len(p_grid)), np.nan)

            for i in range(n_down):
                pres = self.PRES_mvp[i*2, :]
                vz   = v_z_down[i, :]   # vitesse du profil i (avant nanmean)
                
                mask = ~np.isnan(pres) & ~np.isnan(vz)
                if mask.sum() > 1:
                    vz_down_interp[i, :] = np.interp(p_grid, pres[mask], vz[mask],
                                                    left=np.nan, right=np.nan)

            for i in range(n_up):
                pres = self.PRES_mvp[i*2+1, :]
                vz   = v_z_up[i, :]
                mask = ~np.isnan(pres) & ~np.isnan(vz)
                if mask.sum() > 1:
                    p_valid  = pres[mask]
                    vz_valid = vz[mask]
                    
                    # Trier par pression croissante (crucial pour les profils up !)
                    sort_idx = np.argsort(p_valid)
                    p_valid  = p_valid[sort_idx]
                    vz_valid = vz_valid[sort_idx]
                    
                    vz_up_interp[i, :] = np.interp(p_grid, p_valid, vz_valid,
                                                    left=np.nan, right=np.nan)
            v_z_down = np.nanmean(vz_down_interp, axis=0)
            v_z_up   = np.nanmean(vz_up_interp,   axis=0)
            pres_d = p_grid
            pres_u = p_grid
            self.v_z_down = v_z_down
            self.v_z_up = v_z_up



        else:

            v_z_down = np.gradient(self.PRES_mvp[id,:], 1/self.freq_echant)
            v_z_up = np.gradient(self.PRES_mvp[id+1,:], 1/self.freq_echant)

            # smooth speed
            self.v_z_down = np.convolve(v_z_down, np.ones(2*window+1)/(2*window+1), mode='same')
            self.v_z_up = np.convolve(v_z_up, np.ones(2*window+1)/(2*window+1), mode='same')
            pres_d = self.PRES_mvp[id]
            pres_u = self.PRES_mvp[id+1]




        plt.figure()

        plt.plot(v_z_down,pres_d, label='down')
        plt.plot(v_z_up,pres_u, label='up')

        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid()
        plt.xlabel('Vertical speed, m/s')
        plt.ylabel('Pressure, dbar')
        plt.title('Vertical speed profiles')
        plt.legend()


    def plot_profile_map(self):
        """
        Plot a map of the start locations of each profile (MVP and CTD),
        with a land/ocean background and coastlines using cartopy.
        The map is automatically zoomed to the profile area (no excessive margin).
        """

        fig = plt.figure(figsize=(8, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_title('Carte des profils (début de plongée)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black', facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, zorder=0, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        colors = plt.cm.tab10.colors

        # MVP
        if hasattr(self, 'LAT_mvp') and hasattr(self, 'LON_mvp'):
            l_lon,l_lat =[],[]
            put_label = True
            c = 0
            for i in range(0,self.LAT_mvp.shape[0],2):
                if i>0:
                    if self.label_mvp[i] == self.label_mvp[i-1]:
                        put_label = False
                    else:
                        put_label = True
                        c+=1

                lat = self.LAT_mvp[i,0] if self.LAT_mvp.ndim == 2 else  self.LAT_mvp[i]
                lon = self.LON_mvp[i,0] if self.LON_mvp.ndim == 2 else  self.LON_mvp[i]
                l_lon.append(lon)
                l_lat.append(lat)
                ax.scatter(lon, lat, color=colors[c % len(colors)], marker='o', label='MVP '+self.label_mvp[i] if put_label else "", transform=ccrs.PlateCarree())

        # CTD
        if hasattr(self, 'LAT_ctd') and hasattr(self, 'LON_ctd'):
            for i in range(0,self.LAT_ctd.shape[0],2):
                lat = self.LAT_ctd[i,0] if self.LAT_ctd.ndim == 2 else self.LAT_ctd[i]
                lon = self.LON_ctd[i,0] if self.LON_ctd.ndim == 2 else self.LON_ctd[i]
                ax.scatter(lon, lat, color='red', marker='^', label='CTD' if i==0 else "", transform=ccrs.PlateCarree())


        ax.plot(l_lon, l_lat, color='grey', linestyle='-', transform=ccrs.PlateCarree())

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # ax.legend(by_label.values(), by_label.keys())
        plt.show()

    def plot_TSprofile(self, id_mvp,id_ctd=None,correction=False):
        """
        Plot temperature and salinity profiles versus pressure for a given profile (MVP and CTD).
        Args:
            id_mvp (int): Index of the MVP profile to plot.
            id_ctd (int, optional): Index of the CTD profile to plot (default: same as id_mvp).
            correction (bool): If True, plot corrected profiles.
        """

        if id_ctd is None:
            id_ctd = id_mvp
            
        
       
        plt.figure()
        if self.mvp:
            if correction:
                plt.plot(self.TEMP_mvp_corr[id_mvp],self.PRES_mvp_corr[id_mvp],label='MVP down corrected')
                plt.plot(self.TEMP_mvp_corr[id_mvp+1],self.PRES_mvp_corr[id_mvp+1],label='MVP up corrected')             
            else:
                plt.plot(self.TEMP_mvp[id_mvp],self.PRES_mvp[id_mvp],label='MVP down')
                plt.plot(self.TEMP_mvp[id_mvp+1],self.PRES_mvp[id_mvp+1],label='MVP up')
        if self.ctd:
            plt.plot(self.TEMP_ctd[id_ctd],self.PRES_ctd[id_ctd],label='CTD down')
            plt.plot(self.TEMP_ctd[id_ctd+1],self.PRES_ctd[id_ctd+1],label='CTD up')
        plt.legend()    
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Temperature, C')    
        plt.ylabel('Pressure, dbar')


        plt.figure()
        if self.mvp:
            if correction:
                plt.plot(self.SALT_mvp_corr[id_mvp],self.PRES_mvp_corr[id_mvp],label='MVP down corrected')
                plt.plot(self.SALT_mvp_corr[id_mvp+1],self.PRES_mvp_corr[id_mvp+1],label='MVP up corrected')
            else:
                plt.plot(self.SALT_mvp[id_mvp],self.PRES_mvp[id_mvp],label='MVP down')
                plt.plot(self.SALT_mvp[id_mvp+1],self.PRES_mvp[id_mvp+1],label='MVP up')
        if self.ctd:
            plt.plot(self.SALT_ctd[id_ctd],self.PRES_ctd[id_ctd],label='CTD down')
            plt.plot(self.SALT_ctd[id_ctd+1],self.PRES_ctd[id_ctd+1],label='CTD up')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Salinity, psu')
        plt.ylabel('Pressure, dbar')
    
    def plot_BGCprofile(self, id_mvp,id_ctd=None,correction=False):
        """
        Plot raw biogeochemical profiles (O2, turbidity, fluorescence) for a given profile (MVP and CTD).
        Args:
            id_mvp (int): Index of the MVP profile to plot.
            id_ctd (int, optional): Index of the CTD profile to plot (default: same as id_mvp).
        """
    
        if id_ctd is None:
            id_ctd = id_mvp

        

        plt.figure()
        if self.mvp:
            if self.corrected:
                plt.plot(self.oxy[id_mvp], self.PRES_mvp_corr[id_mvp],label='MVP down corrected')
                plt.plot(self.oxy[id_mvp+1], self.PRES_mvp_corr[id_mvp+1],label='MVP up corrected')
            else:
                plt.plot(self.oxy[id_mvp], self.PRES_mvp[id_mvp],label='MVP down')
                plt.plot(self.oxy[id_mvp+1], self.PRES_mvp[id_mvp+1],label='MVP up')
        if self.ctd:
            plt.plot(self.oxy_ctd[id_ctd],self.PRES_ctd[id_ctd],label='CTD down')
            plt.plot(self.oxy_ctd[id_ctd+1],self.PRES_ctd[id_ctd+1],label='CTD up')
        plt.legend()    
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel(' Oxygen, umol/L')    
        plt.ylabel('Pressure, dbar')


        plt.figure()
        if self.mvp:
            plt.plot(self.TURB_mvp[id_mvp],self.PRES_mvp[id_mvp],label='MVP down')
            plt.plot(self.TURB_mvp[id_mvp+1],self.PRES_mvp[id_mvp+1],label='MVP up')
        if self.ctd:
            plt.plot(self.TURB_ctd[id_ctd],self.PRES_ctd[id_ctd],label='CTD down')
            plt.plot(self.TURB_ctd[id_ctd+1],self.PRES_ctd[id_ctd+1],label='CTD up')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Turbidity, NTU')
        plt.ylabel('Pressure, dbar')

        plt.figure()
        if self.mvp:
            plt.plot(self.FLUO_mvp[id_mvp],self.PRES_mvp[id_mvp],label='MVP down')
            plt.plot(self.FLUO_mvp[id_mvp+1],self.PRES_mvp[id_mvp+1],label='MVP up')
        if self.ctd:
            plt.plot(self.FLUO_ctd[id_ctd],self.PRES_ctd[id_ctd],label='CTD down')
            plt.plot(self.FLUO_ctd[id_ctd+1],self.PRES_ctd[id_ctd+1],label='CTD up')
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid()
        plt.xlabel('Fluorescence, ug/L')
        plt.ylabel('Pressure, dbar')




        if self.SUNA:
            plt.figure()
            plt.plot(self.NO3_suna[id_mvp], self.PRES_suna[id_mvp], label="downcast")
            plt.plot(self.NO3_suna[id_mvp+1], self.PRES_suna[id_mvp+1], label="upcast")
            plt.gca().invert_yaxis()
            plt.legend()
            plt.xlabel("NO3 (µmol/L)")
            plt.ylabel("Pressure (dbar)")
            plt.grid()


    def plot_diagramTS(self,id_mvp=None,id_ctd=None,correction=False,save=None):
        """
        Plot the TS diagram (Salinity vs Temperature) for one or more profiles, with isopycnals.
        Args:
            id_mvp (int, optional): Index of the MVP profile to plot, or None for all profiles.
            id_ctd (int, optional): Index of the CTD profile to plot, or None for all profiles.
            correction (bool): If True, plot corrected profiles.
        """
    

       
    
        plt.figure()
        if id_mvp is not None:
            if id_ctd == None:
                id_ctd = id_mvp

            if self.mvp:
                if correction:
                    plt.plot(self.SALT_mvp_corr[id_mvp],self.TEMP_mvp_corr[id_mvp],label='MVP down corrected',linestyle='', marker='.')
                    plt.plot(self.SALT_mvp_corr[id_mvp+1],self.TEMP_mvp_corr[id_mvp+1],label='MVP up corrected',linestyle='', marker='.')
                else:
                    plt.plot(self.SALT_mvp[id_mvp],self.TEMP_mvp[id_mvp],label='MVP down',linestyle='', marker='.')
                    plt.plot(self.SALT_mvp[id_mvp+1],self.TEMP_mvp[id_mvp+1],label='MVP up',linestyle='', marker='.')
            if self.ctd:
                plt.plot(self.SALT_ctd[id_ctd],self.TEMP_ctd[id_ctd],label='CTD down', linestyle='', marker='.')
                plt.plot(self.SALT_ctd[id_ctd+1],self.TEMP_ctd[id_ctd+1],label='CTD up', linestyle='', marker='.')

        else:
            if self.mvp:
                if correction:
                    plt.plot(self.SALT_mvp_corr[0],self.TEMP_mvp_corr[0],linestyle='',color='red', marker='.',label='MVP down corrected')
                    plt.plot(self.SALT_mvp_corr[1],self.TEMP_mvp_corr[1],linestyle='',color='blue', marker='.',label='MVP up corrected')
                    for i in range(2,len(self.PRES_mvp),2):
                        plt.plot(self.SALT_mvp_corr[i],self.TEMP_mvp_corr[i],linestyle='',color='red', marker='.')
                        plt.plot(self.SALT_mvp_corr[i+1],self.TEMP_mvp_corr[i+1],linestyle='',color='blue', marker='.')
                else:
                    plt.plot(self.SALT_mvp[0],self.TEMP_mvp[0],linestyle='',color='red', marker='.',label='MVP down')
                    plt.plot(self.SALT_mvp[1],self.TEMP_mvp[1],linestyle='',color='blue', marker='.',label='MVP up')
                    for i in range(2,len(self.PRES_mvp),2):
                        plt.plot(self.SALT_mvp[i],self.TEMP_mvp[i],linestyle='',color='red', marker='.')
                        plt.plot(self.SALT_mvp[i+1],self.TEMP_mvp[i+1],linestyle='',color='blue', marker='.')
            if self.ctd:
                plt.plot(self.SALT_ctd[0],self.TEMP_ctd[0],color='green', linestyle='', marker='.',label='CTD down')
                plt.plot(self.SALT_ctd[1],self.TEMP_ctd[1],color='orange', linestyle='', marker='.',label='CTD up')
                for i in range(2,len(self.PRES_ctd),2):
                    plt.plot(self.SALT_ctd[i],self.TEMP_ctd[i],color='green', linestyle='', marker='.')
                    plt.plot(self.SALT_ctd[i+1],self.TEMP_ctd[i+1],color='orange', linestyle='', marker='.')


        s_lim = plt.xlim()
        t_lim = plt.ylim()
        SA = np.linspace(s_lim[0], s_lim[1], 100)  # Absolute Salinity [g/kg]
        CT = np.linspace(t_lim[0], t_lim[1], 100)
        SA_grid, CT_grid = np.meshgrid(SA, CT)
        # Calcul de la densité potentielle sigma0 (kg/m³ - 1000)
        sigma0 = gsw.sigma0(SA_grid, CT_grid)
        # Dessiner les contours (les isopycnes)
        contour_plot = plt.contour(SA_grid, CT_grid, sigma0, colors='k', linestyles='dotted')
        # Ajouter les étiquettes (les chiffres) le long des contours
        plt.clabel(contour_plot, inline=True, fontsize=10, fmt='%1.1f')

        plt.legend()  
        plt.xlabel('Salinity, psu') 
        plt.ylabel('Temperature, C')

        if save is not None:
            plt.savefig(save)

    def stat_compar(self,id_mvp=[],id_ctd=None,num_sample=5000,cond=False,speed=False,correction=False):
        """
        Statistically compare MVP and CTD profiles (temperature and salinity),
        print statistics and interpolated differences.
        Args:
            id (list): Indices of profiles to compare (all if empty).
            num_sample (int): Number of pressure levels for interpolation.
        """

        if not self.mvp or not self.ctd:
            raise ValueError("MVP or CTD data not loaded.")
        
        if id_mvp == []:
            id_mvp = list(range(0, self.PRES_mvp.shape[0]))
        if id_ctd is None:
            if len(self.PRES_ctd) == len(self.PRES_mvp)//2:
                id_ctd = list(range(0, self.PRES_ctd.shape[0]))
                id_ctd1 = list(range(0, self.PRES_ctd.shape[0]))
            elif len(self.PRES_ctd) == len(self.PRES_mvp):
                id_ctd = id_mvp

        if len(id_mvp) != len(id_ctd):
            if not len(id_mvp)//2 == len(id_ctd):
                raise ValueError("Length of id_mvp and id_ctd must be the same or id_ctd must be half the length of id_mvp.")
        else:
            # keep only down profiles
            id_ctd1 = [id_ctd[i] for i in range(len(id_ctd)) if id_ctd[i]%2 == 0]

        if correction:
            Pres = self.PRES_mvp_corr
            Temp = self.TEMP_mvp_corr
            Salt = self.SALT_mvp_corr
            Cond = self.COND_mvp_corr
            oxy  = self.oxy

        else:
            Pres = self.PRES_mvp
            Temp = self.TEMP_mvp
            Salt = self.SALT_mvp
            Cond = self.COND_mvp
            oxy = self.DO_mvp

        # Interpolate MVP and CTD data to match pressure levels
        pmin = 5
        pmax = 1000
        pressure_grid = np.linspace(pmin, pmax, num_sample)

        if correction:
            TEMP_mvp_interp = []
            SALT_mvp_interp = []
            oxy_mvp_interp = []
            COND_mvp_interp = []
            for i in id_mvp:
                TEMP_mvp_interp.append(np.interp(pressure_grid, Pres[i], Temp[i]))
                SALT_mvp_interp.append(np.interp(pressure_grid, Pres[i], Salt[i]))
                oxy_mvp_interp.append(np.interp(pressure_grid, Pres[i], oxy[i]))
                COND_mvp_interp.append(np.interp(pressure_grid, Pres[i], Cond[i]))

        else:
            TEMP_mvp_interp = mvp.vertical_interp(Pres[id_mvp],Temp[id_mvp], pressure_grid)
            SALT_mvp_interp = mvp.vertical_interp(Pres[id_mvp], Salt[id_mvp], pressure_grid)
            oxy_mvp_interp = mvp.vertical_interp(Pres[id_mvp], oxy[id_mvp], pressure_grid)
            COND_mvp_interp = mvp.vertical_interp(Pres[id_mvp], Cond[id_mvp], pressure_grid) 

            
 
        TEMP_ctd_interp = mvp.vertical_interp(self.PRES_ctd[id_ctd1,:],self.TEMP_ctd[id_ctd1,:], pressure_grid)
        SALT_ctd_interp = mvp.vertical_interp(self.PRES_ctd[id_ctd1,:],self.SALT_ctd[id_ctd1,:], pressure_grid)
        oxy_ctd_interp = mvp.vertical_interp(self.PRES_ctd[id_ctd1,:],self.oxy_ctd[id_ctd1,:], pressure_grid)
        COND_ctd_interp = mvp.vertical_interp(self.PRES_ctd[id_ctd1,:],self.COND_ctd[id_ctd1,:], pressure_grid)

        # differences study between MVP down and CTD profiles

        # Calcul des différences entre les profils interpolés (MVP - CTD)
        diff_temp_down = TEMP_mvp_interp[0::2] - TEMP_ctd_interp
        diff_temp_up = TEMP_mvp_interp[1::2] - TEMP_ctd_interp
        diff_salt_down = SALT_mvp_interp[0::2] - SALT_ctd_interp
        diff_salt_up = SALT_mvp_interp[1::2] - SALT_ctd_interp
        diff_oxy_down = oxy_mvp_interp[0::2] - oxy_ctd_interp
        diff_oxy_up = oxy_mvp_interp[1::2] - oxy_ctd_interp
        diff_cond_down = COND_mvp_interp[0::2] - COND_ctd_interp
        diff_cond_up = COND_mvp_interp[1::2] - COND_ctd_interp


        # Plot mean error vs depth for each variable (down/up)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Compute mean error along profiles (axis=0: profiles, axis=1: depth)
        mean_temp_down = np.nanmean(np.absolute(diff_temp_down), axis=0)
        mean_temp_up =  np.nanmean(np.absolute(diff_temp_up), axis=0)
        mean_salt_down =  np.nanmean(np.absolute(diff_salt_down), axis=0)
        mean_salt_up =  np.nanmean(np.absolute(diff_salt_up), axis=0)
        mean_oxy_down =  np.nanmean(np.absolute(diff_oxy_down), axis=0)
        mean_oxy_up =  np.nanmean(np.absolute(diff_oxy_up), axis=0)
        mean_cond_down =  np.nanmean(np.absolute(diff_cond_down), axis=0)
        mean_cond_up =  np.nanmean(np.absolute(diff_cond_up), axis=0)

        axes[0].plot(mean_temp_down, pressure_grid, label='Down')
        axes[0].plot(mean_temp_up, pressure_grid, label='Up')
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Absolute Mean Error (°C)')
        axes[0].set_ylabel('Pressure (dbar)')
        axes[0].set_title('Temperature Error')
        axes[0].legend()
        axes[0].grid()


        if cond:

            axes[1].plot(mean_cond_down, pressure_grid, label='Down')
            axes[1].plot(mean_cond_up, pressure_grid, label='Up')   
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Absolute Mean Error (S/m)')
            axes[1].set_ylabel('Pressure (dbar)')
            axes[1].set_title('Conductivity Error')
            axes[1].legend()
            axes[1].grid()

        else:

            axes[1].plot(mean_salt_down, pressure_grid, label='Down')
            axes[1].plot(mean_salt_up, pressure_grid, label='Up')
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Absolute Mean Error (psu)')
            axes[1].set_ylabel('Pressure (dbar)')
            axes[1].set_title('Salinity Error')
            axes[1].legend()
            axes[1].grid()
        
        if speed:

            axes[2].plot(self.v_z_down, self.PRES_mvp[0], label='Down')
            axes[2].plot(self.v_z_up, self.PRES_mvp[0], label='Up')
            axes[2].invert_yaxis()
            axes[2].set_xlabel('Vertical Speed (m/s)')
            axes[2].set_ylabel('Pressure (dbar)')
            axes[2].set_title('Vertical Speed')
            axes[2].legend()
            axes[2].grid()


        else:

            axes[2].plot(mean_oxy_down, pressure_grid, label='Down')
            axes[2].plot(mean_oxy_up, pressure_grid, label='Up')
            axes[2].invert_yaxis()
            axes[2].set_xlabel('Absolute Mean Error (umol/L)')
            axes[2].set_ylabel('Pressure (dbar)')
            axes[2].set_title('Oxygen Error')
            axes[2].legend()
            axes[2].grid()

        fig.suptitle('Absolute Mean Error (MVP - CTD) vs Depth')
        fig.tight_layout()
        plt.show()
        

        # Compute RMSE

        rmse_temp_down = np.mean(np.sqrt(np.nanmean(diff_temp_down**2, axis=1)))
        rmse_temp_up = np.mean(np.sqrt(np.nanmean(diff_temp_up**2, axis=1)))
        rmse_salt_down = np.mean(np.sqrt(np.nanmean(diff_salt_down**2, axis=1)))
        rmse_salt_up = np.mean(np.sqrt(np.nanmean(diff_salt_up**2, axis=1)))
        rmse_oxy_down = np.mean(np.sqrt(np.nanmean(diff_oxy_down**2, axis=1)))
        rmse_oxy_up = np.mean(np.sqrt(np.nanmean(diff_oxy_up**2, axis=1)))
        rmse_cond_down = np.mean(np.sqrt(np.nanmean(diff_cond_down**2, axis=1)))
        rmse_cond_up = np.mean(np.sqrt(np.nanmean(diff_cond_up**2, axis=1)))

        # Find index where depth >= 200 dbar; fallback to 0 if not found
        i_200 = 0
        for i in range(len(pressure_grid)):
            if pressure_grid[i] >= 200:
                i_200 = i
                break

        # Slice along depth axis (columns) to keep depths >= 200 dbar
        rmse_temp_down_deep = np.mean(np.sqrt(np.nanmean(diff_temp_down[:, i_200:]**2, axis=1)))
        rmse_temp_up_deep   = np.mean(np.sqrt(np.nanmean(diff_temp_up[:,   i_200:]**2, axis=1)))
        rmse_salt_down_deep = np.mean(np.sqrt(np.nanmean(diff_salt_down[:, i_200:]**2, axis=1)))
        rmse_salt_up_deep   = np.mean(np.sqrt(np.nanmean(diff_salt_up[:,   i_200:]**2, axis=1)))
        rmse_oxy_down_deep   = np.mean(np.sqrt(np.nanmean(diff_oxy_down[:,   i_200:]**2, axis=1)))
        rmse_oxy_up_deep     = np.mean(np.sqrt(np.nanmean(diff_oxy_up[:,     i_200:]**2, axis=1)))
        rmse_cond_down_deep = np.mean(np.sqrt(np.nanmean(diff_cond_down[:, i_200:]**2, axis=1)))
        rmse_cond_up_deep   = np.mean(np.sqrt(np.nanmean(diff_cond_up[:,   i_200:]**2, axis=1)))    


        # Print statistics + grouped deep RMSE

        temp_rmse = [rmse_temp_down, rmse_temp_up]
        salt_rmse = [rmse_salt_down, rmse_salt_up]
        oxy_rmse = [rmse_oxy_down, rmse_oxy_up]

        temp_rmse_deep = [rmse_temp_down_deep, rmse_temp_up_deep]
        salt_rmse_deep = [rmse_salt_down_deep, rmse_salt_up_deep]
        oxy_rmse_deep = [rmse_oxy_down_deep, rmse_oxy_up_deep]

        labels = ['MVP down', 'MVP up']
        colors = ['blue', 'orange']

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        for idx, (ax, data, data_deep, title, ylabel) in enumerate(zip(
            axes,
            [temp_rmse,  salt_rmse,  oxy_rmse],
            [temp_rmse_deep, salt_rmse_deep, oxy_rmse_deep],
            ['Temperature', 'Salinity', 'Oxygen'],
            ['RMSE (°C)', 'RMSE (psu)', 'RMSE (umol/L)']
        )):
            x = np.arange(len(labels))
            width = 0.35
            # Side-by-side grouped bars: left = All depths, right = Deep
            label_all = 'All depths' if idx == 0 else None
            label_deep = 'Deep (≥200 dbar)' if idx == 0 else None
            bars_all = ax.bar(x - width/2, data, width=width, color=colors, edgecolor='k', label=label_all)
            bars_deep = ax.bar(x + width/2, data_deep, width=width, color=colors, edgecolor='k', alpha=0.6, label=label_deep)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', linestyle=':', alpha=0.5)
            ymax = max(max(data), max(data_deep)) * 1.25  # 25% margin above highest
            ax.set_ylim(0, ymax)

            # Annotations
            for b in bars_all:
                h = b.get_height()
                if np.isfinite(h):
                    ax.annotate(f'{h:.3f}', (b.get_x() + b.get_width()/2, h),
                                xytext=(0, 3), textcoords='offset points',
                                ha='center', va='bottom', fontsize=10, fontweight='bold')
            for b in bars_deep:
                h = b.get_height()
                if np.isfinite(h):
                    ax.annotate(f'{h:.3f}', (b.get_x() + b.get_width()/2, h),
                                xytext=(0, 3), textcoords='offset points',
                                ha='center', va='bottom', fontsize=9)

            if idx == 0:
                ax.legend()

        fig.suptitle('RMSE MVP vs CTD')
        fig.tight_layout()
        plt.show()


        if cond:
            print("Conductivity RMSE (MVP - CTD):")
            print(f"  MVP down: {rmse_cond_down:.4f} S/m (deep: {rmse_cond_down_deep:.4f} S/m)")
            print(f"  MVP up:   {rmse_cond_up:.4f} S/m (deep: {rmse_cond_up_deep:.4f} S/m)")



        print('TEMPERATURE')
        fig, ax = plt.subplots(figsize=(6, 8))
        for i in range(diff_temp_down.shape[0]):
            ax.plot(diff_temp_down[i], pressure_grid, color='blue', alpha=0.15, linewidth=0.8)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel('Erreur (MVP - CTD) °C')
        ax.set_ylabel('Pressure (dbar)')
        ax.set_title('Toutes les différences individuelles (Down)')
        ax.grid()
        plt.show()



        p50 = np.nanpercentile(diff_temp_down, 50, axis=0)
        p25 = np.nanpercentile(diff_temp_down, 25, axis=0)
        p75 = np.nanpercentile(diff_temp_down, 75, axis=0)
        p05 = np.nanpercentile(diff_temp_down, 5, axis=0)
        p95 = np.nanpercentile(diff_temp_down, 95, axis=0)

        fig, ax = plt.subplots(figsize=(6, 8))
        ax.fill_betweenx(pressure_grid, p05, p95, color='blue', alpha=0.15, label='5-95%')
        ax.fill_betweenx(pressure_grid, p25, p75, color='blue', alpha=0.35, label='25-75%')
        ax.plot(p50, pressure_grid, color='blue', label='Médiane')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel('Erreur (MVP - CTD) °C')
        ax.set_ylabel('Pressure (dbar)')
        ax.legend()
        ax.grid()
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))
        vmax = np.nanpercentile(np.abs(diff_temp_down), 95)
        pc = ax.pcolormesh(np.arange(diff_temp_down.shape[0]), pressure_grid, diff_temp_down.T,
                            cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        ax.invert_yaxis()
        ax.set_xlabel('Index profil')
        ax.set_ylabel('Pressure (dbar)')
        fig.colorbar(pc, label='Erreur (°C)')
        plt.show()


        depth_bins = np.arange(0, 1000, 100)
        bin_idx = np.digitize(pressure_grid, depth_bins)

        data_by_bin = [diff_temp_down[:, bin_idx == b].flatten() for b in range(1, len(depth_bins))]
        data_by_bin = [d[~np.isnan(d)] for d in data_by_bin]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(data_by_bin, positions=depth_bins[:-1] + 50, widths=60, vert=False, showfliers=True)
        ax.invert_yaxis()
        ax.axvline(0, color='r', linestyle='--')
        ax.set_xlabel('Erreur (°C)')
        ax.set_ylabel('Pressure (dbar)')
        plt.show()

        fig, ax = plt.subplots()
        ax.hist(diff_temp_down.flatten(), bins=100, color='blue', alpha=0.7)
        ax.axvline(0, color='k', linestyle='--')
        ax.axvline(np.nanmean(diff_temp_down), color='r', label=f'Moyenne={np.nanmean(diff_temp_down):.3f}')
        ax.axvline(np.nanmedian(diff_temp_down), color='g', label=f'Médiane={np.nanmedian(diff_temp_down):.3f}')
        ax.set_xlabel('Erreur (°C)')
        ax.legend()
        plt.show()

        if cond:
            print('CONDUCTIVITY')

            fig, ax = plt.subplots(figsize=(6, 8))
            for i in range(diff_cond_down.shape[0]):
                ax.plot(diff_cond_down[i], pressure_grid, color='blue', alpha=0.15, linewidth=0.8)
            ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
            ax.invert_yaxis()
            ax.set_xlabel('Erreur (MVP - CTD) S/m')
            ax.set_ylabel('Pressure (dbar)')
            ax.set_title('Toutes les différences individuelles (Down)')
            ax.grid()
            plt.show()



            p50 = np.nanpercentile(diff_cond_down, 50, axis=0)
            p25 = np.nanpercentile(diff_cond_down, 25, axis=0)
            p75 = np.nanpercentile(diff_cond_down, 75, axis=0)
            p05 = np.nanpercentile(diff_cond_down, 5, axis=0)
            p95 = np.nanpercentile(diff_cond_down, 95, axis=0)

            fig, ax = plt.subplots(figsize=(6, 8))
            ax.fill_betweenx(pressure_grid, p05, p95, color='blue', alpha=0.15, label='5-95%')
            ax.fill_betweenx(pressure_grid, p25, p75, color='blue', alpha=0.35, label='25-75%')
            ax.plot(p50, pressure_grid, color='blue', label='Médiane')
            ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
            ax.invert_yaxis()
            ax.set_xlabel('Erreur (MVP - CTD) S/m')
            ax.set_ylabel('Pressure (dbar)')
            ax.legend()
            ax.grid()
            plt.show()

            fig, ax = plt.subplots(figsize=(8, 6))
            vmax = np.nanpercentile(np.abs(diff_cond_down), 95)
            pc = ax.pcolormesh(np.arange(diff_cond_down.shape[0]), pressure_grid, diff_cond_down.T,
                                cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
            ax.invert_yaxis()
            ax.set_xlabel('Index profil')
            ax.set_ylabel('Pressure (dbar)')
            fig.colorbar(pc, label='Erreur (S/m)')
            plt.show()


            depth_bins = np.arange(0, 1000, 100)
            bin_idx = np.digitize(pressure_grid, depth_bins)

            data_by_bin = [diff_cond_down[:, bin_idx == b].flatten() for b in range(1, len(depth_bins))]
            data_by_bin = [d[~np.isnan(d)] for d in data_by_bin]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.boxplot(data_by_bin, positions=depth_bins[:-1] + 50, widths=60, vert=False, showfliers=True)
            ax.invert_yaxis()
            ax.axvline(0, color='r', linestyle='--')
            ax.set_xlabel('Erreur (S/m)')
            ax.set_ylabel('Pressure (dbar)')
            plt.show()

            fig, ax = plt.subplots()
            ax.hist(diff_cond_down.flatten(), bins=100, color='blue', alpha=0.7)
            ax.axvline(0, color='k', linestyle='--')
            ax.axvline(np.nanmean(diff_cond_down), color='r', label=f'Moyenne={np.nanmean(diff_cond_down):.3f}')
            ax.axvline(np.nanmedian(diff_cond_down), color='g', label=f'Médiane={np.nanmedian(diff_cond_down):.3f}')
            ax.set_xlabel('Erreur (S/m)')
            ax.legend()
            plt.show()




        print('SALINTY')
        fig, ax = plt.subplots(figsize=(6, 8))
        for i in range(diff_salt_down.shape[0]):
            ax.plot(diff_salt_down[i], pressure_grid, color='blue', alpha=0.15, linewidth=0.8)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel('Erreur (MVP - CTD) PSU')
        ax.set_ylabel('Pressure (dbar)')
        ax.set_title('Toutes les différences individuelles (Down)')
        ax.grid()
        plt.show()



        p50 = np.nanpercentile(diff_salt_down, 50, axis=0)
        p25 = np.nanpercentile(diff_salt_down, 25, axis=0)
        p75 = np.nanpercentile(diff_salt_down, 75, axis=0)
        p05 = np.nanpercentile(diff_salt_down, 5, axis=0)
        p95 = np.nanpercentile(diff_salt_down, 95, axis=0)

        fig, ax = plt.subplots(figsize=(6, 8))
        ax.fill_betweenx(pressure_grid, p05, p95, color='blue', alpha=0.15, label='5-95%')
        ax.fill_betweenx(pressure_grid, p25, p75, color='blue', alpha=0.35, label='25-75%')
        ax.plot(p50, pressure_grid, color='blue', label='Médiane')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel('Erreur (MVP - CTD) PSU')
        ax.set_ylabel('Pressure (dbar)')
        ax.legend()
        ax.grid()
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))
        vmax = np.nanpercentile(np.abs(diff_salt_down), 95)
        pc = ax.pcolormesh(np.arange(diff_salt_down.shape[0]), pressure_grid, diff_salt_down.T,
                            cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        ax.invert_yaxis()
        ax.set_xlabel('Index profil')
        ax.set_ylabel('Pressure (dbar)')
        fig.colorbar(pc, label='Erreur (PSU)')
        plt.show()


        depth_bins = np.arange(0, 1000, 100)
        bin_idx = np.digitize(pressure_grid, depth_bins)

        data_by_bin = [diff_salt_down[:, bin_idx == b].flatten() for b in range(1, len(depth_bins))]
        data_by_bin = [d[~np.isnan(d)] for d in data_by_bin]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(data_by_bin, positions=depth_bins[:-1] + 50, widths=60, vert=False, showfliers=True)
        ax.invert_yaxis()
        ax.axvline(0, color='r', linestyle='--')
        ax.set_xlabel('Erreur (PSU)')
        ax.set_ylabel('Pressure (dbar)')
        plt.show()

        fig, ax = plt.subplots()
        ax.hist(diff_salt_down.flatten(), bins=100, color='blue', alpha=0.7)
        ax.axvline(0, color='k', linestyle='--')
        ax.axvline(np.nanmean(diff_salt_down), color='r', label=f'Moyenne={np.nanmean(diff_salt_down):.3f}')
        ax.axvline(np.nanmedian(diff_salt_down), color='g', label=f'Médiane={np.nanmedian(diff_salt_down):.3f}')
        ax.set_xlabel('Erreur (PSU)')
        ax.legend()
        plt.show()






        print('OXYGEN')
        fig, ax = plt.subplots(figsize=(6, 8))
        for i in range(diff_oxy_down.shape[0]):
            ax.plot(diff_oxy_down[i], pressure_grid, color='blue', alpha=0.15, linewidth=0.8)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel('Erreur (MVP - CTD) umol/L')
        ax.set_ylabel('Pressure (dbar)')
        ax.set_title('Toutes les différences individuelles (Down)')
        ax.grid()
        plt.show()



        p50 = np.nanpercentile(diff_oxy_down, 50, axis=0)
        p25 = np.nanpercentile(diff_oxy_down, 25, axis=0)
        p75 = np.nanpercentile(diff_oxy_down, 75, axis=0)
        p05 = np.nanpercentile(diff_oxy_down, 5, axis=0)
        p95 = np.nanpercentile(diff_oxy_down, 95, axis=0)

        # Régression linéaire sur la médiane (erreur = a * pression + b)
        mask = ~np.isnan(p50)
        slope, intercept = np.polyfit(pressure_grid[mask], p50[mask], 1)

        fit_line = slope * pressure_grid + intercept

        print(f"Pente de l'erreur médiane en oxygène : {slope:.5f} umol/L par dbar")
        print(f"  (soit {slope*100:.3f} umol/L / 100 dbar)")
        print(f"Ordonnée à l'origine : {intercept:.4f} umol/L")

        fig, ax = plt.subplots(figsize=(6, 8))
        ax.fill_betweenx(pressure_grid, p05, p95, color='blue', alpha=0.15, label='5-95%')
        ax.fill_betweenx(pressure_grid, p25, p75, color='blue', alpha=0.35, label='25-75%')
        ax.plot(p50, pressure_grid, color='blue', label='Médiane')
        ax.plot(fit_line, pressure_grid, color='red', linestyle='--', linewidth=1.5,
                label=f'Tendance: {slope:.5f} umol/L/dbar')
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel('Erreur (MVP - CTD) umol/L')
        ax.set_ylabel('Pressure (dbar)')
        ax.set_title('Oxygène — Erreur médiane et tendance linéaire')
        ax.legend()
        ax.grid()
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 6))
        vmax = np.nanpercentile(np.abs(diff_oxy_down), 95)
        pc = ax.pcolormesh(np.arange(diff_oxy_down.shape[0]), pressure_grid, diff_oxy_down.T,
                            cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='auto')
        ax.invert_yaxis()
        ax.set_xlabel('Index profil')
        ax.set_ylabel('Pressure (dbar)')
        fig.colorbar(pc, label='Erreur (umol/L)')
        plt.show()


        depth_bins = np.arange(0, 1000, 100)
        bin_idx = np.digitize(pressure_grid, depth_bins)

        data_by_bin = [diff_oxy_down[:, bin_idx == b].flatten() for b in range(1, len(depth_bins))]
        data_by_bin = [d[~np.isnan(d)] for d in data_by_bin]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(data_by_bin, positions=depth_bins[:-1] + 50, widths=60, vert=False, showfliers=True)
        ax.invert_yaxis()
        ax.axvline(0, color='r', linestyle='--')
        ax.set_xlabel('Erreur (umol/L)')
        ax.set_ylabel('Pressure (dbar)')
        plt.show()

        fig, ax = plt.subplots()
        ax.hist(diff_oxy_down.flatten(), bins=100, color='blue', alpha=0.7)
        ax.axvline(0, color='k', linestyle='--')
        ax.axvline(np.nanmean(diff_oxy_down), color='r', label=f'Moyenne={np.nanmean(diff_oxy_down):.3f}')
        ax.axvline(np.nanmedian(diff_oxy_down), color='g', label=f'Médiane={np.nanmedian(diff_oxy_down):.3f}')
        ax.set_xlabel('Erreur (umol/L)')
        ax.legend()
        plt.show()


    def correct_oxygen(self,id_mvp=None,id_ctd=None,plotting=False,):
        """
        Apply oxygen correction to MVP dissolved oxygen profiles thanks to CTD data.
        Args:
            id_mvp (int): Index of the MVP profile to use for correction.
            id_ctd (int): Index of the CTD profile to use for correction.
            num_sample (int): Number of pressure levels for interpolation.
            plotting (bool): If True, plot the correction results.
            correction (bool): If True, update corrected attributes.
        """

        if not self.mvp or not self.ctd:
            raise ValueError("MVP or CTD data not loaded.")
        

        if id_mvp is None:
            id_mvp,id_ctd = 0,0
            print(f"No profile index provided, using first profiles: MVP {id_mvp} and CTD {id_ctd}.")
        elif id_ctd is None:
            id_ctd = id_mvp
        

        if hasattr(self,'DO_mvp_corr_interp') == False:
            raise ValueError("Please run the interpolation method first to create the DO_mvp_corr_interp attribute.")
        

        oxy_mvp = self.DO_mvp_corr_interp[id_mvp]
        oxy_ctd = self.DO_ctd_interp[id_ctd]
        pres = self.PRES_mvp_corr_interp[id_mvp]

        mask = ~np.isnan(oxy_mvp) & ~np.isnan(oxy_ctd)
        oxy_mvp = oxy_mvp[mask]
        oxy_ctd = oxy_ctd[mask]
        pres = pres[mask]

        diff = oxy_mvp - oxy_ctd

        A = np.vstack([oxy_ctd, np.ones(len(oxy_ctd))]).T
        diff = diff.flatten()

        a_estim, b_estim = np.linalg.lstsq(A, diff, rcond=None)[0]
        print(f"Estimated linear relationship: diff = {a_estim:.4f} * oxy_ctd + {b_estim:.4f}")

        Do_mvp_corr = self.DO_mvp_corr_interp[id_mvp] - (a_estim * self.DO_ctd_interp[id_ctd] + b_estim)


        rmse_before = np.sqrt(np.nanmean((self.DO_mvp_corr_interp[id_mvp] - self.DO_ctd_interp[id_ctd])**2))
        rmse_after = np.sqrt(np.nanmean((Do_mvp_corr - self.DO_ctd_interp[id_ctd])**2))
        if plotting:
            print(f"RMSE before correction: {rmse_before:.4f}")
            print(f"RMSE after correction: {rmse_after:.4f}")


        if plotting:

            plt.figure(figsize=(6,8))
            plt.plot(oxy_mvp, pres, label='MVP DO')
            plt.plot(Do_mvp_corr[mask],pres,label='MVP DO corrigé')
            plt.plot(oxy_ctd, pres, label='CTD DO')
            plt.gca().invert_yaxis()
            plt.xlabel('Oxygène dissous [µmol/L]')
            plt.ylabel('Profondeur [m]')
            plt.title(f'Profil de DO - Profil {id_mvp} MVP vs Profil {id_ctd} CTD')
            plt.legend()
            plt.show()

        self.DO_mvp_corr_interp[id_mvp] = Do_mvp_corr


    def correct_oxygen_all(self,mode):
        """
        Apply oxygen correction to all MVP profiles using the nearest CTD profiles.
        Args:
            mode (str): Mode for finding nearest profile ('Dist' or 'Time').
        """

        
        for id_mvp in range(0,self.PRES_mvp.shape[0]):

            id_nearest_ctd = mvp.find_nearest_profile(self.TIME_mvp_corr_interp[id_mvp],self.Lat_mvp_corr_interp[id_mvp], self.Lon_mvp_corr_interp[id_mvp],self.TIME_ctd ,self.LAT_ctd, self.LON_ctd,mode)[0]
            self.correct_oxygen(id_mvp=id_mvp, id_ctd=id_nearest_ctd, plotting=False)

        print("Oxygen correction applied to all MVP profiles using nearest CTD profiles.")

    def correct_oxygen_with_slope(self,slope):
        """
        Apply oxygen correction to all MVP profiles using a specified slope.
        Args:
            slope (float): Slope  coefficient for the linear correction: oxy_new = oxy_old - slope * pressure
        """
        if self.corrected:
            for i in range(len(self.oxy.keys())):
                self.oxy[i] = self.oxy[i] - slope * self.PRES_mvp_corr[i]
        else:
            for i in range(len(self.oxy.keys())):
                self.oxy[i] = self.oxy[i] - slope * self.PRES_mvp[i]

    def convert_DO_to_umolL(self,correction=False):
        """
        Convert dissolved oxygen from percentage saturation to µmol/L
        Args:
            correction (bool): If True, use corrected profiles; otherwise, use raw profiles.   
        """


        if correction:  
            oxy = {}
        else:
            oxy = np.empty_like(self.DO_mvp)


        for id_mvp in range(0,self.PRES_mvp.shape[0]):
            if correction:
                sal = self.SALT_mvp_corr[id_mvp]
                pres = self.PRES_mvp_corr[id_mvp]
                Lon = np.full_like(pres, self.LON_mvp[id_mvp][0])
                Lat = np.full_like(pres, self.LAT_mvp[id_mvp][0])
                temp = self.TEMP_mvp_corr[id_mvp]
                # interpolate self.DO_mvp to self.PRES_mvp_corr
                sort_idx = np.argsort(self.PRES_mvp[id_mvp])
                pres_mvp_sorted = self.PRES_mvp[id_mvp][sort_idx]
                do_mvp_sorted   = self.DO_mvp[id_mvp][sort_idx]

                do = np.interp(pres, pres_mvp_sorted, do_mvp_sorted)
            else:
                sal = self.SALT_mvp[id_mvp]
                pres = self.PRES_mvp[id_mvp]
                Lon = np.full_like(pres, self.LON_mvp[id_mvp][0])
                Lat = np.full_like(pres, self.LAT_mvp[id_mvp][0])
                temp = self.TEMP_mvp[id_mvp]
                do = self.DO_mvp[id_mvp]

            SA = gsw.SA_from_SP(sal, pres, Lon, Lat)
            CT = gsw.CT_from_pt(SA, temp)
            rho = gsw.rho(SA, CT, pres)  # Density in kg/m^3
            oxy_sat = gsw.O2sol_SP_pt(sal, temp) * rho /1000
            oxy[id_mvp] = do * oxy_sat / 100 +self.oxy_offset

        self.oxy = oxy

    def mvp_correction(self,high_cutoff=1,dp=0.1):
        """
        Apply corrections to MVP profiles: filtering, temporal lag correction, bin averaging, and median filtering.
        Args:
            high_cutoff (float): High cutoff frequency for filtering (Hz).
            dp (float): Pressure bin size for bin averaging (dbar)
        """

        T_MVP_corr = []
        P_MVP_corr = []
        C_MVP_corr = []
        S_MVP_corr = []
        Time_MVP_corr = []

        print("Applying corrections to MVP profiles...")

        for id in tqdm(range(0,self.PRES_mvp.shape[0])):

            if np.sum(np.isnan(self.PRES_mvp[id]))==len(self.PRES_mvp[id]):
                T_MVP_corr.append(np.array([np.nan]))
                P_MVP_corr.append(np.array([np.nan]))
                C_MVP_corr.append(np.array([np.nan]))
                S_MVP_corr.append(np.array([np.nan]))
                Time_MVP_corr.append(np.array([np.nan]))

                continue
                
   
            T = self.TEMP_mvp[id]
            C = self.COND_mvp[id]
            P = self.PRES_mvp[id]
            S = self.SALT_mvp[id]
            Time = np.linspace(0,len(P)/self.freq_echant,len(P)) 

            mask = ~np.isnan(C) & ~np.isnan(T)

            C = C[mask]
            T = T[mask] 
            Time = Time[mask]
            P = P[mask]
            S = S[mask]

            T,C = mvp.filtering_tc(T,C,self.freq_echant,high_cutoff)
            T_corr,S_corr = mvp.temporal_lag(T,C,P,self.freq_echant)

            if dp != None:
                P_ba,T_corr_ba,C_ba,S_corr_ba,Time_ba = mvp.bin_average_v2(P,T_corr,C,S_corr,Time,dp=0.2)

                S_corr_medfilt = median_filter(S_corr_ba, size=5)
                # S_corr_medfilt = S_corr_ba
            else:

                T_corr_ba = T_corr
                C_ba = C
                S_corr_medfilt = S_corr
                P_ba = P
                Time_ba = Time

            T_MVP_corr.append(T_corr_ba)
            P_MVP_corr.append(P_ba)
            C_MVP_corr.append(C_ba)
            S_MVP_corr.append(S_corr_medfilt)
            Time_MVP_corr.append(Time_ba)
        
        self.TEMP_mvp_corr = {i: sublist for i, sublist in enumerate(T_MVP_corr)}
        self.PRES_mvp_corr = {i: sublist for i, sublist in enumerate(P_MVP_corr)}
        self.COND_mvp_corr = {i: sublist for i, sublist in enumerate(C_MVP_corr)}
        self.SALT_mvp_corr = {i: sublist for i, sublist in enumerate(S_MVP_corr)}
        self.TIME_mvp_corr = {i: sublist for i, sublist in enumerate(Time_MVP_corr)}

        
        self.corrected = True

        self.convert_DO_to_umolL(correction=True)
        print("MVP profiles corrected.")


    def process_NO3(self,cal_file,only_new=False):
        """
        Process SUNA data to compute NO3 concentrations using the provided calibration file.
        Args:
            cal_file (str): Path to the calibration file.
            only_new (bool): If True, only process new files.
        """

        if not self.corrected:
            raise ValueError("MVP data not corrected. Apply corrections first.")

        if self.subdirs:
            files = sorted(filter(os.path.isfile,glob.glob(self.data_path + '**/*.raw', recursive=True)))
        else:
            files = sorted(filter(os.path.isfile,glob.glob(self.data_path + '*.raw', recursive=self.subdirs)))


        if only_new:
            list_output = [f for f in os.listdir(self.output_path) if f.endswith(".nc")]
            files = [f for f in files if not "MVP_"+os.path.basename(f).replace('.raw', '.nc') in list_output]

    

        time_suna = []
        pressure_suna = []
        no3_suna = []
        lat_suna = []
        lon_suna = []
        temp_suna = []
        sal_suna = []
        

        nprof = 0
        self.delp.sort(reverse=True)
        for i in self.delp:
            del files[i]
        
        for f in tqdm(files):


            (l_m_timestamp, l_pressure, l_dark, l_NO3_raw, l_spectre, l_d_line_raw, lat, lon, header_dt) = mvp.parse_SUNA(f,self.freq_echant)

            # print(f"Error processing {f}: {e}")


            if np.isnan(l_pressure[0]) or l_pressure[0] is None:
                l_pressure[0] = l_pressure[1]
            
            i_bottom = np.nanmin(np.where(l_pressure == np.max(l_pressure)))

            #downcast
            #interpolktat self.TEMP_mvp_corr onto the SUNA pressure levels
            pres = l_pressure[:i_bottom]
            temp = np.interp(pres, self.PRES_mvp_corr[nprof], self.TEMP_mvp_corr[nprof])
            sal = np.interp(pres, self.PRES_mvp_corr[nprof], self.SALT_mvp_corr[nprof])
            if len(pres)<10:
                for l in [time_suna,pressure_suna,no3_suna,lat_suna,lon_suna,temp_suna,sal_suna]:
                    l.append([np.nan])
            else:
                [no3_down,_] = no3_p.NO3_HSorens_RTQC(cal_file,nprof,temp,sal,pres,l_spectre[:i_bottom],l_dark[:i_bottom],l_NO3_raw[:i_bottom],N=41)

                time_suna.append(l_m_timestamp[:i_bottom])
                pressure_suna.append(l_pressure[:i_bottom])
                no3_suna.append(no3_down)
                lat_suna.append(lat)
                lon_suna.append(lon)
                temp_suna.append(temp)
                sal_suna.append(sal)


            nprof +=1
            #upcast
            pres = l_pressure[i_bottom:]
            temp = np.interp(pres, self.PRES_mvp_corr[nprof], self.TEMP_mvp_corr[nprof])
            sal = np.interp(pres, self.PRES_mvp_corr[nprof], self.SALT_mvp_corr[nprof])
            if len(pres)<10:
                for l in [time_suna,pressure_suna,no3_suna,lat_suna,lon_suna,temp_suna,sal_suna]:
                    l.append([np.nan])
            else:
                [no3_up,_] = no3_p.NO3_HSorens_RTQC(cal_file,nprof,temp,sal,pres,l_spectre[i_bottom:],l_dark[i_bottom:],l_NO3_raw[i_bottom:],N=41)

                time_suna.append(l_m_timestamp[i_bottom:])
                pressure_suna.append(l_pressure[i_bottom:])
                no3_suna.append(no3_up)
                lat_suna.append(lat)
                lon_suna.append(lon)
                temp_suna.append(temp)
                sal_suna.append(sal)



            nprof +=1


        self.TIME_suna = {i: sublist for i, sublist in enumerate(time_suna)}
        self.PRES_suna = {i: sublist for i, sublist in enumerate(pressure_suna)}
        self.NO3_suna = {i: sublist for i, sublist in enumerate(no3_suna)}
        self.LAT_suna = {i: sublist for i, sublist in enumerate(lat_suna)}
        self.LON_suna = {i: sublist for i, sublist in enumerate(lon_suna)}
        self.TEMP_suna = {i: sublist for i, sublist in enumerate(temp_suna)}
        self.SALT_suna = {i: sublist for i, sublist in enumerate(sal_suna)}




        self.SUNA=True
        print("SUNA data processed and stored in the object.")



    def interpolate_CTD_and_MVPcorrected(self,length):

        """
        Interpolate CTD data onto the corrected MVP pressure levels.
        """

        if not self.corrected:
            raise ValueError("MVP data not corrected. Apply corrections first.")
        


        if not hasattr(self, 'PRES_mvp_corr'):
            raise ValueError("Corrected MVP data not available. Apply corrections first.")


        max_lenpres = max([len(p) for p in self.PRES_mvp_corr.values()])
        PRES_mvp_corr_mat =  np.array([list(row) + [np.nan] * (max_lenpres - len(row)) for row in self.PRES_mvp_corr.values()])

        max_lentemp = max([len(p) for p in self.TEMP_mvp_corr.values()])
        TEMP_mvp_corr_mat =  np.array([list(row) + [np.nan] * (max_lentemp - len(row)) for row in self.TEMP_mvp_corr.values()])

        max_lencond = max([len(p) for p in self.COND_mvp_corr.values()])
        COND_mvp_corr_mat =  np.array([list(row) + [np.nan] * (max_lencond - len(row)) for row in self.COND_mvp_corr.values()])

        max_lensalt = max([len(p) for p in self.SALT_mvp_corr.values()])
        SALT_mvp_corr_mat =  np.array([list(row) + [np.nan] * (max_lensalt - len(row)) for row in self.SALT_mvp_corr.values()])

        max_lenoxy = max([len(p) for p in self.oxy.values()])
        oxy_mvp_corr_mat =  np.array([list(row) + [np.nan] * (max_lenoxy - len(row)) for row in self.oxy.values()])

        if self.speed:
            max_lenvspd = max([len(p) for p in self.SPEED_mvp_corr.values()])
            SPEED_mvp_corr_mat =  np.array([list(row) + [np.nan] * (max_lenvspd - len(row)) for row in self.SPEED_mvp_corr.values()])

        if self.SUNA:
            max_lenno3 = max([len(p) for p in self.NO3_suna.values()])
            NO3_suna_mat =  np.array([list(row) + [np.nan] * (max_lenno3 - len(row)) for row in self.NO3_suna.values()])
            max_lenpresno3 = max([len(p) for p in self.PRES_suna.values()])
            PRES_suna_mat = np.array([list(row) + [np.nan] * (max_lenpresno3 - len(row)) for row in self.PRES_suna.values()])



        max_lentime = max([len(p) for p in self.TIME_mvp_corr.values()])
        TIME_mvp_corr_mat =  np.array([list(row) + [np.nan] * (max_lentime - len(row)) for row in self.TIME_mvp_corr.values()])


        pressure_grid = np.linspace(np.nanmin(PRES_mvp_corr_mat), np.nanmax(PRES_mvp_corr_mat), length)
        if self.ctd:
            self.TEMP_ctd_interp = mvp.vertical_interp(self.PRES_ctd, self.TEMP_ctd, pressure_grid)
            self.PRES_ctd_interp = mvp.vertical_interp(self.PRES_ctd, self.PRES_ctd, pressure_grid)
            self.COND_ctd_interp = mvp.vertical_interp(self.PRES_ctd, self.COND_ctd, pressure_grid)
            self.SALT_ctd_interp = mvp.vertical_interp(self.PRES_ctd, self.SALT_ctd, pressure_grid)
            self.oxy_ctd_interp = mvp.vertical_interp(self.PRES_ctd, self.oxy_ctd, pressure_grid)
            self.FLUO_ctd_interp = mvp.vertical_interp(self.PRES_ctd, self.FLUO_ctd, pressure_grid)
            self.TURB_ctd_interp = mvp.vertical_interp(self.PRES_ctd, self.TURB_ctd, pressure_grid)
        self.TEMP_mvp_corr_interp = mvp.vertical_interp(PRES_mvp_corr_mat, TEMP_mvp_corr_mat, pressure_grid)
        self.PRES_mvp_corr_interp = np.tile(pressure_grid, (PRES_mvp_corr_mat.shape[0], 1))
        self.COND_mvp_corr_interp = mvp.vertical_interp(PRES_mvp_corr_mat, COND_mvp_corr_mat, pressure_grid)
        self.SALT_mvp_corr_interp = mvp.vertical_interp(PRES_mvp_corr_mat, SALT_mvp_corr_mat, pressure_grid)
        self.oxy_mvp_corr_interp = mvp.vertical_interp(PRES_mvp_corr_mat, oxy_mvp_corr_mat, pressure_grid)
        self.FLUO_mvp_corr_interp = mvp.vertical_interp(self.PRES_mvp, self.FLUO_mvp, pressure_grid)
        self.TURB_mvp_corr_interp = mvp.vertical_interp(self.PRES_mvp, self.TURB_mvp, pressure_grid)
        self.PH_mvp_corr_interp = mvp.vertical_interp(self.PRES_mvp, self.PH_mvp, pressure_grid)
        self.SUNA_mvp_corr_interp = mvp.vertical_interp(self.PRES_mvp, self.SUNA_mvp, pressure_grid)
        if self.GPS:
            self.Lat_mvp_corr_interp = mvp.vertical_interp(self.PRES_mvp, self.Lat_mvp, pressure_grid)
            self.Lon_mvp_corr_interp = mvp.vertical_interp(self.PRES_mvp, self.Lon_mvp, pressure_grid)

        if self.speed:
            self.SPEED_mvp_corr_interp = mvp.vertical_interp(PRES_mvp_corr_mat, SPEED_mvp_corr_mat, pressure_grid)
        self.TIME_mvp_corr_interp = mvp.vertical_interp(PRES_mvp_corr_mat, TIME_mvp_corr_mat, pressure_grid)

        if self.SUNA:
            self.NO3_mvp_corr_interp = mvp.vertical_interp(PRES_suna_mat, NO3_suna_mat, pressure_grid)
            self.NO3_mvp_corr_interp = mvp.correct_suna(self.NO3_mvp_corr_interp,self.PRES_mvp_corr_interp)


        print('CTD data interpolated onto corrected MVP pressure levels.')


    def corrige_MVP_offset_on_ctd_exact(self,id_mvp,id_ctd,min_depth=-1):
        """
        This function corrects the offset between the MVP and CTD profiles by aligning the temperature, conductivity profiles. It calculates the mean difference in temperature between the two profiles and applies this correction to the CTD temperature data.
        id_mvp and id_ctd must be the same length as each MVP profile will be be corrected with the corresponding CTD profile. The function returns the corrected MVP temperature and conductivity profiles.
        This version of the correction suppose that CTD and MVP should be exactly the same profile (same location, same time). If it not the case, you shouldf use the other function _imple
        Args:
            id_mvp (list): List of indices of MVP profiles to correct.
            id_ctd (list): List of indices of CTD profiles to use for correction (must be the same length as id_mvp).
            min_depth (float): Minimum depth (in dbar) to consider for calculating mean differences

        """

        mean_temp_diff = []
        mean_cond_diff = []
        mean_salt_diff = []
        print("Calculating mean differences between MVP and CTD profiles before correction:")
        for i in range(len(id_mvp)):
            # Calculate the mean difference in temperature between the MVP and CTD profiles
            temp_diff = np.nanmean(self.TEMP_mvp_corr_interp[id_mvp[i]] - self.TEMP_ctd_interp[id_ctd[i]])
            mean_temp_diff.append(temp_diff)

            cond_diff = np.nanmean(self.COND_mvp_corr_interp[id_mvp[i]] - self.COND_ctd_interp[id_ctd[i]])
            mean_cond_diff.append(cond_diff)

            salt_diff = np.nanmean(self.SALT_mvp_corr_interp[id_mvp[i]] - self.SALT_ctd_interp[id_ctd[i]])
            mean_salt_diff.append(salt_diff)
        
        print("Mean temperature difference between MVP and CTD profiles:", np.mean(mean_temp_diff))
        print("Mean conductivity difference between MVP and CTD profiles:", np.mean(mean_cond_diff))
        print("Mean salinity difference between MVP and CTD profiles:", np.mean(mean_salt_diff))

        for i in range(len(id_mvp)):
            self.TEMP_mvp_corr_interp[id_mvp[i]] = mvp.align_profiles(self.PRES_mvp_corr_interp[id_mvp[i]], self.TEMP_ctd_interp[id_ctd[i]], self.TEMP_mvp_corr_interp[id_mvp[i]],min_depth)[0]
            self.COND_mvp_corr_interp[id_mvp[i]] = mvp.align_profiles(self.PRES_mvp_corr_interp[id_mvp[i]], self.COND_ctd_interp[id_ctd[i]], self.COND_mvp_corr_interp[id_mvp[i]],min_depth)[0]
            self.SALT_mvp_corr_interp[id_mvp[i]] = mvp.align_profiles(self.PRES_mvp_corr_interp[id_mvp[i]], self.SALT_ctd_interp[id_ctd[i]], self.SALT_mvp_corr_interp[id_mvp[i]],min_depth)[0]

        mean_temp_diff = []
        mean_cond_diff = []
        mean_salt_diff = []
        print("After correction:")
        for i in range(len(id_mvp)):
            # Calculate the mean difference in temperature between the MVP and CTD profiles
            temp_diff = np.nanmean(self.TEMP_mvp_corr_interp[id_mvp[i]] - self.TEMP_ctd_interp[id_ctd[i]])
            mean_temp_diff.append(temp_diff)

            cond_diff = np.nanmean(self.COND_mvp_corr_interp[id_mvp[i]] - self.COND_ctd_interp[id_ctd[i]])
            mean_cond_diff.append(cond_diff)

            salt_diff = np.nanmean(self.SALT_mvp_corr_interp[id_mvp[i]] - self.SALT_ctd_interp[id_ctd[i]])
            mean_salt_diff.append(salt_diff)
        print("Mean temperature difference between MVP and CTD profiles:", np.mean(mean_temp_diff))
        print("Mean conductivity difference between MVP and CTD profiles:", np.mean(mean_cond_diff))
        print("Mean salinity difference between MVP and CTD profiles:", np.mean(mean_salt_diff))
    

    def corrige_MVP_offset_on_ctd_simple(self,id_mvp,id_ctd,min_depth):
        """
        This function corrects the offset between the MVP and CTD profiles by aligning the temperature, conductivity profiles. It calculates the mean difference in temperature between the two profiles and applies this correction to the CTD temperature data.
        id_mvp and id_ctd must be the same length as each MVP profile will be be corrected with the corresponding CTD profile. The function returns the corrected MVP temperature and conductivity profiles.
        This version of the correction is less restritive than the other one, does not need the CTD aand MVP profiles to be exactly similar
        We advice to choose a min_depth that avoid to take into acount the surface layer which can introduce errors.
        Args:
            id_mvp (list): List of indices of MVP profiles to correct.
            id_ctd (list): List of indices of CTD profiles to use for correction (must be the same length as id_mvp).
            min_depth (float): Minimum depth (in dbar) to consider for calculating mean differences

        """

        mean_temp_diff = []
        mean_cond_diff = []
        mean_salt_diff = []
        print("Calculating mean differences between MVP and CTD profiles before correction:")
        for i in range(len(id_mvp)):
            id_valid = self.PRES_mvp_corr_interp[id_mvp[i]] >= min_depth
            # Calculate the mean difference in temperature between the MVP and CTD profiles
            temp_diff = np.nanmean(self.TEMP_mvp_corr_interp[id_mvp[i], id_valid] - self.TEMP_ctd_interp[id_ctd[i], id_valid])
            mean_temp_diff.append(temp_diff)

            cond_diff = np.nanmean(self.COND_mvp_corr_interp[id_mvp[i], id_valid] - self.COND_ctd_interp[id_ctd[i], id_valid])
            mean_cond_diff.append(cond_diff)

            salt_diff = np.nanmean(self.SALT_mvp_corr_interp[id_mvp[i], id_valid] - self.SALT_ctd_interp[id_ctd[i], id_valid])
            mean_salt_diff.append(salt_diff)
        print("Mean temperature difference between MVP and CTD profiles:", np.mean(mean_temp_diff))
        print("Mean conductivity difference between MVP and CTD profiles:", np.mean(mean_cond_diff))
        print("Mean salinity difference between MVP and CTD profiles:", np.mean(mean_salt_diff))

        mean_temp_diff = []
        mean_cond_diff = []
        mean_salt_diff = []
        for i in range(len(id_mvp)):
            id_valid = self.PRES_mvp_corr_interp[id_mvp[i]] >= min_depth

            # Calculate the mean difference in temperature between the MVP and CTD profiles
            temp_diff = np.nanmean(self.TEMP_mvp_corr_interp[id_mvp[i], id_valid] - self.TEMP_ctd_interp[id_ctd[i], id_valid])
            self.TEMP_mvp_corr_interp[id_mvp[i]] -= temp_diff

            cond_diff = np.nanmean(self.COND_mvp_corr_interp[id_mvp[i], id_valid] - self.COND_ctd_interp[id_ctd[i], id_valid])
            self.COND_mvp_corr_interp[id_mvp[i]] -= cond_diff

            salt_diff = np.nanmean(self.SALT_mvp_corr_interp[id_mvp[i], id_valid] - self.SALT_ctd_interp[id_ctd[i], id_valid])
            self.SALT_mvp_corr_interp[id_mvp[i]] -= salt_diff

        mean_temp_diff = []
        mean_cond_diff = []
        mean_salt_diff = []

        print("After correction:")
        for i in range(len(id_mvp)):
            id_valid = self.PRES_mvp_corr_interp[id_mvp[i]] >= min_depth

            # Calculate the mean difference in temperature between the MVP and CTD profiles
            temp_diff = np.nanmean(self.TEMP_mvp_corr_interp[id_mvp[i], id_valid] - self.TEMP_ctd_interp[id_ctd[i], id_valid])
            mean_temp_diff.append(temp_diff)

            cond_diff = np.nanmean(self.COND_mvp_corr_interp[id_mvp[i], id_valid] - self.COND_ctd_interp[id_ctd[i], id_valid])
            mean_cond_diff.append(cond_diff)

            salt_diff = np.nanmean(self.SALT_mvp_corr_interp[id_mvp[i], id_valid] - self.SALT_ctd_interp[id_ctd[i], id_valid])
            mean_salt_diff.append(salt_diff)

        print("Mean temperature difference between MVP and CTD profiles:", np.mean(mean_temp_diff))
        print("Mean conductivity difference between MVP and CTD profiles:", np.mean(mean_cond_diff))
        print("Mean salinity difference between MVP and CTD profiles:", np.mean(mean_salt_diff))

    def corrige_MVP_offset_on_ctd_all(self,min_depth,mode):

        """
        Detect the offset between MVP adn CTD and corrige MVP profiles using the nearest CTD profiles.
        Args:
            min_depth (float): Minimum depth (in dbar) to consider for calculating mean differences
            mode (str): Mode for finding nearest profile ('Dist' or 'Time').

        """

        for id_mvp in range(self.PRES_mvp.shape[0]):
            id_nearest_ctd = mvp.find_nearest_profile(self.TIME_mvp_corr_interp[id_mvp],self.Lat_mvp_corr_interp[id_mvp], self.Lon_mvp_corr_interp[id_mvp],self.TIME_ctd ,self.LAT_ctd, self.LON_ctd,mode)[0]
            self.corrige_MVP_offset_on_ctd_simple(id_mvp=id_mvp, id_ctd=id_nearest_ctd, min_depth=min_depth)

        print("MVP profiles corrected for offset against CTD profiles using nearest CTD profiles.")

    def to_netcdf(self, filepath, corrected=False, compression=True, engine=None, per_profile_files=False):
        """
        Export MVP data to a NetCDF file using xarray.

        Args:
            filepath (str): Output NetCDF file path (or directory if per_profile_files=True).
            corrected (bool): If False, write raw data. If True, write only corrected &
                              interpolated data with corrected coordinates. Default False.
            compression (bool): Enable compression (engine dependent). Default True.
            engine (str|None): One of 'netcdf4', 'h5netcdf', 'scipy'. If None, choose netcdf4.
            per_profile_files (bool): If True, write one .nc per MVP cycle (two rows: down and up).
        """
        if not getattr(self, 'mvp', False):
            raise RuntimeError("No MVP data loaded. Call load_mvp_data() first.")

        engine = 'netcdf4' if engine is None else engine
        if engine == 'scipy' and compression:
            print('Warning: scipy backend does not support compression; writing without compression.')
            compression = False

        n_prof = self.PRES_mvp.shape[0]

        # Profile labels
        if hasattr(self, 'label_mvp') and len(self.label_mvp) == n_prof:
            profile_labels = np.array(self.label_mvp, dtype='U')
        else:
            profile_labels = np.array([f'profile_{i}' for i in range(n_prof)], dtype='U')

        # Per-profile datetime
        if hasattr(self, 'DATETIME_mvp') and len(getattr(self, 'DATETIME_mvp', [])) > 0:
            prof_times = []
            for i in range(n_prof):
                j = i // 2
                if j < len(self.DATETIME_mvp) and self.DATETIME_mvp[j] is not None:
                    prof_times.append(np.datetime64(self.DATETIME_mvp[j]))
                else:
                    prof_times.append(np.datetime64('NaT'))
            profile_time = np.array(prof_times, dtype='datetime64[ns]')
        else:
            profile_time = np.array([np.datetime64('NaT')] * n_prof, dtype='datetime64[ns]')

        profile_idx = np.arange(n_prof, dtype=np.int32)

        data_vars = {}

        def _add(name, arr, dims, units=None, long_name=None):
            if arr is None:
                return
            attrs = {}
            if units is not None:
                attrs['units'] = units
            if long_name is not None:
                attrs['long_name'] = long_name
            data_vars[name] = (dims, arr, attrs)

        # =====================================================================
        # corrected=True  -> only corrected/interpolated data + corrected coords
        # corrected=False -> raw data + raw coords
        # =====================================================================
        if corrected:
            if not hasattr(self, 'PRES_mvp_corr_interp'):
                raise RuntimeError(
                    "Corrected/interpolated data not available. "
                    "Call interpolate_CTD_and_MVPcorrected() first."
                )

            n_samp = self.PRES_mvp_corr_interp.shape[1]
            sample_idx = np.arange(n_samp, dtype=np.int32)
            dims = ('profile', 'sample')

            # Data variables (corrected & interpolated only)
            _add('TEMP',  self.TEMP_mvp_corr_interp,  dims, 'degC',  'Corrected & interpolated temperature')
            _add('COND',  self.COND_mvp_corr_interp,  dims, 'mS/cm', 'Corrected & interpolated conductivity')
            _add('SAL',   self.SALT_mvp_corr_interp,  dims, 'psu',   'Corrected & interpolated salinity')
            _add('DO',    self.DO_mvp_corr_interp,    dims, 'ml/L',  'Corrected & interpolated dissolved oxygen')
            _add('FLUO',  self.FLUO_mvp_corr_interp,  dims, 'ug/L',  'Corrected & interpolated fluorescence')
            _add('TURB',  self.TURB_mvp_corr_interp,  dims, 'NTU',   'Corrected & interpolated turbidity')
            _add('PH',    self.PH_mvp_corr_interp,    dims, '1',     'Corrected & interpolated pH')
            _add('SUNA',  self.SUNA_mvp_corr_interp,  dims, None,    'Corrected & interpolated SUNA')
            if hasattr(self, 'SPEED_mvp_corr_interp'):
                _add('SPEED', self.SPEED_mvp_corr_interp, dims, 'm s-1', 'Corrected & interpolated profiling speed')

            # Coordinates: corrected versions
            _add('PRES', self.PRES_mvp_corr_interp, dims, 'dbar', 'Corrected & interpolated pressure')
            if hasattr(self, 'Lat_mvp_corr_interp'):
                _add('LATITUDE', self.Lat_mvp_corr_interp, dims, 'degrees_north', 'Corrected & interpolated latitude')
            if hasattr(self, 'Lon_mvp_corr_interp'):
                _add('LONGITUDE', self.Lon_mvp_corr_interp, dims, 'degrees_east', 'Corrected & interpolated longitude')
            if hasattr(self, 'TIME_mvp_corr_interp'):
                data_vars['TIME_s'] = (
                    dims,
                    self.TIME_mvp_corr_interp * 24.0 * 3600.0,
                    {'units': f'seconds since {self.date_ref.strftime("%Y-%m-%d %H:%M:%S")}',
                     'long_name': 'Corrected & interpolated time'}
                )

        else:
            # ---- Raw data ----
            n_samp = self.PRES_mvp.shape[1]
            sample_idx = np.arange(n_samp, dtype=np.int32)
            dims = ('profile', 'sample')

            _add('PRES',     self.PRES_mvp,     dims, 'dbar',           'Sea water pressure')
            _add('TEMP',     self.TEMP_mvp,     dims, 'degC',           'In-situ temperature')
            _add('COND',     self.COND_mvp,     dims, 'mS/cm',          'Conductivity')
            _add('SAL',      self.SALT_mvp,     dims, 'psu',            'Practical salinity')
            _add('SOUNDVEL', self.SOUNDVEL_mvp, dims, 'm s-1',          'Sound speed')
            _add('DO',       self.DO_mvp,       dims, 'ml/L',           'Dissolved oxygen')
            _add('TEMP2',    self.TEMP2_mvp,    dims, 'degC',           'Oxygen sensor temperature')
            _add('SUNA',     self.SUNA_mvp,     dims, None,             'SUNA raw/derived')
            _add('FLUO',     self.FLUO_mvp,     dims, 'ug/L',           'Chl fluorescence')
            _add('TURB',     self.TURB_mvp,     dims, 'NTU',            'Turbidity')
            _add('PH',       self.PH_mvp,       dims, '1',              'pH')
            _add('LATITUDE', self.Lat_mvp,      dims, 'degrees_north',  'Latitude at sample')
            _add('LONGITUDE',self.Lon_mvp,      dims, 'degrees_east',   'Longitude at sample')

            if hasattr(self, 'TIME_mvp'):
                time_seconds = self.TIME_mvp * 24.0 * 3600.0
            else:
                time_seconds = np.full((n_prof, n_samp), np.nan)
            data_vars['TIME_s'] = (
                dims, time_seconds,
                {'units': f'seconds since {self.date_ref.strftime("%Y-%m-%d %H:%M:%S")}',
                 'long_name': 'Time at sample'}
            )

        # ---- Coordinates ----
        coords = {
            'profile': ('profile', profile_idx),
            'sample': ('sample', sample_idx),
        }

        #  profile labels, profile time
        if engine in ('netcdf4', 'h5netcdf'):
            coords['profile_time'] = ('profile', profile_time, {'long_name': 'Profile nominal time'})
            coords['profile_label'] = ('profile', profile_labels, {'long_name': 'Profile label / file name'})
        else:
            ref = np.datetime64(self.date_ref)
            pt = profile_time.astype('datetime64[s]')
            mask = (pt == np.datetime64('NaT'))
            secs = (pt - ref).astype('timedelta64[s]').astype('float64')
            secs[mask] = np.nan
            coords['profile_time_sec'] = (
                'profile', secs,
                {'units': f'seconds since {self.date_ref.strftime("%Y-%m-%d %H:%M:%S")}',
                 'long_name': 'Profile nominal time'}
            )

        # Per-profile lat/lon (first valid sample from the appropriate source)
        def first_valid(vec):
            out = np.full(vec.shape[0], np.nan)
            for i in range(vec.shape[0]):
                idx = np.where(~np.isnan(vec[i]))[0]
                if idx.size:
                    out[i] = vec[i, idx[0]]
            return out

        if corrected and hasattr(self, 'Lat_mvp_corr_interp'):
            coords['profile_lat'] = ('profile', first_valid(self.Lat_mvp_corr_interp), {'units': 'degrees_north', 'long_name': 'Profile latitude'})
        elif hasattr(self, 'Lat_mvp'):
            coords['profile_lat'] = ('profile', first_valid(self.Lat_mvp), {'units': 'degrees_north', 'long_name': 'Profile latitude'})

        if corrected and hasattr(self, 'Lon_mvp_corr_interp'):
            coords['profile_lon'] = ('profile', first_valid(self.Lon_mvp_corr_interp), {'units': 'degrees_east', 'long_name': 'Profile longitude'})
        elif hasattr(self, 'Lon_mvp'):
            coords['profile_lon'] = ('profile', first_valid(self.Lon_mvp), {'units': 'degrees_east', 'long_name': 'Profile longitude'})

        # ---- Global attributes ----
        attrs = {
            'title': 'MVP profile data',
            'Conventions': 'CF-1.8',
            'institution': 'LMD/CNRS',
            'source': 'PyMVP',
            'history': f"Created on {datetime.now().isoformat()}",
            'mvp_Yorig': int(self.Yorig),
            'sampling frequency_hz': float(self.freq_echant) if hasattr(self, 'freq_echant') else -1.0,
            'corrected': int(corrected),
        }

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # ---- Compression encoding ----
        encoding = None
        if compression:
            if engine == 'netcdf4':
                encoding = {name: {'zlib': True, 'complevel': 4} for name in data_vars}
            elif engine == 'h5netcdf':
                encoding = {name: {'compression': 'gzip', 'compression_opts': 4} for name in data_vars}

        # ---- Write ----
        if (not per_profile_files) and filepath.lower().endswith('.nc'):
            ds.to_netcdf(filepath, encoding=encoding, engine=engine)
            print(f"NetCDF written: {filepath} using engine={engine}")
            return

        base_dir = filepath
        if not base_dir.endswith(os.sep):
            base_dir += os.sep
        os.makedirs(base_dir, exist_ok=True)

        base_name = "MVP_" + os.path.basename(self.data_path).rstrip(os.sep)

        if per_profile_files:
            total_pairs = (n_prof + 1) // 2
            for i in range(total_pairs):
                idxs = [k for k in (2 * i, 2 * i + 1) if k < n_prof]
                if not idxs:
                    continue
                ds_i = ds.isel(profile=idxs)
                fname = f"{base_name}_profile_{i:03d}.nc"
                out_path = os.path.join(base_dir, fname)
                ds_i.to_netcdf(out_path, encoding=encoding, engine=engine)
            print(f"NetCDF written per profile into: {base_dir} using engine={engine}")
        else:
            file_name = f"{base_name}.nc"
            out_path = os.path.join(base_dir, file_name)
            ds.to_netcdf(out_path, encoding=encoding, engine=engine)
            print(f"NetCDF written: {out_path} using engine={engine}")


    def to_csv(self, filepath, corrected=False, per_profile_files=False):
        """
        Export MVP data to CSV files.

        Args:
            filepath (str): Output CSV file path (or directory if per_profile_files=True).
            corrected (bool): If False, write raw data. If True, write only corrected data. Default False.
            per_profile_files (bool): If True, write one .csv per MVP profile; else write all in one file.
        """
        if not getattr(self, 'mvp', False):
            raise RuntimeError("No MVP data loaded. Call load_mvp_data() first.")

        n_prof = self.PRES_mvp.shape[0]

        # Profile labels
        if hasattr(self, 'label_mvp') and len(self.label_mvp) == n_prof:
            profile_labels = np.array(self.label_mvp, dtype='U')
        else:
            profile_labels = np.array([f'profile_{i}' for i in range(n_prof)], dtype='U')

        # Per-profile datetime
        if hasattr(self, 'DATETIME_mvp') and len(getattr(self, 'DATETIME_mvp', [])) > 0:
            prof_times = []
            for i in range(n_prof):
                j = i // 2
                if j < len(self.DATETIME_mvp) and self.DATETIME_mvp[j] is not None:
                    prof_times.append(self.DATETIME_mvp[j])
                else:
                    prof_times.append(None)
        else:
            prof_times = [None] * n_prof

        # Prepare output directory
        if per_profile_files:
            base_dir = filepath
            if not base_dir.endswith(os.sep):
                base_dir += os.sep
            os.makedirs(base_dir, exist_ok=True)
            base_name = "MVP_" + os.path.basename(self.data_path).rstrip(os.sep)
        else:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        # =====================================================================
        # Export all profiles or per-profile
        # =====================================================================
        
        if corrected:
            if not hasattr(self, 'PRES_mvp_corr'):
                raise RuntimeError("Corrected data not available. Call mvp_correction() first.")

            # Export corrected (non-interpolated) profiles
            for prof_id in range(n_prof):
                pres = self.PRES_mvp_corr[prof_id]
                temp = self.TEMP_mvp_corr[prof_id]
                salt = self.SALT_mvp_corr[prof_id]
                cond = self.COND_mvp_corr[prof_id]
                oxy = self.oxy[prof_id] if hasattr(self, 'oxy') and prof_id in self.oxy else np.full_like(pres, np.nan)
                time = self.TIME_mvp_corr[prof_id]

                # Calculate absolute datetimes
                if prof_times[prof_id] is not None:
                    if isinstance(prof_times[prof_id], str):
                        prof_start_time = datetime.fromisoformat(prof_times[prof_id])
                    else:
                        prof_start_time = prof_times[prof_id]
                else:
                    prof_start_time = self.date_ref

                datetimes = [prof_start_time + timedelta(seconds=float(t)) for t in time]

                # Build DataFrame for this profile
                df_data = {
                    'PRES (dbar)': pres,
                    'TEMP (°C)': temp,
                    'SAL (psu)': salt,
                    'COND (mS/cm)': cond,
                    'OXY (µmol/L)': oxy,
                    'TIME': datetimes,
                }
                
                for attr, col in [('FLUO_mvp', 'FLUO (ug/L)'), ('TURB_mvp', 'TURB (NTU)'), ('PH_mvp', 'PH')]:
                    if hasattr(self, attr):
                        xp = np.asarray(self.PRES_mvp[prof_id], dtype=float)
                        fp = np.asarray(getattr(self, attr)[prof_id], dtype=float)
                        sort_idx = np.argsort(xp)
                        xp_sorted, fp_sorted = xp[sort_idx], fp[sort_idx]
                        # Étendre les bornes pour éviter les NaN de bord
                        df_data[col] = np.interp(pres, xp_sorted, fp_sorted)  # sans left/right=nan

                if hasattr(self, 'NO3_suna'):
                    xp = np.asarray(self.PRES_suna[prof_id], dtype=float)
                    fp = np.asarray(self.NO3_suna[prof_id], dtype=float)
                    sort_idx = np.argsort(xp)
                    df_data['NO3 (µmol/L)'] = np.interp(pres, xp[sort_idx], fp[sort_idx])


                if hasattr(self, 'Lat_mvp_corr_interp'):
                    lat = self.Lat_mvp_corr_interp[prof_id][:len(pres)]
                    lon = self.Lon_mvp_corr_interp[prof_id][:len(pres)]
                else:
                    lat = np.full_like(pres, self.Lat_mvp[prof_id, 0], dtype=float)
                    lon = np.full_like(pres, self.Lon_mvp[prof_id, 0], dtype=float)

                df_data['LAT (°N)'] = lat
                df_data['LON (°E)'] = lon

                df = pd.DataFrame(df_data)
                df = df.sort_values('TIME', kind='mergesort').reset_index(drop=True)

                if per_profile_files:
                    fname = f"{base_name}_profile_{prof_id:03d}.csv"
                    out_path = os.path.join(base_dir, fname)
                    df.to_csv(out_path, index=False, na_rep='NaN')
                else:
                    df['profile_id'] = prof_id
                    df['profile_label'] = profile_labels[prof_id]
                    df['profile_time'] = prof_times[prof_id]
                    
                    if prof_id == 0:
                        df_all = df
                    else:
                        df_all = pd.concat([df_all, df], ignore_index=True)

            if not per_profile_files:
                df_all = df_all.sort_values('TIME', kind='mergesort').reset_index(drop=True)
                df_all.to_csv(filepath, index=False, na_rep='NaN')
                print(f"CSV written (corrected, single file): {filepath}")
            else:
                print(f"CSV written (corrected, per-profile) into: {base_dir}")

        else:
            # Export raw profiles
            for prof_id in range(n_prof):
                pres = self.PRES_mvp[prof_id]
                temp = self.TEMP_mvp[prof_id]
                salt = self.SALT_mvp[prof_id]
                cond = self.COND_mvp[prof_id]
                do = self.DO_mvp[prof_id]
                time = self.TIME_mvp[prof_id]

                # Calculate absolute datetimes
                if prof_times[prof_id] is not None:
                    if isinstance(prof_times[prof_id], str):
                        prof_start_time = datetime.fromisoformat(prof_times[prof_id])
                    else:
                        prof_start_time = prof_times[prof_id]
                else:
                    prof_start_time = self.date_ref

                datetimes = [prof_start_time + timedelta(seconds=float(t)) for t in time]

                # Build DataFrame for this profile
                df_data = {
                    'PRES (dbar)': pres,
                    'TEMP (°C)': temp,
                    'SAL (psu)': salt,
                    'COND (mS/cm)': cond,
                    'DO (ml/L)': do,
                    'TIME': datetimes,
                }
                for attr, col in [('FLUO_mvp', 'FLUO (ug/L)'), ('TURB_mvp', 'TURB (NTU)'), ('PH_mvp', 'PH')]:
                    if hasattr(self, attr):
                        xp = self.TIME_mvp[prof_id]      # temps brut (axe source)
                        fp = getattr(self, attr)[prof_id]
                        target_time = self.TIME_mvp_corr[prof_id]  # temps corrigé (axe cible)

                        # Trier par temps croissant
                        sort_idx = np.argsort(xp)
                        xp_s, fp_s = xp[sort_idx], fp[sort_idx]

                        # Supprimer doublons temporels
                        _, unique_idx = np.unique(xp_s, return_index=True)
                        df_data[col] = np.interp(target_time, xp_s[unique_idx], fp_s[unique_idx])

                if hasattr(self, 'NO3_suna'):
                    xp = self.TIME_suna[prof_id]
                    fp = self.NO3_suna[prof_id]
                    target_time = self.TIME_mvp_corr[prof_id]

                    sort_idx = np.argsort(xp)
                    xp_s, fp_s = xp[sort_idx], fp[sort_idx]

                    _, unique_idx = np.unique(xp_s, return_index=True)
                    df_data['NO3 (µmol/L)'] = np.interp(target_time, xp_s[unique_idx], fp_s[unique_idx])

                df_data['LAT (°N)'] = self.Lat_mvp[prof_id]
                df_data['LON (°E)'] = self.Lon_mvp[prof_id]

                df = pd.DataFrame(df_data)
                df = df.sort_values('TIME', kind='mergesort').reset_index(drop=True)

                if per_profile_files:
                    fname = f"{base_name}_profile_{prof_id:03d}.csv"
                    out_path = os.path.join(base_dir, fname)
                    df.to_csv(out_path, index=False, na_rep='NaN')
                else:
                    df['profile_id'] = prof_id
                    df['profile_label'] = profile_labels[prof_id]
                    df['profile_time'] = prof_times[prof_id]

                    if prof_id == 0:
                        df_all = df
                    else:
                        df_all = pd.concat([df_all, df], ignore_index=True)

            if not per_profile_files:
                df_all = df_all.sort_values('TIME', kind='mergesort').reset_index(drop=True)
                df_all.to_csv(filepath, index=False, na_rep='NaN')
                print(f"CSV written (raw, single file): {filepath}")
            else:
                print(f"CSV written (raw, per-profile) into: {base_dir}")

            

    def help(self):
        """
        Print all methods of the class with their docstring (header).
        """
        for attr in dir(self):
            if callable(getattr(self, attr)) and not attr.startswith("__"):
                method = getattr(self, attr)
                doc = method.__doc__
                print(f"{attr}:\n{doc}\n{'-'*40}")      


    def compute_dist(self):
        """
        Create a 2D array of cumulative distance along the transect based on the corrected and interpolated latitude and longitude of the MVP profiles. The distance is calculated using the geodesic distance between consecutive valid points, with a gap threshold to handle large jumps in position.
        The resulting distance array is stored in self.DIST_interp.
        """

        lat = self.Lat_mvp_corr_interp
        lon = self.Lon_mvp_corr_interp

        n_profiles, n_points = self.PRES_mvp_corr_interp.shape

        dist_all = np.full_like(self.PRES_mvp_corr_interp, np.nan, dtype=float)

        dist_cum = 0.0
        prev_lat, prev_lon = None, None

        for i in range(n_profiles):
            valid_j = np.where(~np.isnan(lat[i, :]) & ~np.isnan(lon[i, :]))[0]
            if len(valid_j) == 0:
                continue

            j_first, j_last = valid_j[0], valid_j[-1]
            lat_first, lon_first = lat[i, j_first], lon[i, j_first]
            lat_last, lon_last = lat[i, j_last], lon[i, j_last]

            if prev_lat is not None:
                gap = geodesic((prev_lat, prev_lon), (lat_first, lon_first)).km
                if gap >15:
                    dist_cum += gap
                else:
                    gap = 0.
                        
            else:
                gap = 0.0

            dist_profile = geodesic((lat_first, lon_first), (lat_last, lon_last)).km if j_last != j_first else 0.0


            dist_all[i, j_first] = dist_cum

            if j_last != j_first:
                for j in valid_j:
                    frac = (j - j_first) / (j_last - j_first)
                    dist_all[i, j] = dist_cum + frac * dist_profile

            dist_cum += dist_profile
            prev_lat, prev_lon = lat_last, lon_last

        self.DIST_interp = dist_all
        self.dist = True

                

    def plot_MVP_transect(self,VAR='TEMP',depth_max=None,depth_min=None,vmax=None,vmin=None,cmap=None,save=None,gap_threshold=20,save_ncdf=None):
        """
        Plot a section of 2D inteprolated MVP data
        Args:
            VAR (str): Variable to plot. Choose from 'TEMP', 'COND', 'SAL', 'DO', 'FLUO', 'TURB', 'PH', 'SUNA', 'SPEED'.
=            depth_max (float): Maximum depth to display in the plot. If None, use max depth in data.
            depth_min (float): Minimum depth to display in the plot. If None, use 0.
            vmax (float): Maximum value for color scale. If None, use max value in data.
            vmin (float): Minimum value for color scale. If None, use min value in data.
            cmap: Matplotlib colormap to use. If None, use default colormap.
            save (str): If provided, save the plot to the specified file path.
            save_ncdf (str): If provided, save the transect data to a NetCDF file at the specified path.

        
        """

        if hasattr(self, 'PRES_mvp_corr_interp') == False:
            raise ValueError("Corrected and interpolated MVP data not available. Apply corrections and interpolation first.")


        if  self.dist is False or self.corrected is False:
                    raise ValueError("MVP data must be corrected and distance must be computed before plotting transect. Call self.mvp_correction() and self.compute_distance() first.")


        match VAR:
            case 'TEMP':
                var = self.TEMP_mvp_corr_interp
                unit = '°C'
            case 'COND':
                var = self.COND_mvp_corr_interp
                unit = 'mS/cm'
            case 'SAL':
                var = self.SALT_mvp_corr_interp
                unit = 'psu'
            case 'OX':
                var = self.oxy_mvp_corr_interp
                unit = 'µmol/L'
            case 'FLUO':
                var = self.FLUO_mvp_corr_interp
                unit = 'µg/L'
            case 'TURB':
                var = self.TURB_mvp_corr_interp
                unit = 'NTU'
            case 'PH':
                var = self.PH_mvp_corr_interp
                unit = '1'
            case 'SUNA':
                var = self.NO3_mvp_corr_interp
                unit = 'µmol/L'
            case 'SPEED':
                var = self.SPEED_mvp_corr_interp
                unit = 'm/s'
            case _: 
                raise ValueError(f"Variable {var} not recognized. Choose from 'TEMP', 'COND', 'SAL', 'OX', 'FLUO', 'TURB', 'PH', 'SUNA', 'SPEED'.")


        if depth_max is None:
            depth_max = np.nanmax(PRES_2d)
        if depth_min is None:
            depth_min = 0


 


        DIST_2d = self.DIST_interp             # (n_profils, n_prof)
        LAT_2d  = self.Lat_mvp_corr_interp     # (n_profils, n_prof)
        LON_2d  = self.Lon_mvp_corr_interp     # (n_profils, n_prof)
        PRES_2d = self.PRES_mvp_corr_interp    # (n_profils, n_prof)
        T_2d    = var   # (n_profils, n_prof)

        datetime_julian_days = mdates.date2num(self.DATETIME_mvp)
        DATE_2D = np.zeros_like(DIST_2d)
        for i in range(0,DATE_2D.shape[0]//2):
            DATE_2D[2*i, :] = np.tile(datetime_julian_days[i], DATE_2D.shape[1])
            DATE_2D[2*i+1, :] = np.tile(datetime_julian_days[i], DATE_2D.shape[1])


        n_profils, n_prof = DIST_2d.shape


        dist_profile = np.nanmean(DIST_2d, axis=1)
        date_profile = np.nanmean(DATE_2D, axis=1)
        lat_profile  = np.nanmean(LAT_2d, axis=1)
        lon_profile  = np.nanmean(LON_2d, axis=1)

        valid = ~np.isnan(dist_profile) & ~np.isnan(lat_profile) & ~np.isnan(lon_profile)
        dist_profile = dist_profile[valid]
        date_profile = date_profile[valid]
        lat_profile  = lat_profile[valid]
        lon_profile  = lon_profile[valid]
        T_valid      = T_2d[valid, :]      # on garde les mêmes profils valides pour T

        # tri par distance croissante
        order = np.argsort(dist_profile)
        dist_profile = dist_profile[order]
        date_profile = date_profile[order]
        lat_profile  = lat_profile[order]
        lon_profile  = lon_profile[order]
        T_valid      = T_valid[order, :]

        # suppression des doublons de distance (obligatoire pour interp1d)
        dist_profile, uniq_idx = np.unique(dist_profile, return_index=True)
        date_profile = date_profile[uniq_idx]
        lat_profile = lat_profile[uniq_idx]
        lon_profile = lon_profile[uniq_idx]
        T_valid     = T_valid[uniq_idx, :]


        dist_grid = np.linspace(dist_profile.min(), dist_profile.max(), int(dist_profile.max()-dist_profile.min()))  
        datetime_grid = np.interp(dist_grid, dist_profile, date_profile)


        pres_ref = np.nanmean(PRES_2d, axis=0)   # (n_prof,) -> les profondeurs "canoniques"
        P_grid = pres_ref.copy()

        # ------------------------------------------------------------------
        #  Lat/lon : interpolation 1D le long de la distance, dupliquée en profondeur
        # ------------------------------------------------------------------
        f_lat = interp1d(dist_profile, lat_profile, kind='linear', bounds_error=False, fill_value=np.nan)
        f_lon = interp1d(dist_profile, lon_profile, kind='linear', bounds_error=False, fill_value=np.nan)

        Lat_1d = f_lat(dist_grid)
        Lon_1d = f_lon(dist_grid)

        Lat_grid = np.tile(Lat_1d, (len(P_grid), 1))   # (n_prof, 500)
        Lon_grid = np.tile(Lon_1d, (len(P_grid), 1))

        # ------------------------------------------------------------------
        #  T : interpolation 1D le long de la distance, profondeur par profondeur
        # ------------------------------------------------------------------
        T_grid = np.full((len(P_grid), len(dist_grid)), np.nan)

        for j in range(n_prof):
            col = T_valid[:, j]                     # valeurs de T à la profondeur j, pour chaque profil
            ok = ~np.isnan(col)
            if ok.sum() < 2:
                continue
            f_T = interp1d(dist_profile[ok], col[ok], kind='linear',
                            bounds_error=False, fill_value=np.nan)
            T_grid[j, :] = f_T(dist_grid)

        # smoothing (not used here)
        T_grid_smooth = T_grid.copy()  
        # ------------------------------------------------------------------
        #  Masquage des gros écarts entre profils (gaps de distance)
        # ------------------------------------------------------------------

        gaps = np.diff(dist_profile)                     # écarts entre profils consécutifs
        gap_idx = np.where(gaps > gap_threshold)[0]       # indices i tels que gap entre profil i et i+1

        # masque booléen sur dist_grid : True = à l'intérieur d'un trou
        mask_gap = np.zeros_like(dist_grid, dtype=bool)

        for i in gap_idx:
            d_start = dist_profile[i]
            d_end   = dist_profile[i + 1]
            mask_gap |= (dist_grid > d_start) & (dist_grid < d_end)

        # application du masque (NaN sur toutes les profondeurs pour ces colonnes)
        T_grid_smooth[:, mask_gap] = np.nan
        Lat_grid[:, mask_gap] = np.nan
        Lon_grid[:, mask_gap] = np.nan

     

        # -----------------------------
        # Plot
        # -----------------------------


        if cmap is None:
            cmap = plt.get_cmap('viridis')


        fig, ax = plt.subplots(figsize=(12,6))
        pcm = ax.pcolormesh(dist_grid, P_grid, T_grid_smooth, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.invert_yaxis()
        ax.set_xlabel("Distance le long du transect [km]")
        ax.set_ylabel("Profondeur [m]")
        ax.set_title(f"{VAR} transect (interpolated)")
        ax.set_ylim(depth_max, depth_min)
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.set_label(f"{VAR} (units)")



        n_dates = len(self.DATETIME_mvp)
        # lignes paires correspondantes dans le tableau complet DIST_interp
        profile_rows_full = np.arange(0, 2 * n_dates, 2)  # 0, 2, 4, ..., jusqu'à n_dates dates


        profile_rows_sel = profile_rows_full
        datetime_sel = np.asarray(self.DATETIME_mvp)

        if len(profile_rows_sel) >= 2:
            # position représentative en distance pour chaque profil (moyenne le long de la ligne)
            dist_positions = np.array([
                np.nanmean(self.DIST_interp[row, :]) for row in profile_rows_sel
            ])

            # tri par distance croissante (requis pour l'interpolation)
            order = np.argsort(dist_positions)
            dist_positions = dist_positions[order]
            datetime_sel = np.asarray(datetime_sel)[order]

            # retirer d'éventuels doublons de distance
            dist_unique, unique_idx = np.unique(dist_positions, return_index=True)
            datenum_unique = mdates.date2num(datetime_sel[unique_idx])

            def dist_to_datenum(d):
                return np.interp(d, dist_unique, datenum_unique)

            def datenum_to_dist(n):
                return np.interp(n, datenum_unique, dist_unique)

            secax = ax.secondary_xaxis(-0.18, functions=(dist_to_datenum, datenum_to_dist))
            secax.set_xticks(datenum_unique[::5])  # une date sur 5
            secax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %Hh'))  # mois/jour heure
            secax.tick_params(axis='x', rotation=45, labelsize=8)

            fig.subplots_adjust(bottom=0.28)
        else:
            print("Warning: pas assez de dates disponibles pour l'axe secondaire (DATETIME_mvp).")


        if save is not None:
            plt.savefig(save)

        self.dist_grid = dist_grid
        self.P_grid = P_grid
        if save_ncdf is not None:
            # Save the interpolated transect data to a NetCDF file
            # add also the units for the variable

            ds_transect = xr.Dataset(
                {
                    VAR: (('depth', 'distance'), T_grid_smooth),

                    'lat':  (('depth', 'distance'), Lat_grid),
                    'lon':  (('depth', 'distance'), Lon_grid),
                },
                coords={
                    'depth': P_grid,
                    'distance': dist_grid,
                    'datetime': (('distance',), datetime_grid)
                },
                attrs={
                    'title': f'Interpolated {VAR} transect',
                    'source': 'MVP data',
                    'created_on': datetime.now().isoformat(),
                }
            )


            ds_transect[VAR].attrs['units'] = unit
            ds_transect['lat'].attrs['units'] = 'degrees_north'
            ds_transect['lon'].attrs['units'] = 'degrees_east'
            ds_transect['depth'].attrs['units'] = 'dbar'  # ou 'm' selon ta variable de pression/profondeur
            ds_transect['distance'].attrs['units'] = 'km'  # ou 'm' selon ton unité
            ds_transect['datetime'].attrs['units'] = 'days since 1970-01-01'  # optionnel si déjà en datetime64

            ds_transect.to_netcdf(save_ncdf)
            print(f"Interpolated transect data saved to {save_ncdf}")
            
        plt.show()



    def plot_MVP_transect_simple(self,VAR='TEMP',delta=None,l_id=None,depth_max=None,depth_min=None):
        """
        Plot a simple transect of corrected and interpolated MVP data for a specified variable. Each profile is offset vertically by a specified delta to visualize multiple profiles on the same plot.
        Args:
            VAR (str): Variable to plot. Choose from 'TEMP', 'COND', 'SAL', 'OX', 'FLUO', 'TURB', 'PH', 'SUNA', 'SPEED'.
            delta (float): Vertical offset between profiles. If None, default values are used based on the variable.
            l_id (list): List of profile indices to plot. If None, all profiles are plotted.
            depth_max (float): Maximum depth to display in the plot. If None, use max depth in data.
            depth_min (float): Minimum depth to display in the plot. If None, use 0.

        
        """

        if hasattr(self, 'PRES_mvp_corr_interp') == False:
            raise ValueError("Corrected and interpolated MVP data not available. Apply corrections and interpolation first.")
        
        if l_id is None:
            l_id = list(range(self.PRES_mvp_corr_interp.shape[0]))

        match VAR:
            case 'TEMP':
                var = self.TEMP_mvp_corr_interp
                delta = 0.5 if delta is None else delta
            case 'COND':
                var = self.COND_mvp_corr_interp
                delta = 0.5 if delta is None else delta
            case 'SAL':
                var = self.SALT_mvp_corr_interp
                delta = 0.1 if delta is None else delta
            case 'OX':
                var = self.oxy_mvp_corr_interp
                delta = 5 if delta is None else delta
            case 'FLUO':
                var = self.FLUO_mvp_corr_interp
                delta = 0.5 if delta is None else delta
            case 'TURB':
                var = self.TURB_mvp_corr_interp
                delta = 0.5 if delta is None else delta
            case 'PH':
                var = self.PH_mvp_corr_interp
                delta = 0.1 if delta is None else delta
            case 'SUNA':
                var = self.NO3_mvp_corr_interp
                delta = 0.5 if delta is None else delta
            case 'SPEED':
                var = self.SPEED_mvp_corr_interp
                delta = 0.1 if delta is None else delta
            case _: 
                raise ValueError(f"Variable {var} not recognized. Choose from 'TEMP', 'COND', 'SAL', 'OX', 'FLUO', 'TURB', 'PH', 'SUNA', 'SPEED'.")



        P = self.PRES_mvp_corr_interp[l_id]
        plt.close('all')
        plt.figure()
        for i in range(len(l_id)):
            plt.plot(var[l_id[i]]+i*delta, P[l_id[i]])
        plt.gca().invert_yaxis()
        plt.xlabel(f"{VAR}")
        plt.gca().set_xticks([])
        plt.ylabel("Depth [m]")
        plt.ylim(depth_max, depth_min)
        plt.title(f"{VAR}")
        plt.show()



def split_ctd(pres, array):

    ibot = np.min(np.where(pres == pres.max()))

    array_down = array[:ibot]
    array_up = array[ibot:]

    return array_down, array_up