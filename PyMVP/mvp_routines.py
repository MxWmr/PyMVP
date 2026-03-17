
################################################################################
#
# Collection of routines for MVP treatment
#
#
# Function get_log: Read MPV log file to get starting and ending times of the cycle
#
#
# Function read_mvp_cycle: Read one MVP cycle from a mvp_2022XXX.m1 file
#
#
# Function time_mvp_cycle_up: 
#	Allocate time to each sample for a MVP cycle
#	Select the ascending MVP cycle (less noise)
#
# Function time_mvp_cycle_down: 
#	Allocate time to each sample for a MVP cycle
#	Select the descending MVP cycle
#
#
# 
#
# 
# 
#
#
################################################################################

#
# Import libraries
#

import numpy as np 
import scipy.stats as st 
from datetime import date
from datetime import datetime
from scipy import interpolate
from scipy.signal import butter, freqz
from scipy import signal
import gsw
from scipy.interpolate import pchip_interpolate
import similaritymeasures
from netCDF4 import Dataset
from scipy.signal import butter, filtfilt, correlate, correlation_lags

#
################################################################################
#
# Function get_log: Read MPV log file to get starting and ending times of the cycle, latitude, longitude, and datetime
#
#   input:
#     mvp_log_name : ASCII MVP Log file (MVP_2022xxx.log)
#     Yorig        : time is counted in days since Yorig/1/1 (here 1950/1/1)
#
#   output:
#     mvp_tstart  : Start of the dive in days since Yorig/1/1
#     mvp_tend    : End of the dive in days since Yorig/1/1
#     cycle_dur   : Duration of the cycle in seconds
#     lat         : Latitude (float, if available)
#     lon         : Longitude (float, if available)
#     dt_station  : Datetime object (datetime.datetime) of the station (if available)
#
################################################################################

def get_log(mvp_log_name,Yorig):

    #print('Reading '+mvp_log_name)

    # Get start time and end time of the cycle
    flog = open(mvp_log_name, 'r', encoding = "ISO-8859-1")

    # --- Read header and extract date, lat, lon, datetime ---
    lat = None
    lon = None
    dt_station = None
    mvptime = None
    mvpdate = None
    header_lines = []
    # Read first 14 lines (header)
    for i in range(14):
        line = flog.readline()
        header_lines.append(line)
    # Parse LAT (line 9, index 8)
    try:
        lat_line = header_lines[8]
        lat_str = lat_line.split(':')[1].strip().split(',')[0]  # e.g. '4253.6113800'
        lat_dir = lat_line.strip().split(',')[-1].replace(':','').strip()  # e.g. 'N'
        lat_deg = float(lat_str[:2])
        lat_min = float(lat_str[2:])
        lat = lat_deg + lat_min/60.0
        if lat_dir.upper() == 'S':
            lat = -lat
    except Exception:
        lat = None
    # Parse LON (line 10, index 9)
    try:
        lon_line = header_lines[9]
        lon_str = lon_line.split(':')[1].strip().split(',')[0]  # e.g. '00614.5387900'
        lon_dir = lon_line.strip().split(',')[-1].replace(':','').strip()  # e.g. 'E'
        lon_deg = float(lon_str[:3])
        lon_min = float(lon_str[3:])
        lon = lon_deg + lon_min/60.0
        if lon_dir.upper() == 'W':
            lon = -lon
    except Exception:
        lon = None
    # Parse Time (line 12, index 11)
    try:
        time_line = header_lines[11]
        time_str = time_line.split(':',1)[1].strip()  # e.g. '10:28:58.6'
    except Exception:
        time_str = None
    # Parse Date (line 13, index 12)
    try:
        date_line = header_lines[12]
        date_str = date_line.split(':',1)[1].strip()  # e.g. '09/08/2025'
        mvpdate = datetime.strptime(date_str, "%d/%m/%Y").date()
        mvptime = mvpdate.toordinal() - date.toordinal(date(Yorig, 1, 1))
    except Exception:
        mvpdate = None
        mvptime = None
    # Compose datetime object
    try:
        if time_str is not None and mvpdate is not None:
            dt_station = datetime.strptime(date_str + ' ' + time_str, "%d/%m/%Y %H:%M:%S.%f")
    except ValueError:
        try:
            dt_station = datetime.strptime(date_str + ' ' + time_str, "%d/%m/%Y %H:%M:%S")
        except Exception:
            dt_station = None

    # Read 3 more lines (as before)
    for i in range(3):
        line = flog.readline()

    # --- Read data for start/stop times ---
    mvp_tstart = None
    mvp_tend = None
    hh1 = mn1 = sc1 = None
    while True:
        line = flog.readline()
        if line == '':
            break
        words = line.split()
        if len(words) < 3:
            continue
        if words[1] == 'EVENT:':
            if words[2][:5] == 'START':
                hh1 = float(words[2][6:8])
                mn1 = float(words[2][9:11])
                sc1 = float(words[2][12:16])
                mvp_tstart = mvptime + (hh1 + (mn1 + sc1 / 60.) / 60.) / 24.
            if words[2][:5] == 'STOP_':
                hh2 = float(words[2][6:8])
                mn2 = float(words[2][9:11])
                sc2 = float(words[2][12:16])
                if hh2 < hh1:
                    hh2 = hh2 + 24.
                mvp_tend = mvptime + (hh2 + (mn2 + sc2 / 60.) / 60.) / 24.

    cycle_dur = (mvp_tend - mvp_tstart) * (24 * 60 * 60)

    flog.close()

    return mvp_tstart, mvp_tend, cycle_dur, lat, lon, dt_station






#
################################################################################
#
# Function read_mvp_cycle_raw: Read one MVP cycle from a mvp_2022XXX.raw file
#
#   input:
#	mvp_dat_name : ASCII MVP .raw file (mvp_2022xxx.raw)
#
#
#   output:
#
#	pres        : pressure [dbar] 
#	soundvel    : Sound velocity [m/s]
#	do          :  dissolved oxygen [umol/kg]
#	temp2       : Temperature from DO sensor [oC]
#	suna        :  SUNA data [umol/kg]
#	fluo        : Fluorometer data [ug/l]
#	turb        : Turbidity data [NTU]
#	ph          : pH data [pH units]
# 
# 
################################################################################
#

def read_mvp_cycle_raw(mvp_dat_name):

    #print('Reading '+mvp_dat_name)

    # Open the file (there was a problem with encoding)
    fdat = open(mvp_dat_name, 'r', encoding = "ISO-8859-1")

    # Préparer les listes pour chaque variable utile
    pres = []         # Pressure
    cond = []         # Conductivity
    temp = []         # Temperature
    soundvel = []     # Sound velocity
    dox = []          # Dissolved oxygen
    temp2 = []        # Temperature from DO sensor
    suna = []         # SUNA data
    fluo = []         # Fluorometer data
    turb = []         # Turbidity data
    ph = []           # pH data


    # Sauter le header : lire jusqu'à la première ligne commençant par 'M'
    while True:
        pos = fdat.tell()
        line = fdat.readline()
        if line == '':
            break
        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] == 'M':
            # On revient en arrière pour traiter cette ligne dans la boucle principale
            fdat.seek(pos)
            break

    # Lecture des données à partir de la première ligne 'M'
    while True:
        line = fdat.readline()
        if line == '':
            break
        line = line.strip()
        if len(line) == 0:
            continue
        if line[0] == 'Z':
            continue  # ignorer les lignes Z
        if line[0] == 'M':
            words = line.split()
            if len(words) < 11:
                continue
            try:
                pres.append(float(words[1]))
                soundvel.append(float(words[2]))
                cond.append(float(words[3]))
                temp.append(float(words[4]))
                dox.append(float(words[5]))
                temp2.append(float(words[6]))
                suna.append(float(words[7]))
                fluo.append(float(words[8]))
                turb.append(float(words[9]))
                ph.append(float(words[10]))
            except Exception:
                continue

    fdat.close()

    # Convertir en numpy arrays
    pres = np.array(pres)
    soundvel = np.array(soundvel)
    cond = np.array(cond)
    temp = np.array(temp)
    do = np.array(dox)
    temp2 = np.array(temp2)
    suna = np.array(suna)
    fluo = np.array(fluo)
    turb = np.array(turb)
    ph = np.array(ph)

    return pres, soundvel, cond, temp, do, temp2, suna, fluo, turb, ph



#
################################################################################
#
# Function read_mvp_cycle_ncdf: Read one MVP cycle from a mvp_2022XXX.ncdf file
#
#   input:
#	mvp_dat_name : NetCDF MVP .ncdf file (mvp_2022xxx.ncdf)
#
#
#   output:
#
#	pres        : pressure [dbar] 
#	soundvel    : Sound velocity [m/s]
#	do          :  dissolved oxygen [umol/kg]
#	temp2       : Temperature from DO sensor [oC]
#	suna        :  SUNA data [umol/kg]
#	fluo        : Fluorometer data [ug/l]
#	turb        : Turbidity data [NTU]
#	ph          : pH data [pH units]
# 
# 
################################################################################
#

def read_mvp_cycle_ncdf(mvp_dat_name):


    #print('Reading '+mvp_dat_name)

    # Open the file
    nc = Dataset(mvp_dat_name, 'r')

    # Read variables
    pres = nc.variables['PRES'][:]
    soundvel = nc.variables['SOUNDVEL'][:]
    cond = nc.variables['COND'][:]
    temp = nc.variables['TEMP'][:]
    do = nc.variables['DO'][:]
    temp2 = nc.variables['TEMP2'][:]
    suna = nc.variables['SUNA'][:] if 'SUNA' in nc.variables else None
    fluo = nc.variables['FLUO'][:] if 'FLUO' in nc.variables else None
    turb = nc.variables['TURB'][:] if 'TURB' in nc.variables else None
    ph = nc.variables['PH'][:] if 'PH' in nc.variables else None

    nc.close()

    return pres, soundvel, cond, temp, do, temp2, suna, fluo, turb, ph










# ################################################################################
#
# Function time_mvp_cycle_up_bgc:
#     Allocate time to each sample for a MVP cycle (BGC version)
#     Select the ascending MVP cycle (less noise)
#
#   input:
#     pres      : pressure [dbar]
#     soundvel  : sound velocity [m/s]
#     do        : dissolved oxygen [umol/kg]
#     temp2     : temperature from DO sensor [°C]
#     suna      : SUNA data [umol/kg]
#     fluo      : fluorometer data [ug/l]
#     turb      : turbidity data [NTU]
#     ph        : pH data [pH units]
#     mvp_tstart: start of the dive in days since Yorig/1/1
#     mvp_tend  : end of the dive in days since Yorig/1/1
#
#   output:
#     pres_up, soundvel_up, do_up, temp2_up, suna_up, fluo_up, turb_up, ph_up
#     (données ascendantes pour chaque variable)
#	  time_cycle   : Time of each sample in days since Yorig/1/1 
#
# ################################################################################
#

def time_mvp_cycle_up(args,mvp_tstart,mvp_tend):

    # Allocate time to each data point
    N = np.size(args[0])
    time_cycle = np.linspace(mvp_tstart, mvp_tend, N)

    # Get only the ascending lines
    ibot = np.min(np.where(args[0] == args[0].max()))
    for i, arg in enumerate(args):
        args[i] = arg[ibot:]


    time_cycle_up = time_cycle[ibot:]

    return args + [time_cycle_up]


# ################################################################################
#
# Function time_mvp_cycle_down_bgc:
#     Allocate time to each sample for a MVP cycle (BGC version)
#     Select the descending MVP cycle
#
#   input:
#     pres      : pressure [dbar]
#     soundvel  : sound velocity [m/s]
#     do        : dissolved oxygen [umol/kg]
#     temp2     : temperature from DO sensor [°C]
#     suna      : SUNA data [umol/kg]
#     fluo      : fluorometer data [ug/l]
#     turb      : turbidity data [NTU]
#     ph        : pH data [pH units]
#     mvp_tstart: start of the dive in days since Yorig/1/1
#     mvp_tend  : end of the dive in days since Yorig/1/1
#
#   output:
#     pres_down, soundvel_down, do_down, temp2_down, suna_down, fluo_down, turb_down, ph_down
#     (données descendantes pour chaque variable)
#     time_cycle_down: Time of each sample in days since Yorig/1/1
#
# ################################################################################
def time_mvp_cycle_down(args, mvp_tstart, mvp_tend):

    N = np.size(args[0])
    time_cycle = np.linspace(mvp_tstart, mvp_tend, N)

    # Trouver les indices pour la partie descendante
    ibot = np.min(np.where(args[0] == args[0].max()))

    for i, arg in enumerate(args):
        args[i] = arg[:ibot]

    time_cycle_down = time_cycle[:ibot]

    return args + [time_cycle_down]



# ################################################################################
#
# Function raw_data_conversion:
#     Converts raw BGC sensor data (in V or mV) to physical units.
#     Not very efficien because of np.vectorize, if too long, recode the conversion functions to work with numpy arrays directly.
#
#   input:
#     pres, soundvel : already in physical units
#     do_raw, temp2_raw, suna_raw, fluo_raw, turb_raw, ph_raw : raw sensor data to convert
#
#   output:
#     pres, soundvel, do, temp2, suna, fluo, turb, ph : all variables in physical units (currently identical to input)
#
# ################################################################################
def raw_data_conversion(pres, soundvel, cond, temp, do_raw, temp2_raw, suna_raw, fluo_raw, turb_raw, ph_raw):
    """
    Converts raw BGC sensor data (in V or mV) to physical units.
    
    """

    temp2 = np.vectorize(TEMP2_conversion)(temp2_raw)  
    do = np.vectorize(DO_conversion)(do_raw, temp, pres)  
    suna = np.vectorize(SUNA_conversion)(suna_raw)
    fluo = np.vectorize(FLUO_conversion)(fluo_raw)
    turb = np.vectorize(TURBIDITY_conversion)(turb_raw)
    ph = np.vectorize(PH_conversion)(ph_raw, temp)
    return pres, soundvel, cond, temp, do, temp2, suna, fluo, turb, ph

def TEMP2_conversion(temp_raw2):
    """
    Converts raw temperature data from the DO sensor to physical units.
    """

    A = -1.191875e1
    B = 2.145289e1
    C = -3.611291
    D = 6.788267e-1

    # Assuming temp_raw2 is in V
    temp2 = A + B * temp_raw2 + C * temp_raw2**2 + D * temp_raw2**3

    return temp2


def DO_conversion(do_raw, temp, pres):
    """
    Converts raw dissolved oxygen data to physical units.

    """

    A = -4.329108e1
    B = 1.326965e2
    C = -3.459148e-1
    D = 1.011300e-2
    E = 3.9e-3
    F = 4.03e-5
    G = 0
    H = 1

    P_ = A/(1+D*(temp-25)+F*(temp-25)**2) + B/(do_raw*(1+D*(temp-25)+F*(temp-25)**2)+C)

    do = G + H*P_
    do = do*(1+E*(pres+10.1325)/100)  # Apply pressure correction in MPa (air+water column)

    return do

def SUNA_conversion(suna_raw):
    """
    Converts raw SUNA data to physical units.
    """
    # Not sur at all about these coefficients TO CHECK
    Vmax = 1.8621
    Vmin  = 0.3666
    DACmax = 39.56 
    DACmin = -0.2919

    A1 = (DACmax - DACmin)/(Vmax-Vmin)
    A0 = DACmin - A1 * Vmin

    # Assuming suna_raw is in mV
    # suna is Cnitrate in umol/L
    suna = A0 + A1 * suna_raw*1e-3

    suna = suna_raw*1e-3
    return suna

def TURBIDITY_conversion(turb_raw):

    """
    Converts raw turbidity data to physical units.
    """

    # for chlorophyll concentration
    # Scale_factor =6
    # Darkcounts = 0.091

    # for turbidity in NTU
    Scale_factor = 2
    Dark_counts = 0.098

    # Assuming turb_raw is in mV
    turb = Scale_factor * (turb_raw*1e-3 - Dark_counts)  

    return turb

def FLUO_conversion(fluo_raw):
    """
    Converts raw fluorometer data to physical units.
    """

    # for chlorophyll concentration in ug/l
    Scale_factor = 6
    Dark_counts = 0.091
    fluo = Scale_factor * (fluo_raw*1e-3 - Dark_counts)

    return fluo

def PH_conversion(ph_raw, temp):
    """
    Converts raw pH data to physical units.
    """

    pHslope = 4.6630
    pHoffset = 2.5330
    # Assuming ph_raw is in m
    pH = 7.0 + (ph_raw*1e-3 - pHoffset) / (pHslope * (temp+273.15) * 1.98416e-4)

    return pH


#
################################################################################
#
#
#
# Function remove surface waves : apply a butterworth filter to the profiles
# to remove the surface waves, set here to 0.2 Hz
# 
#   input:
#	data     : variable to filter
#	TIME        : time is counted in days since Yorig/1/1 (here 1950/1/1)
#
#   output:
#	data_final   : filtered data
#       fluo_chla   : fluorometer intensity
# 
################################################################################
#

def remove_surface_waves(data,TIME,f_s,f_c,order):
    data_final = np.zeros((data.shape[0],data.shape[1]))
    data_final[:] = np.nan

    for i_profile in range(data.shape[0]):
        if np.isnan(np.nanmean(TIME[i_profile,:]))==0:
            time_sampling=np.arange(np.nanmin(TIME[i_profile,:]),np.nanmax(TIME[i_profile,:]),1/f_s)
            time_sampling = time_sampling[np.where(time_sampling<=np.nanmax(TIME[i_profile,:]))[0]]

            ind = np.where((np.isnan(TIME[i_profile,:])==0) & (np.isnan(data[i_profile,:])==0))[0]
            if len(ind)>6:
                f1 = interpolate.interp1d(TIME[i_profile,ind], data[i_profile,ind],'linear',fill_value="extrapolate")
                W_sampling = f1(time_sampling)

                # Fourier transform
                #f_s = 10*f_s 
                #f_s = 25*10
                #f_c = 1   # Cut-off frequency in Hz
                #order = 10    # Order of the butterworth filter

                omega_c = f_c       # Cut-off angular frequency
                omega_c_d = omega_c / (f_s)    # Normalized cut-off frequency (digital)

                # Design the digital Butterworth filter
                b, a = butter(order, omega_c_d) 

                W_lisse = signal.filtfilt(b, a, signal.detrend(W_sampling))

                W_lisse = W_lisse + W_sampling - signal.detrend(W_sampling)
                ind = np.where(TIME[i_profile,:]<=np.nanmax(time_sampling))[0]
                f2 = interpolate.interp1d(time_sampling, W_lisse,'linear',fill_value="extrapolate")
                data_final[i_profile,ind] = f2(TIME[i_profile,ind])
                del ind, time_sampling, b, a, f1, f2, W_lisse, W_sampling, omega_c, omega_c_d

    return data_final


#
################################################################################
#
#
#
# Correct thermistor viscous heating
# 
#   input:
#	TEMP0     : In-situ Temperature
#	SAL_PRA0  : Practical Salinity
#	PRES0     : Pressure
#	LON0      : Longitude
#	LAT0      : Latitude
#	TIME        : time is counted in days since Yorig/1/1 (here 1950/1/1)
#
#   output:
#	TEMP1   : Corrected temperature
# 
################################################################################
#

def viscous_heating(TEMP0, SAL_PRA0, PRES0, LON0, LAT0, TIME):
    TEMP1 = np.zeros((TEMP0.shape[0], TEMP0.shape[1]))
    TEMP1[:] = np.nan
    for i in range(TEMP0.shape[0]):

        T = np.zeros(TEMP0.shape[1])
        S = np.zeros(TEMP0.shape[1])
        P = np.zeros(PRES0.shape[1])
        S = gsw.SA_from_SP(SAL_PRA0[i,:], PRES0[i,:], LON0[i,:], LAT0[i,:])
        T = TEMP0[i,:]

        T = T + 273.15;

        a = [-5.8002206e+03, 1.3914993e00, -4.8640239e-02, 4.1764768e-05, -1.4452093e-08, 6.5459673e+00]
        Pv_w = np.exp((a[0]/T) + a[1] + a[2]*T + a[3]*T**2 + a[4]*T**3 + a[5]*np.log(T))

        b  = [-4.5818e-4,-2.0443e-6]
        P0 = Pv_w*np.exp(b[0]*S+b[1]*S**2)/1e6

        T = TEMP0[i,:]
        P0[np.where(T<100)[0]] = 0.101325
        T68 = 1.00024*(T+273.15)

        S = SAL_PRA0[i,:]
        S_gkg=S

        P = PRES0[i,:]

        A = 5.328 - 9.76e-2*S + 4.04e-4*S**2
        B = -6.913e-3 + 7.351e-4*S - 3.15e-6*S**2
        C = 9.6e-6 - 1.927e-6*S + 8.23e-9*S**2
        D = 2.5e-9 + 1.666e-9*S - 7.125e-12*S**2
        cp_sw_P0 = 1000*(A + B*T68 + C*(T68**2) + D*(T68**3))

        c1 = -3.1118
        c2 = 0.0157
        c3 = 5.1014e-5
        c4 = -1.0302e-6
        c5 = 0.0107
        c6 = -3.9716e-5
        c7 = 3.2088e-8
        c8 = 1.0119e-9

        cp_sw_P = (P - P0)*(c1 + c2*T + c3*(T**2) + c4*(T**3) + S_gkg*(c5 + c6*T + c7*(T**2) + c8*(T**3)))

        cp = cp_sw_P0 + cp_sw_P
        del cp_sw_P0, cp_sw_P, P, P0, c1, c2, c3, c4, c5, c6, c7, c8, T, S, T68, S_gkg

        T = np.zeros(TEMP0.shape[1])
        S = np.zeros(TEMP0.shape[1])
        S = SAL_PRA0[i,:]
        T = TEMP0[i,:]
        S = S/1000;

        a = [1.5700386464e-01, 6.4992620050e01, -9.1296496657e+01, 4.2844324477e-05, 1.5409136040e+00, 1.9981117208e-02, -9.5203865864e-05, 7.9739318223e+00, -7.5614568881e-02, 4.7237011074e-04]

        mu_w = a[3] + 1/(a[0]*(T+a[1])**2+a[2])

        A  = a[4] + a[5]*T + a[6]*T**2
        B  = a[7] + a[8]*T + a[9]*T**2
        mu = mu_w*(1 + A*S + B*S**2)
        del mu_w, A, S, B, T, a

        T = np.zeros(TEMP0.shape[1])
        S = np.zeros(TEMP0.shape[1])
        S = SAL_PRA0[i,:]
        T = TEMP0[i,:]
        T68 = 1.00024*T
        S = S / 1.00472
        k = 10**(np.log10(240+0.0002*S)+0.434*(2.3-(343.5+0.037*S)/(T+273.15))*(1-(T+273.15)/(647.3+0.03*S))**(1/3)-3)
        del T, S, T68

        W = np.zeros(TEMP0.shape[1])
        W[1:-1] = (PRES0[i,2::]-PRES0[i,0:-2])/(TIME[i,2::]*24*3600-TIME[i,0:-2]*24*3600)
        Pr = cp*mu/k

        dT=0.80e-4*(W**2)*(Pr**(1/2))
        TEMP1[i,:] = TEMP0[i,:] - dT
        del dT, Pr, cp, mu, k, W
    return(TEMP1)

#
################################################################################
#
# Function vertical_interp: Interpolate each profile on a required variable
#
#   input:
#	Depth_mat    : original variable acquisition
#	Mat          : field to be interpolated
#	Depth_interp : variable on which the field is interpolated (regularly sampled)
#
#   output:
#
#	Mat_Z_interp        : interpolated field
# 
# 
################################################################################
#

def vertical_interp(Depth_mat,Mat,Depth_interp):

    Mat_Z_interp = np.zeros((Mat.shape[0],len(Depth_interp)))
    Mat_Z_interp[:] = np.nan
    
    for i in range(Mat_Z_interp.shape[0]):
        Depth_temp, ind = np.unique(Depth_mat[i,:],return_index=True)
        Mat_temp = Mat[i,ind]
        del ind
        Mat_temp = Mat_temp[np.where(np.isnan(Depth_temp)==0)[0]]
        Depth_temp = Depth_temp[np.where(np.isnan(Depth_temp)==0)[0]]
        Depth_temp = Depth_temp[np.where(np.isnan(Mat_temp)==0)[0]]
        Mat_temp = Mat_temp[np.where(np.isnan(Mat_temp)==0)[0]]
        if (len(Mat_temp)>2) & (len(Depth_temp)>2):
            ind = np.arange(np.where(Depth_interp>=np.nanmin(Depth_temp))[0][0], np.where(Depth_interp<=np.nanmax(Depth_temp))[0][-1])
            f1 = interpolate.interp1d(Depth_temp, Mat_temp,'linear')
            Mat_Z_interp[i,ind] = f1(Depth_interp[ind])
            #Mat_Z_interp[i,ind] = pchip_interpolate(Depth_temp, Mat_temp, Depth_interp[ind])
            del ind
        del Depth_temp, Mat_temp
    del i
    return Mat_Z_interp


#
################################################################################
#
#
#
# Function mvp_bin(pres,salt,temp,time_cycle,Pbin,bin_method)
#
# Do a binning of MVP data on regulat pressure levels 
# 
#   input:
#	pres         : pressure [dbar] 
#	fluo         : fluo data [mg m-3] 
#	salt         : Salinity [PSS-78]
#	temp         : Temperature [oC] 
#	time_cycle   : Time of each sample in days since Yorig/1/1 
#	Pbin         : pressure bins [dbar] 
#       bin_method   : binning method ('mean', 'median',...)
#       add_fluo     : Flag to process fluorometer
#
#   output:
#	pres         : pressure [dbar] 
#	fluo         : fluo data [mg m-3] 
#	salt         : Salinity [PSS-78]
#	temp         : Temperature [oC] 
#	time_cycle   : Time in days since Yorig/1/1 
# 
# 
################################################################################
#

def median(Depth_mat,Mat,Depth_interp):
    
    Mat_Z_interp = np.zeros((Mat.shape[0],len(Depth_interp)))
    Mat_Z_interp[:] = np.nan
    
    for i in range(Mat_Z_interp.shape[0]):
        Depth_temp, ind = np.unique(Depth_mat[i,:],return_index=True)
        Mat_temp = Mat[i,ind]
        del ind
        ind = np.argsort(Depth_temp, axis=0)
        Mat_temp=np.take_along_axis(Mat_temp, ind, axis=0)
        Depth_temp=np.take_along_axis(Depth_temp, ind, axis=0)
        del ind
        Mat_temp = Mat_temp[np.where(np.isnan(Depth_temp)==0)[0]]
        Depth_temp = Depth_temp[np.where(np.isnan(Depth_temp)==0)[0]]
        Depth_temp = Depth_temp[np.where(np.isnan(Mat_temp)==0)[0]]
        Mat_temp = Mat_temp[np.where(np.isnan(Mat_temp)==0)[0]]
        if (len(Mat_temp)>2) & (len(np.where(Depth_interp>=np.nanmin(Depth_temp))[0])>2) & (len(np.where(Depth_interp<=np.nanmax(Depth_temp))[0])>2):
            ind = np.arange(np.where(Depth_interp>=np.nanmin(Depth_temp))[0][0], np.where(Depth_interp<=np.nanmax(Depth_temp))[0][-1])
            if len(ind)>0:
                vout = st.binned_statistic(Depth_temp,Mat_temp,statistic='median', bins=Depth_interp[ind])
                Mat_Z_interp[i,ind[0:-1]] = vout.statistic
                del ind, vout
        del Depth_temp, Mat_temp
    del i
    return Mat_Z_interp

def Calc_dist_time(TIME1, LON1, LAT1, TIME2, LON2, LAT2):
    Dist = np.zeros((len(TIME1), len(TIME2)))
    Time = np.zeros((len(TIME1), len(TIME2)))
    R = 6373.0

    for i in range(len(TIME1)):
        lat1 = np.radians(LAT1[i])
        lon1 = np.radians(LON1[i])
        lat2 = np.radians(LAT2[:])
        lon2 = np.radians(LON2[:])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        Dist[i,:] = R * c * 1e3
        Time[i,:] = np.abs(TIME2[:]-TIME1[i])

    return Dist, Time




def filtering_tc(T,C,freq_echant,high_cutoff=1):


    sampling_frequency = freq_echant
    order = 2

    nyquist = sampling_frequency / 2.0
    normalized_cutoff = high_cutoff / nyquist

    b_bp, a_bp = butter(order,normalized_cutoff, btype='lowpass')


    # Filter only the valid (non-NaN) leading segment
    valid_idx = np.where(np.isfinite(T) & np.isfinite(C))[0]
    T_low = np.full_like(T, np.nan)
    C_low = np.full_like(C, np.nan)

    if valid_idx.size > 0:
        n_valid = valid_idx[-1] + 1
        T_low[:n_valid] = filtfilt(b_bp, a_bp, T[:n_valid])
        C_low[:n_valid] = filtfilt(b_bp, a_bp, C[:n_valid])
    return T_low,C_low


def temporal_lag(T,C,P,freq_echant):


    # Band-pass filter to keep frequencies between 0.1 Hz and 9 Hz
    low_cutoff = 0.1
    high_cutoff = 5
    sampling_frequency = freq_echant
    order = 1

    nyquist = sampling_frequency / 2.0
    normalized_low = low_cutoff / nyquist
    normalized_high = high_cutoff / nyquist

    b_bp, a_bp = butter(order, [normalized_low, normalized_high], btype='band')

    # Filter only the valid (non-NaN) leading segment
    valid_idx = np.where(np.isfinite(T) & np.isfinite(C))[0]
    T_high = np.full_like(T, np.nan)
    C_high = np.full_like(C, np.nan)

    if valid_idx.size > 0:
        n_valid = valid_idx[-1] + 1
        T_high[:n_valid] = filtfilt(b_bp, a_bp, T[:n_valid])
        C_high[:n_valid] = filtfilt(b_bp, a_bp, C[:n_valid])

    C_high = C_high - np.nanmean(C_high)
    T_high = T_high - np.nanmean(T_high)



    corr = correlate(T_high,
                    C_high,
                    mode='full')

    lags = correlation_lags(len(T_high), len(C_high), mode='full')

    corr = corr / (np.std(T_high) * np.std(C_high) * len(T_high))

    # lag optimal
    lag_samples = lags[np.argmax(np.abs(corr))]
    lag_time = lag_samples / 20  # 20 Hz

    # print("Lag (samples) =", lag_samples)
    # print("Lag (sec) =", lag_time)

    if lag_samples == 0:
        return T, gsw.SP_from_C(C,T,P)
    
    T_corr = T.copy()
    T_corr[:-lag_samples] = T[lag_samples:]
    T_corr[-lag_samples:] = T_corr[-lag_samples-1]
    S_corr = gsw.SP_from_C(C,T_corr,P)


    # t_shifted = Time + lag_time_sub
    # T_corr2 = np.interp(t_shifted, Time, T, left=np.nan, right=np.nan)
    # S_corr2 = gsw.SP_from_C(C,T_corr2,P)



    normalized_cutoff = 0.5 / freq_echant / 2.0
    b_bp, a_bp = butter(4,normalized_cutoff, btype='lowpass')
    n_valid = np.isfinite(S_corr)
    S_corr[n_valid] = filtfilt(b_bp, a_bp, S_corr[n_valid])

    return T_corr,S_corr




def bin_average(P,T,C,time,dp=0.05):
    if np.sum(np.isnan(P)) > 0:
        print('nan in pressure, cannot bin')
    if np.sum(np.isnan(T)) > 0:
        print('nan in temperature, cannot bin')
    if np.sum(np.isnan(C)) > 0:
        print('nan in conductivity, cannot bin')

    idx = np.argsort(P)
    P, T, C, time = P[idx], T[idx], C[idx], time[idx]


    bins = np.arange(P.min(), P.max(), dp)
    digitized = np.digitize(P, bins)

    P_bin = []
    T_bin = []
    C_bin = []
    time_bin = []

    for i in range(1, len(bins)):
        mask_bin = digitized == i
        if np.any(mask_bin):
            P_bin.append(P[mask_bin].mean())
            T_bin.append(T[mask_bin].mean())
            C_bin.append(C[mask_bin].mean())
            time_bin.append(time[mask_bin].mean())

    return (np.array(P_bin),
            np.array(T_bin),
            np.array(C_bin),
            np.array(time_bin))



def bin_average_v2(P,T,C,S,time,dp=0.05):
    if np.sum(np.isnan(P)) > 0:
        print('nan in pressure, cannot bin')
    if np.sum(np.isnan(T)) > 0:
        print('nan in temperature, cannot bin')
    if np.sum(np.isnan(C)) > 0:
        print('nan in conductivity, cannot bin')
    if np.sum(np.isnan(S)) > 0:
        print('nan in salinity, cannot bin')


    idx = np.argsort(P)
    P, T, C, S, time = P[idx], T[idx], C[idx], S[idx], time[idx]


    bins = np.arange(P.min(), P.max(), dp)
    digitized = np.digitize(P, bins)

    P_bin = []
    T_bin = []
    C_bin = []
    S_bin = []
    time_bin = []

    for i in range(1, len(bins)):
        mask_bin = digitized == i
        if np.any(mask_bin):
            P_bin.append(P[mask_bin].mean())
            T_bin.append(T[mask_bin].mean())
            C_bin.append(C[mask_bin].mean())
            S_bin.append(S[mask_bin].mean())
            time_bin.append(time[mask_bin].mean())

    return (np.array(P_bin),
            np.array(T_bin),
            np.array(C_bin),
            np.array(S_bin),
            np.array(time_bin))
