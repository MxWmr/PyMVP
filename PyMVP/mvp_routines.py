import numpy as np 
import scipy.stats as st 
from datetime import date
from datetime import datetime, timedelta
from scipy import interpolate
from scipy import signal
import gsw
from scipy.interpolate import interp1d
from netCDF4 import Dataset
from scipy.signal import butter, filtfilt, correlate, correlation_lags,savgol_filter
from typing import Optional
import re

def get_log(mvp_log_name,Yorig):

    """
    Read MPV log file to get starting and ending times of the cycle, latitude, longitude, and datetime

    Args:
        mvp_log_name : ASCII MVP Log file (MVP_2022xxx.log)
        Yorig        : time is counted in days since Yorig/1/1 (here 1950/1/1)

    Return:
        mvp_tstart  : Start of the dive in days since Yorig/1/1
        mvp_tend    : End of the dive in days since Yorig/1/1
        cycle_dur   : Duration of the cycle in seconds
        lat         : Latitude (float, if available)
        lon         : Longitude (float, if available)
        dt_station  : Datetime object (datetime.datetime) of the station (if available)
    """

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






def read_mvp_cycle_raw(mvp_dat_name):

    """
    Read one MVP cycle from a mvp_2022XXX.raw file

    Args:
        mvp_dat_name : ASCII MVP .raw file (mvp_2022xxx.raw)


    Return:

        pres        : pressure [dbar] 
        soundvel    : Sound velocity [m/s]
        do          :  dissolved oxygen [umol/kg]
        temp2       : Temperature from DO sensor [oC]
        suna        :  SUNA data [umol/kg]
        fluo        : Fluorometer data [ug/l]
        turb        : Turbidity data [NTU]
        ph          : pH data [pH units]

    """

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




def read_mvp_cycle_ncdf(mvp_dat_name):
    """
    Read one MVP cycle from a mvp_2022XXX.ncdf file

    Args:
        mvp_dat_name : NetCDF MVP .ncdf file (mvp_2022xxx.ncdf)


    Return:

        pres        : pressure [dbar] 
        soundvel    : Sound velocity [m/s]
        do          :  dissolved oxygen [umol/kg]
        temp2       : Temperature from DO sensor [oC]
        suna        :  SUNA data [umol/kg]
        fluo        : Fluorometer data [ug/l]
        turb        : Turbidity data [NTU]
        ph          : pH data [pH units]
    """

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







def time_mvp_cycle_up(args,mvp_tstart,mvp_tend):
    """
    Allocate time to each sample for a MVP cycle
    Select the ascending MVP cycle 

    Args:
        mvp_tstart: start of the dive in days since Yorig/1/1
        mvp_tend  : end of the dive in days since Yorig/1/1

    return:
        args: ist of outputs
        (données ascendantes pour chaque variable)
        time_cycle   : Time of each sample in days since Yorig/1/1    
    """

    # Allocate time to each data point
    N = np.size(args[0])
    time_cycle = np.linspace(mvp_tstart, mvp_tend, N)

    # Get only the ascending lines
    ibot = np.min(np.where(args[0] == args[0].max()))
    for i, arg in enumerate(args):
        args[i] = arg[ibot:]


    time_cycle_up = time_cycle[ibot:]

    return args + [time_cycle_up]


def time_mvp_cycle_down(args, mvp_tstart, mvp_tend):
    """
    Allocate time to each sample for a MVP cycle
    Select the descending MVP cycle

    Args:
        args: list of input to read
        mvp_tstart: start of the dive in days since Yorig/1/1
        mvp_tend  : end of the dive in days since Yorig/1/1

    Return:
        args: ist of outputs
        (données descendantes pour chaque variable)
        time_cycle_down: Time of each sample in days since Yorig/1/1

    """

    N = np.size(args[0])
    time_cycle = np.linspace(mvp_tstart, mvp_tend, N)

    # Trouver les indices pour la partie descendante
    ibot = np.min(np.where(args[0] == args[0].max()))

    for i, arg in enumerate(args):
        args[i] = arg[:ibot]

    time_cycle_down = time_cycle[:ibot]

    return args + [time_cycle_down]



def raw_data_conversion(pres, soundvel, cond, temp, do_raw, temp2_raw, suna_raw, fluo_raw, turb_raw, ph_raw):
    """
    Converts raw BGC sensor data (in V or mV) to physical units.
    Args:
        pres, soundvel : already in physical units
        do_raw, temp2_raw, suna_raw, fluo_raw, turb_raw, ph_raw : raw sensor data to convert

    Return:
        all inputs in physical units

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

    A = -4.670955e1
    B = 1.354704e2
    C = -3.317170e-1
    D = 1.003680e-2
    E = 4.1e-3
    F = 3.924e-5
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

    pHslope = 4.6331
    pHoffset = 2.5392
    # Assuming ph_raw is in m
    pH = 7.0 + (ph_raw*1e-3 - pHoffset) / (pHslope * (temp+273.15) * 1.98416e-4)

    return pH



def viscous_heating(TEMP0, SAL_PRA0, PRES0, LON0, LAT0, TIME):
    """
    Correct thermistor viscous heating
    Args:
        TEMP0     : In-situ Temperature
        SAL_PRA0  : Practical Salinity
        PRES0     : Pressure
        LON0      : Longitude
        LAT0      : Latitude
        TIME      : time is counted in days since Yorig/1/1 (here 1950/1/1)

    Returns:    
        TEMP1   : Corrected temperature
    """
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



def vertical_interp(Depth_mat,Mat,Depth_interp):
    """
    Interpolate each profile on a required variable (here depth).
    Args:
        Depth_mat (2D array): Original depth values for each profile (shape: n_profiles x n_depths)
        Mat (2D array): Field to be interpolated (shape: n_profiles x n_depths)
        Depth_interp (1D array): Variable on which the field is interpolated (regularly sampled depth levels)
        Returns:
        Mat_Z_interp (2D array): Interpolated field on the specified depth levels (shape: n_profiles x n_interp_depths)
"""

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








def Calc_dist_time(TIME1, LON1, LAT1, TIME2, LON2, LAT2):
    """
    Calculate the distance and time difference between two sets of points given their latitudes, longitudes, and times.
    Args:
    TIME1 (array): Time of the first set of points (in days since Yorig/1/1)
    LON1 (array): Longitudes of the first set of points 
    LAT1 (array): Latitudes of the first set of points
    TIME2 (array): Time of the second set of points (in days since Yorig/1/1)
    LON2 (array): Longitudes of the second set of points
    LAT2 (array): Latitudes of the second set of points
    Returns:
    Dist (2D array): Distance in meters between each pair of points from the two sets
    Time (2D array): Absolute time difference in seconds between each pair of points from the
    """
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
    """
    Filter temperature and conductivity data.
    A low-pass Butterworth filter is applied to both temperature and conductivity profiles to remove high-frequency noise.
    The cut-off frequency is set to 1 Hz, which is suitable for removing surface waves while preserving the main signal of interest in oceanographic profiles.
    The filter is applied only to the valid (non-NaN) leading segment of the profiles to avoid introducing artifacts in the NaN regions. The filtered profiles are returned with NaN values preserved in the same locations as the original profiles.
    Args:
        T (array): Temperature profile
        C (array): Conductivity profile
        freq_echant (float): Sampling frequency in Hz
        high_cutoff (float): Cut-off frequency for the low-pass filter in Hz (default is 1 Hz)
    """


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
    """
    Correction of the temporal lag between temperature and conductivity sensors by cross-correlation. 
    The lag is estimated on the band-pass filtered signals to focus on the high frequencies, but without electronic noise. 
    The lag is ofudn thanks to correlation between signal
    It is then applied to the original temperature profile before calculating salinity.
    The corrected salinity profile is then low-pass filtered to remove high-frequency noise introduced by the lag correction.
    Args:
        T (array): Temperature profile
        C (array): Conductivity profile
        P (array): Pressure profile
        freq_echant (float): Sampling frequency in Hz
    """


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
    """
    Bin average data by pressure.

    Args:
        P (array): Pressure profile (dbar)
        T (array): Temperature profile
        C (array): Conductivity profile
        time (array): Time profile
        dp (float): Pressure bin size

    Returns:
        tuple: Binned pressure, temperature, conductivity, and time profiles
    """
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
    """
    Bin average data by pressure.

    Args:
        P (array): Pressure profile (dbar)
        T (array): Temperature profile
        C (array): Conductivity profile
        S (array): Salinity profile
        time (array): Time profile
        dp (float): Pressure bin size

    Returns:
        tuple: Binned pressure, temperature, conductivity, salinity, and time profiles
    """
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







def align_profiles(P, T_ref, T_to_align_raw, min_depth=0,max_shift=20):
    """
    Pipeline complet :
    - estime ΔP
    - recale
    - estime ΔT
    - corrige


    Args:
        P (array): Pressure profile (dbar)
        T_ref (array): Reference temperature profile
        T_to_align_raw (array): Temperature profile to align
        min_depth (float): Minimum depth (in dbar) to consider for calculating mean differences
        max_shift (float): Maximum shift (in dbar) to consider for alignment

    """

    ### 1. calcul delta de pression

    # Masque pour exclure les valeurs non finies
    mask_nan = (
    np.isfinite(P) &
    np.isfinite(T_ref) &
    np.isfinite(T_to_align_raw)
    )

    P = P[mask_nan]
    T_ref = T_ref[mask_nan]
    T_to_align = T_to_align_raw[mask_nan]

    # Masque pour exclure la surface
    mask = P >= min_depth

    P = P[mask]
    T_ref = T_ref[mask]
    T_to_align = T_to_align[mask]

    # Lissage léger
    T1s = savgol_filter(T_ref, 11, 2)
    T2s = savgol_filter(T_to_align, 11, 2)

    # Gradients
    dT1 = np.gradient(T1s, P)
    dT2 = np.gradient(T2s, P)

    # Normalisation (important pour corrélation)
    dT1 = (dT1 - np.mean(dT1)) / np.std(dT1)
    dT2 = (dT2 - np.mean(dT2)) / np.std(dT2)

    # Corrélation
    corr = correlate(dT2, dT1, mode='full')
    lags = np.arange(-len(dT1)+1, len(dT1))

    # Convertir en décalage en pression
    dP = np.mean(np.diff(P))
    shifts = lags * dP

    # Limiter les shifts plausibles
    valid = np.abs(shifts) <= max_shift

    deltaP = shifts[valid][np.argmax(corr[valid])]

    ### 2. recalage pression
    f = interp1d(P + deltaP, T_to_align, bounds_error=False, fill_value=np.nan)
    T_shifted = f(P)

    ### 3. calcul delta de température
    mask = (P >= min_depth) & np.isfinite(T_ref) & np.isfinite(T_shifted)
    deltaT = np.median(T_shifted[mask] - T_ref[mask])

    ### 4. recalage thermique
    T_corrected = T_shifted - deltaT

    mask_corrected = np.isfinite(T_corrected)

    # copie pour ne pas modifier l'original directement
    T_out = T_to_align_raw.copy()

    # injection uniquement là où c’est valide
    T_out_indices = np.where(mask_nan)[0]
    T_out[T_out_indices[mask_corrected]] = T_corrected[mask_corrected]

    return T_out, deltaP, deltaT




def find_nearest_profile(time_mvp,Lat_mvp,Lon_mvp,time_ctd,Lat_ctd,Lon_ctd,mode):
    """
    Find the nearest CTD profile to each MVP profile based on time or spatial distance.
    Args:
        time_mvp (array): Time of the MVP profiles.
        Lat_mvp (array): Latitude of the MVP profiles.
        Lon_mvp (array): Longitude of the MVP profiles.
        time_ctd (array): Time of the CTD profiles.
        Lat_ctd (array): Latitude of the CTD profiles.
        Lon_ctd (array): Longitude of the CTD profiles.
        mode (str): Mode for finding nearest profile ('Dist' or 'Time').
    Returns:
        tuple: Index of the nearest CTD profile and the corresponding distance or time difference.
    """

    if mode=='Dist':
        idx = len(Lat_mvp)//2
        Lat_mvp = np.radians(Lat_mvp[idx])
        Lon_mvp = np.radians(Lon_mvp[idx])

        R = 6371.0

        min_dist = np.inf
        nearest_index = -1

        for i in range(len(Lat_ctd)):

            lat,lon = np.radians(Lat_ctd[i]), np.radians(Lon_ctd[i]) 
            mask = np.isfinite(lat) & np.isfinite(lon)
            lat,lon = lat[mask], lon[mask]
            lat,lon = lat[0],lon[0]

            dlon = lon - Lon_mvp 
            dlat = lat - Lat_mvp
            a = np.sin(dlat / 2)**2 + np.cos(Lat_mvp) * np.cos(lat) * np.sin(dlon / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            dist = R * c * 1e3  # Convert to meters
            if dist < min_dist:
                min_dist = dist
                nearest_index = i


        return nearest_index, min_dist
    
    elif mode=='Time':
        time_mvp = time_mvp[len(time_mvp)//2]  # Take the middle time of the MVP cycle as reference

        min_time_diff = np.inf
        nearest_index = -1
        for i in range(len(time_ctd)):
            time_diff = np.abs(time_ctd[i,-1] - time_mvp)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                nearest_index = i
        return nearest_index, min_time_diff
    
    else:
        raise ValueError("Mode should be 'Dist' or 'Time'")



def parse_SUNA(suna_file,m_line_frequency=20) :
    """
    Parse SUNA data from a given file.

    Args:
        suna_file (str): Path to the SUNA data file.
        m_line_frequency (int): Frequency of 'M' lines in the SUNA data file.

    Returns
    -------
          - lat          : float, decimal degrees (negative = South)
          - lon          : float, decimal degrees (negative = West)
          - header_datetime : datetime, from the file header
        One list per variable, with per D line found element:
          - m_timestamp  : list[datetime], estimated time of the preceding M line
          - pressure     : list[float], first value of the preceding M line (dbar)
          - dark         : list[float], 18th field after 'D' token
          - NO3_raw      : list[float], 21st field after 'D' token
          - spectre      : list[int],   second-to-last field of the D line (hex string)
          - d_line_raw   : list[str],   full raw D line (for debugging)
    """


    with open(suna_file, "r", errors="replace") as fh:
        raw = fh.read()
 
    # ------------------------------------------------------------------ header
    header, body = _split_header_body(raw)
    lat, lon = _parse_latlon(header)
    header_dt = _parse_header_datetime(header)
 
    # ------------------------------------------------------------------ body lines
    lines = body.splitlines()
    lines = _rejoin_split_d_lines(lines)
 
    l_m_timestamp = []
    l_pressure = []
    l_dark = []
    l_NO3_raw = []
    l_spectre = []
    l_d_line_raw = []

    m_line_index = -1          # counts M lines seen so far (0-based)
    last_m_pressure: Optional[float] = None
    last_m_index: int = -1     # index of the last M line seen
 
    for n,line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
 
        first_token = stripped.split()[0]
 
        # ---- M line --------------------------------------------------------
        if first_token == "M":
            m_line_index += 1
            last_m_index = m_line_index
            parts = stripped.split()
            try:
                last_m_pressure = float(parts[1])
            except (IndexError, ValueError):
                last_m_pressure = None
 
        # ---- D line --------------------------------------------------------
        elif first_token.startswith("D"):
            # D lines look like:  D 0xHEX,A,date,field4,...,field_n-1(hex_spectre),field_n
            # Split on comma after the leading "D 0xHEX" token
            # Rebuild as comma-separated fields (the first token has no comma)
            # e.g.  "D 0x7900,A,06/25/2026 00:04:38,0,-1.00,..."
            # → fields[0]="D 0x7900", fields[1]="A", fields[2]="06/25/2026 00:04:38", ...
 
            # Join token + rest after the space, then split on comma
            # but the raw line already has the 'D 0x...' together with commas
            fields = stripped.split(",")
            # fields[0] = "D 0x7900"  (or similar)
 
            try:
                dark   = float(fields[18])   # 19th field  (0-indexed: 18)
                NO3_raw = float(fields[21])  # 22nd field  (0-indexed: 21)
                spectre = fields[-2].strip() # second-to-last
                spectre_list = [int(spectre[i:i+4], 16) for i in range(0, len(spectre), 4)]
            except (IndexError, ValueError) as exc:
                # Malformed D line – skip but warn
                print(f"Warning: could not parse D line ({exc}): {stripped[:80]}")
                print(f"Raw line: {stripped} ")
                print(lines[n-1])
                print(line)
                print(lines[n+1])
                continue
 
            # Compute M-line timestamp
            m_timestamp: Optional[datetime] = None
            if header_dt is not None and last_m_index >= 0:
                dt_seconds = last_m_index / m_line_frequency
                m_timestamp = header_dt + timedelta(seconds=dt_seconds)
 
            l_m_timestamp.append(m_timestamp)
            l_pressure.append(last_m_pressure)
            l_dark.append(dark)
            l_NO3_raw.append(NO3_raw)
            l_spectre.append(spectre_list)
            l_d_line_raw.append(stripped)
 
    return np.array(l_m_timestamp), np.array(l_pressure,dtype=float), np.array(l_dark,dtype=float), np.array(l_NO3_raw,dtype=float), np.array(l_spectre), np.array(l_d_line_raw), lat, lon, header_dt
 
 
def _rejoin_split_d_lines(lines: list[str]) -> list[str]:
    """
    Rejoint les lignes D qui ont été coupées en deux par un saut de ligne parasite.
    Une ligne fragmentée ressemble à :
        ligne n   : "D"  ou  "D "
        ligne n+1 : "0x1234,A,..."   (commence par 0x)
    """
    result = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # Détecte un fragment : ligne qui vaut exactement "D" seul
        if stripped == "D" and i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            if next_stripped.startswith("0x"):
                # Fusionne les deux fragments
                result.append("D " + next_stripped)
                i += 2
                continue
        result.append(lines[i])
        i += 1
    return result


 
def _split_header_body(raw: str) -> tuple[str, str]:
    """Split file into header (up to <END_OF_HEADER>) and body."""
    marker = "<END_OF_HEADER>"
    idx = raw.find(marker)
    if idx == -1:
        return raw, ""
    end = idx + len(marker)
    return raw[:end], raw[end:]
 
 
def _parse_latlon(header: str) -> tuple[Optional[float], Optional[float]]:
    """Extract decimal lat/lon from header LAT/LON lines."""
    lat = lon = None
 
    # LAT ( ddmm.mmmmmmm,N):  3432.3192800,S
    lat_match = re.search(
        r"LAT\s*\(.*?\):\s*([\d.]+),([NS])", header, re.IGNORECASE
    )
    if lat_match:
        raw_lat = float(lat_match.group(1))
        hemi    = lat_match.group(2).upper()
        deg = int(raw_lat / 100)
        minutes = raw_lat - deg * 100
        lat = deg + minutes / 60.0
        if hemi == "S":
            lat = -lat
 
    # LON (dddmm.mmmmmmm,E): 01740.7099800,E
    lon_match = re.search(
        r"LON\s*\(.*?\):\s*([\d.]+),([EW])", header, re.IGNORECASE
    )
    if lon_match:
        raw_lon = float(lon_match.group(1))
        hemi    = lon_match.group(2).upper()
        deg = int(raw_lon / 100)
        minutes = raw_lon - deg * 100
        lon = deg + minutes / 60.0
        if hemi == "W":
            lon = -lon
 
    return lat, lon
 
 
def _parse_header_datetime(header: str) -> Optional[datetime]:
    """
    Extract the start datetime from the header using the Time + Date fields.
    Time (hh|mm|ss.s): 00:05:18.0
    Date (dd/mm/yyyy): 25/06/2026
    """
    time_match = re.search(r"Time\s*\(.*?\):\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", header)
    date_match = re.search(r"Date\s*\(.*?\):\s*(\d{2})/(\d{2})/(\d{4})", header)
 
    if not time_match or not date_match:
        return None
 
    hh, mm = int(time_match.group(1)), int(time_match.group(2))
    ss_frac = float(time_match.group(3))
    ss = int(ss_frac)
    us = int(round((ss_frac - ss) * 1e6))
 
    dd   = int(date_match.group(1))
    mo   = int(date_match.group(2))
    yyyy = int(date_match.group(3))
 
    return datetime(yyyy, mo, dd, hh, mm, ss, us)
 
 
