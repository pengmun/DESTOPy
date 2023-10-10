import numpy as np
import datetime
import math
from collections import defaultdict
from sgp4.api import Satrec
import spiceypy
from scipy.interpolate import RegularGridInterpolator
import h5py
import os 
from python_utils.rom_function import generateROM_JB2008 #, generateMLROM_JB2008, generateMLROM_JB2008_fullsw
from python_utils.JB2008_subfunc import JB2008
from scipy.integrate import ode
from scipy import interpolate
from numba import jit
import multiprocessing
from functools import partial
from contextlib import contextmanager
from IPython.core.debugger import set_trace
import time as timer
import psutil

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

def unstandard_data(xval,xdata_log10_mean,xdata_log10_std):
    gt_2 = np.swapaxes(xval[:,:,:,:],1,3)
    gt_data_unstandard = (gt_2*xdata_log10_std.reshape(45,90,25))+xdata_log10_mean.reshape(45,90,25) #2
    return gt_data_unstandard

def ML_standard_data(xdata,xdata_log10_mean,xdata_log10_std):
    xdata_log10 = np.log(xdata)
    xdata_standardized = (xdata_log10-xdata_log10_mean)/xdata_log10_std
    xdata_reshaped = np.reshape(xdata_standardized,(45,90,25,xdata_standardized.shape[1])) #103404 for 2008, else 105192
    xdata_reshaped = np.swapaxes(xdata_reshaped,0,3)
    xdata_reshaped = np.swapaxes(xdata_reshaped,1,2)
    return xdata_reshaped

def from_jd(jd: float, fmt: str = 'jd') -> datetime:
    """
    Converts a Julian Date to a datetime object.
    Algorithm is from Fliegel and van Flandern (1968)
    
    Inputs:
    _______
        jd      Julian Date as type specified in the string fmt
        fmt     String format
        
    Outputs:
    _______
        dt      datetime
    
    """
    jd, jdf = __from_format(jd, fmt)

    l = jd+68569
    n = 4*l//146097
    l = l-(146097*n+3)//4
    i = 4000*(l+1)//1461001
    l = l-1461*i//4+31
    j = 80*l//2447
    k = l-2447*j//80
    l = j//11
    j = j+2-12*l
    i = 100*(n-49)+i+l

    year = int(i)
    month = int(j)
    day = int(k)

    # in microseconds
    frac_component = int(jdf * (1e6*24*3600))

    hours = int(frac_component // (1e6*3600))
    frac_component -= hours * 1e6*3600

    minutes = int(frac_component // (1e6*60))
    frac_component -= minutes * 1e6*60

    seconds = int(frac_component // 1e6)
    frac_component -= seconds*1e6

    frac_component = int(frac_component)

    dt = datetime.datetime(year=year, month=month, day=day,
                  hour=hours, minute=minutes, second=seconds, microsecond=frac_component)
    return dt

def __from_format(jd: float, fmt: str) -> (int, float):
    """
    Converts a Julian Day format into the "standard" Julian
    day format.
    
    Inputs:
    _______
        jd                  Julian Date as type specified in the string fmt
        fmt                 String format
        
    Outputs:
    _______
        (jd, fractional)    A tuple representing a Julian day.  The first number is the
                            Julian Day Number, and the second is the fractional component of the
                            day.  A fractional component of 0.5 represents noon.  Therefore
                            the standard julian day would be (jd + fractional + 0.5)
                            
    """
    if fmt.lower() == 'jd':
        # If jd has a fractional component of 0, then we are 12 hours into
        # the day
        return math.floor(jd + 0.5), jd + 0.5 - math.floor(jd + 0.5)
    elif fmt.lower() == 'mjd':
        return __from_format(jd + 2400000.5, 'jd')
    elif fmt.lower() == 'rjd':
        return __from_format(jd + 2400000, 'jd')
    else:
        raise ValueError('Invalid Format')

def get_julian_datetime(date):
    """
    Convert a datetime object into julian float.
    
    Inputs:
    _______
        date                datetime-object of date
        
    Outputs:
    _______
        julian_datetime     Julian calculated datetime.
    
    """

    # Ensure correct format
    if not isinstance(date, datetime.datetime):
        raise TypeError('Invalid type for parameter "date" - expecting datetime')
    elif date.year < 1801 or date.year > 2099:
        raise ValueError('Datetime must be between year 1801 and 2099')

    # Perform the calculation
    julian_datetime = 367 * date.year - int((7 * (date.year + int((date.month + 9) / 12.0))) / 4.0) + int(
        (275 * date.month) / 9.0) + date.day + 1721013.5 + (
                          date.hour + date.minute / 60.0 + date.second / math.pow(60,
                                                                                  2)) / 24.0 - 0.5 * math.copysign(
        1, 100 * date.year + date.month - 190002.5) + 0.5

    return julian_datetime

def inputSWnrlmsise(swfName):
    """
    Reads space weather file from CelesTrack and output space
    weather in the format for the NRLMSISE-00 atmosphere model
    
    Inputs:
    _______
        SWFNAME             Filepath to space weather file
        
    Outputs:
    _______
        SWMATDAILY          matrix for F10.7Daily, F10.7Average, magnetic index
                            Daily observed and predicted AP (8)
                            from start of historic to end of Daily predicted
                        
        SWMATMONTHLYPRED    matrix for Monthly predicted F10.7Daily, F10.7Average
                            Magnetic index and AP (8)
    
    """

    fid = open( swfName, 'r');
    lines = fid.readlines()

    n_daily_obs = int(lines[15][20:25]);

    SWaux = np.zeros((n_daily_obs, 11));

    for i in range (0,n_daily_obs):

        SWaux[i, 0] = float(lines[17+i][93:98]); # F10.7 Daily

        SWaux[i, 1] = float(lines[17+i][101:106]); # F10.7 Average

        SWaux[i, 2] = float(lines[17+i][79:82]); # Daily Magnetic index

        SWaux[i, 3:11] = ([float(lines[17+i][46:50]),float(lines[17+i][50:54]),float(lines[17+i][54:58]),float(lines[17+i][58:62]),
           float(lines[17+i][62:66]),float(lines[17+i][66:70]),float(lines[17+i][70:74]),float(lines[17+i][74:78])]); # Daily 3h APs

        if SWaux[i, 0] == 0:
            SWaux[i, 0] = SWaux[i, 1];

    idx = 17+i

    pdt_pnt = int(lines[idx+3][27:29]);

    SWmatDaily = np.zeros((n_daily_obs + pdt_pnt, 11));
    SWmatDaily[0:n_daily_obs, :] = SWaux;

    ik = 0;
    for i in range(n_daily_obs,n_daily_obs+pdt_pnt):
        ik = ik + 1

        SWmatDaily[i, 0] =  float(lines[idx+4+ik][93:98]); # F10.7 Daily

        SWmatDaily[i, 1] = float(lines[idx+4+ik][101:106]); # F10.7 Average

        SWmatDaily[i, 2] = float(lines[idx+4+ik][79:82]); # Daily Magnetic index

        SWmatDaily[i, 3:11] = ([float(lines[idx+4+ik][46:50]),float(lines[idx+4+ik][50:54]),float(lines[idx+4+ik][54:58]),float(lines[idx+4+ik][58:62]),
           float(lines[idx+4+ik][62:66]),float(lines[idx+4+ik][66:70]),float(lines[idx+4+ik][70:74]),float(lines[idx+4+ik][74:78])]); # Daily 3h APs

    idx = idx+4+ik+3

    mpd_pnt = int(lines[idx][29:31]);

    SWmatMonthlyPred = np.zeros((mpd_pnt, 2));

    for i in range (0,mpd_pnt):
        SWmatMonthlyPred[i, 0] =  float(lines[idx+2+i][93:98]); # F10.7 Daily
        SWmatMonthlyPred[i, 1] =  float(lines[idx+2+i][101:106]); # F10.7 Average
        # Daily Magnetic indeces are not available.

    return SWmatDaily, SWmatMonthlyPred

def inputSWtiegcm(swfName):
    """
    Reads space weather file from CelesTrack and output space
    weather in the format for the TIE-GCM model
    
    Inputs:
    _______
        SWFNAME             Filepath to space weather file
        
    Outputs:
    _______
        SWMATDAILY          matrix for F10.7Daily, F10.7Average, magnetic index
                            Daily observed and predicted Kp (8)
                            from start of historic to end of Daily predicted
                            
        SWMATMONTHLYPRED    matrix for Monthly predicted F10.7Daily and F10.7Average
    
    """

    fid = open( swfName, 'r');
    lines = fid.readlines()

    n_daily_obs = int(lines[15][20:25]);

    SWaux = np.zeros((n_daily_obs, 11));

    for i in range (0,n_daily_obs):

        SWaux[i, 0] = float(lines[17+i][93:98]); # F10.7 Daily

        SWaux[i, 1] = float(lines[17+i][101:106]); # F10.7 Average

#         SWaux[i, 2] = float(lines[17+i][79:82]); # Daily Magnetic index

        SWaux[i, 3:11] = np.asarray([float(lines[17+i][18:21]),float(lines[17+i][21:24]),float(lines[17+i][24:27]),float(lines[17+i][27:30]),
           float(lines[17+i][30:33]),float(lines[17+i][33:36]),float(lines[17+i][36:39]),float(lines[17+i][39:42])])/10; # Daily 3h Kp

        if SWaux[i, 0] == 0:
            SWaux[i, 0] = SWaux[i, 1];

    idx = 17+i

    pdt_pnt = int(lines[idx+3][27:29]);

    SWmatDaily = np.zeros((n_daily_obs + pdt_pnt, 11));
    SWmatDaily[0:n_daily_obs, :] = SWaux;

    ik = 0;
    for i in range(n_daily_obs,n_daily_obs+pdt_pnt):
        ik = ik + 1

        SWmatDaily[i, 0] =  float(lines[idx+4+ik][93:98]); # F10.7 Daily

        SWmatDaily[i, 1] = float(lines[idx+4+ik][101:106]); # F10.7 Average

#         SWmatDaily[i, 2] = float(lines[idx+4+ik][79:82]); # Daily Magnetic index

        SWmatDaily[i, 3:11] = np.asarray([float(lines[idx+4+ik][18:21]),float(lines[idx+4+ik][21:24]),float(lines[idx+4+ik][24:27]),float(lines[idx+4+ik][27:30]),
           float(lines[idx+4+ik][30:33]),float(lines[idx+4+ik][33:36]),float(lines[idx+4+ik][36:39]),float(lines[idx+4+ik][39:42])])/10; # Daily 3h Kp

    idx = idx+4+ik+3

    mpd_pnt = int(lines[idx][29:31]);

    SWmatMonthlyPred = np.zeros((mpd_pnt, 2));

    for i in range (0,mpd_pnt):
        SWmatMonthlyPred[i, 0] =  float(lines[idx+2+i][93:98]); # F10.7 Daily
        SWmatMonthlyPred[i, 1] =  float(lines[idx+2+i][101:106]); # F10.7 Average
        # Daily Magnetic indeces are not available.

    return SWmatDaily, SWmatMonthlyPred

def inputEOP_Celestrak_Full(EOPfilename):
    """
    Read Earth Orientation Parameters (EOP) from text file. The EOP file can be obtained from
    https://www.celestrak.com/SpaceData/EOP-All.txt

    Inputs:
    _______
        EOPfilename         Filepath to Earth Orientation Parameters file

    Outputs:
    _______
        EOPMat              Year
                            Month (01-12)
                            Day
                            Modified Julian Date (Julian Date at 0h UT minus 2400000.5)
                            x (arc seconds)
                            y (arc seconds)
                            UT1-UTC (seconds)
                            Length of Day (seconds)
                            dPsi (arc seconds)
                            dEpsilon (arc seconds)
                            dX (arc seconds)
                            dY (arc seconds)
                            Delta Atomic Time, TAI-UTC (seconds)

    """

    # Open file
    fid = open(EOPfilename, 'r');
    lines = fid.readlines()

    nofObs = int(lines[32][20:25])

    ## Read Earth Orientation Parameters
    EOPMat = np.loadtxt(EOPfilename,skiprows=34,max_rows=nofObs)

    return EOPMat.T

def readSOLFSMY(filename, startRow = 3, endRow = np.NaN):
    """
    Import solar data from a SOLFSMY text file as a matrix.
    
    Inputs:
    _______
        filename            Filepath to space weather file
        
    Outputs:
    _______
        SOLFSMY             Solar data
    """

    # Open file
    fid = open(filename, 'r');
    lines = fid.readlines()

    if np.isnan(endRow):
        endRow = int(lines[1][22:26])

    SOLFSMY = np.genfromtxt(filename,skip_header=startRow,max_rows=endRow)

    return (SOLFSMY[:,:-1]).T

def readDTCFILE(filename, startRow = 0, endRow = None):
    """
    Import magnetic (DSTDTC) data from a SOLFSMY text file as a matrix.
    
    Inputs:
    _______
        filename            Filepath to space weather file
        
    Outputs:
    _______
        DTCFILE             Magnetic (DSTDTC) data
    """

    DTCFILE = np.genfromtxt(filename,skip_header=0,max_rows=endRow)
    
    return DTCFILE[:,1:].T

def loc_gravLegendre_scaleFactor(maxdegree):
    """
    Internal function computing normalized associated legendre polynomials, P,
    via recursion relations for spherical harmonic gravity
    
    Inputs:
    _______
        maxdegree           Maximum degree for spherical harmonic gravity
        
    Outputs:
    _______
        scaleFactor         scale factor for normalized associated legendre polynomials
    """
        
    scaleFactor = np.zeros((maxdegree+3, maxdegree+3));
        
    # Seeds for recursion formula
    scaleFactor[0,0] = 0;
    scaleFactor[1,0] = 1;
    scaleFactor[1,1] = 0;
        
    for n in range(2,maxdegree+2):
        k = n + 1;
            
        for m in range(0,n):
            p = m + 1;
            # Scale Factor needed for normalization of dUdphi partial derivative
                
            if (n == m):
                scaleFactor[k-1,k-1] = 0;
            elif (m == 0):
                scaleFactor[k-1,p-1] = np.sqrt((n+1)*(n)/2);
            else:
                scaleFactor[k-1,p-1] = np.sqrt((n+m+1)*(n-m));
    return scaleFactor



def inputEOP_Celestrak(EOPfilename):
    """
    Read partial Earth Orientation Parameters (EOP) from text file. The EOP file can be obtained from
    https://www.celestrak.com/SpaceData/EOP-All.txt

    Inputs:
    _______
        EOPfilename         Filepath to Earth Orientation Parameters file

    Outputs:
    _______
        EOPMat              x (arc seconds)
                            y (arc seconds)
                            UT1-UTC (seconds)
                            Length of Day (seconds)
                            dPsi (arc seconds)
                            dEpsilon (arc seconds)

    """

    ## File processing
    file_EOP = open( EOPfilename, 'r');
    lines = file_EOP.readlines()

    # Read number of observed points
    nofObs = int(lines[32][20:25])

    # Initialize output
    EOPMat = np.zeros((nofObs,6));

    ## Read Earth Orientation Parameters
    for ind in range(1,nofObs+1):

        # Read PM-x
        EOPMat[ind-1,0] = float(lines[33+ind][17:26]); # arcsec

        # Read PM-y
        EOPMat[ind-1,1] = float(lines[33+ind][27:36]); # arcsec

        # Read UT1-UTC
        EOPMat[ind-1,2] = float(lines[33+ind][37:47]);

        # Read length of day
        EOPMat[ind-1,3] = float(lines[33+ind][48:58]);

        # Read dPsi
        EOPMat[ind-1,4] = float(lines[33+ind][59:68]);  # arcsec

        # Read dEps
        EOPMat[ind-1,5] = float(lines[33+ind][69:78]); # arcsec

    EOPMat[np.isnan(EOPMat)] = 0;
    EOPMat[:,0:2] = EOPMat[:,0:2]/3600*np.pi/180; # rad
    EOPMat[:,4:6] = EOPMat[:,4:6]/3600*np.pi/180; # rad

    return EOPMat

def loadSGP4():
    """
    Load SGP4 constants

    Inputs:
    _______
        -

    Outputs:
    _______
        SGP4 constants      tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2, opsmode, whichconst

    """
    
    opsmode = 'i';
    whichconst = 72;
    [tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2] = getgravc(whichconst);
    return tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2, opsmode, whichconst

def getgravc(whichconst):
    """
    Load gravitational constant

    Inputs:
    _______
        whichconst                  Gravitational option

    Outputs:
    _______
        gravitational constant      tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2

    """
    if whichconst==72:
        # ------------ wgs-72 constants ------------
        mu     = 398600.8;            #// in km3 / s2
        radiusearthkm = 6378.135;     #// km
        xke    = 60.0 / np.sqrt(radiusearthkm*radiusearthkm*radiusearthkm/mu);
        tumin  = 1.0 / xke;
        j2     =   0.001082616;
        j3     =  -0.00000253881;
        j4     =  -0.00000165597;
        j3oj2  =  j3 / j2;
    else:
        print('Unkown gravity option')
    
    return tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2

def getTLEsForEstimation(startYear, startMonth, startDay, endYear, endMonth, endDay, selectedObjects):
    """
    Read TLE data from file
    
    Inputs:
    _______
        startYear               Start year
        startMonth              Start month
        startDay                Start day
        endYear                 End year
        endMonth                End month
        endDay                  End day
        selectedObjects         ID of selected RSOs
        

    Outputs:
    _______
        TLE_satrecs_valid       TLE OE data
        TLE_jdsatepoch_valid    Epoch of associated TLE data
    
    """
    
    selectedObjects = np.sort(selectedObjects);
    
    # Read TLEs from "estimationObjects.tle"
    filename = 'TLEdata/estimationObjects.tle';
    fid = open( filename, 'r');
    lines = fid.readlines()

    # Count number of line-feeds and increase by one for EOF.
    numLines = int(len(lines)/2);

    # Using noradID as key
    TLE_satrecs = defaultdict(list)
    for ik in range(0,numLines):
        sat_temp = Satrec.twoline2rv(lines[2*ik], lines[2*ik+1])
        TLE_satrecs[sat_temp.satnum].append(sat_temp)

    # Filter TLEs on date
    jdStart = get_julian_datetime(datetime.datetime(startYear, startMonth, startDay,0,0,0));
    jdEnd   = get_julian_datetime(datetime.datetime(endYear, endMonth, endDay,0,0,0));

    TLE_satrecs_valid = defaultdict(list)
    TLE_jdsatepoch_valid = defaultdict(list)
    
    for idx in range(0,len(selectedObjects)):
        # print(selectedObjects[idx])
        tle_temp = TLE_satrecs.get(selectedObjects[idx])
        tle_jdsatepoch = []
        for idx2 in range(0,len(tle_temp)):
            tle_jdsatepoch.append(tle_temp[idx2].jdsatepoch)

        firstTLE = np.where(np.asarray(tle_jdsatepoch)>=jdStart)[0][0]
        lastTLE = np.where(np.asarray(tle_jdsatepoch)<=jdEnd)[0][-1]

        TLE_satrecs_valid[selectedObjects[idx]].append(tle_temp[firstTLE:lastTLE+1])
        TLE_jdsatepoch_valid[selectedObjects[idx]].append(np.asarray(tle_jdsatepoch)[firstTLE:lastTLE+1])
    
    return TLE_satrecs_valid, TLE_jdsatepoch_valid

def computeEOP_Celestrak( EOPMat, jdate ):
    """
    Reads Earth Orientation Parameters
    
    Inputs:
    _______
        EOPfilename

    Outputs:
    _______
        xp:    [arcsec]
        yp:    [arcsec]
        dut1:  [s]
        lod:   [s]
        ddpsi: [rad]
        ddeps: [rad]
    
    """
    
    ## Output initialization
    xp = 0;
    yp = 0;
    dut1 = 0;
    lod = 0;
    ddpsi = 0;
    ddeps = 0;
    dat = 26; # s, value for 1992 til July 1
    
    ## Set reference date
    # Julian Date of 1962 January 1 00:00:00
    jdate0 = 2437665.5;

    ## File processing
    row = int(np.floor(jdate-jdate0));

    if row<=EOPMat.shape[0]:
        # Read PM-x
        xp = EOPMat[row,0];

        # Read PM-y
        yp = EOPMat[row,1];

        # Read UT1-UTC
        dut1 = EOPMat[row,2];

        # Read length of day
        lod = EOPMat[row,3];

        # Read dPsi
        ddpsi = EOPMat[row,4];

        # Read dEps
        ddeps = EOPMat[row,5];

    ## Read DAT file
    # Leap seconds (Data from UTC-TAI.history)
    DAT = np.array([[       10,       11,       12,       13,       14,       15,       16,       17,       18,       19,       20,       21,       22,       23,       24,       25,       26,       27,       28,       29,       30,       31,       32,       33,       34,       35,       36,       37],
           [2441317.5,2441499.5,2441683.5,2442048.5,2442413.5,2442778.5,2443144.5,2443509.5,2443874.5,2444239.5,2444786.5,2445151.5,2445516.5,2446247.5,2447161.5,2447892.5,2448257.5,2448804.5,2449169.5,2449534.5,2450083.5,2450630.5,2451179.5,2453736.5,2454832.5,2456109.5,2457204.5,2457754.5],
           [2441499.5,2441683.5,2442048.5,2442413.5,2442778.5,2443144.5,2443509.5,2443874.5,2444239.5,2444786.5,2445151.5,2445516.5,2446247.5,2447161.5,2447892.5,2448257.5,2448804.5,2449169.5,2449534.5,2450083.5,2450630.5,2451179.5,2453736.5,2454832.5,2456109.5,2457204.5,2457754.5, np.Inf]])
    i=0;

    while ~((jdate<DAT[2,i])&(jdate>=DAT[1,i])):
        i = i+1;

    dat = DAT[0,i];
    
    return xp, yp, dut1, lod, ddpsi, ddeps, dat

def convtime(year, mon, day, hr, min, sec, timezone, dut1, dat):
    """
    Compute time parameters and julian century values

    """
    deg2rad = np.pi/180.0;

    jd =  get_julian_datetime(datetime.datetime(year,mon,day,0,0,0));
    mjd  = jd - 2400000.5;
    mfme = hr*60.0 + min + sec/60.0;

    localhr= timezone + hr;

    utc = localhr * 3600.0 + min * 60.0 + sec;

    ut1= utc + dut1;
    toff = datetime.timedelta(seconds=ut1)
    jdut1 = get_julian_datetime(datetime.datetime(year,mon,day,0,0,0) + toff);
    tut1= (jdut1 - 2451545.0  )/ 36525.0;

    tai= utc + dat;
    toff2 = datetime.timedelta(seconds=tai)
    jdtai = get_julian_datetime(datetime.datetime(year,mon,day,0,0,0) + toff2);

    tt= tai + 32.184;   # sec
    toff3 = datetime.timedelta(seconds=tt)
    jdtt = get_julian_datetime(datetime.datetime(year,mon,day,0,0,0) + toff3);
    ttt= (jdtt - 2451545.0  )/ 36525.0;
    
    return ttt

def precess ( ttt ):
    """
    Compute precession

    """
    # convert to rad
    convrt = np.pi / (180.0*3600.0);
    ttt2= ttt * ttt;
    ttt3= ttt2 * ttt;
    
    psia =             5038.7784*ttt - 1.07259*ttt2 - 0.001147*ttt3; 
    wa   = 84381.448                 + 0.05127*ttt2 - 0.007726*ttt3;
    ea   = 84381.448 -   46.8150*ttt - 0.00059*ttt2 + 0.001813*ttt3;
    xa   =               10.5526*ttt - 2.38064*ttt2 - 0.001125*ttt3;
    zeta =             2306.2181*ttt + 0.30188*ttt2 + 0.017998*ttt3; 
    theta=             2004.3109*ttt - 0.42665*ttt2 - 0.041833*ttt3;
    z    =             2306.2181*ttt + 1.09468*ttt2 + 0.018203*ttt3;
    
    # convert units to rad
    psia = psia  * convrt; # rad
    wa   = wa    * convrt;
    ea   = ea    * convrt;
    xa   = xa    * convrt;
    zeta = zeta  * convrt; 
    theta= theta * convrt;
    z    = z     * convrt;
    
    coszeta  = np.cos(zeta);
    sinzeta  = np.sin(zeta);
    costheta = np.cos(theta);
    sintheta = np.sin(theta);
    cosz     = np.cos(z);
    sinz     = np.sin(z);
    
    prec = np.zeros((3,3))
    # ----------------- form matrix  mod to j2000 -----------------
    prec[0,0] =  coszeta * costheta * cosz - sinzeta * sinz;
    prec[0,1] =  coszeta * costheta * sinz + sinzeta * cosz;
    prec[0,2] =  coszeta * sintheta;
    prec[1,0] = -sinzeta * costheta * cosz - coszeta * sinz;
    prec[1,1] = -sinzeta * costheta * sinz + coszeta * cosz;
    prec[1,2] = -sinzeta * sintheta;
    prec[2,0] = -sintheta * cosz;
    prec[2,1] = -sintheta * sinz;
    prec[2,2] =  costheta;
    
    return prec,psia, wa, ea, xa

def iau80in():
    """
    iau80 model input helper function

    """
    convrt= 0.0001 * np.pi / (180*3600.0);
    
    filename = 'Data/nut80.dat'
    nut80 = np.loadtxt(filename)

    iar80 = nut80[:,0:5];
    rar80 = nut80[:,5:9];

    for i in range (0,106):
        for j in range (0,4):
            rar80[i,j]= rar80[i,j] * convrt;
            
    return iar80,rar80

def rem(value,mod_value):
    """
    Mimic the rem function of Matlab

    """
    sign = np.sign(value)
    val_temp = np.mod( abs(value),mod_value )
    return sign*val_temp

def fundarg( ttt ):
    
    deg2rad = np.pi/180.0;
    
    # ---- determine coefficients for iau 2000 nutation theory ----
    ttt2 = ttt*ttt;
    ttt3 = ttt2*ttt;
    ttt4 = ttt2*ttt2;
    
    l = ((((0.064) * ttt + 31.310) * ttt + 1717915922.6330) * ttt) / 3600.0 + 134.96298139;
    l1 = ((((-0.012) * ttt - 0.577) * ttt + 129596581.2240) * ttt) / 3600.0 + 357.52772333;
    f = ((((0.011) * ttt - 13.257) * ttt + 1739527263.1370) * ttt) / 3600.0 + 93.27191028;
    d = ((((0.019) * ttt - 6.891) * ttt + 1602961601.3280) * ttt) / 3600.0 + 297.85036306;
    omega = ((((0.008) * ttt + 7.455) * ttt - 6962890.5390) * ttt) / 3600.0 + 125.04452222;

    l    = rem( l,360.0  )     * deg2rad; # rad
    l1   = rem( l1,360.0  )    * deg2rad;
    f    = rem( f,360.0  )     * deg2rad;
    d    = rem( d,360.0  )     * deg2rad;
    omega= rem( omega,360.0  ) * deg2rad;
            
    return l, l1, f, d, omega

def nutation ( ttt, ddpsi, ddeps ):
    """
    Compute nutation

    """
    
    deg2rad = np.pi/180.0;
    
    iar80,rar80 = iau80in()

    # ---- determine coefficients for iau 1980 nutation theory ----
    ttt2= ttt*ttt;
    ttt3= ttt2*ttt;

    meaneps = -46.8150 *ttt - 0.00059 *ttt2 + 0.001813 *ttt3 + 84381.448;
    meaneps = rem(meaneps/3600.0,360.0);
    meaneps = meaneps * deg2rad;

    l, l1, f, d, omega = fundarg(ttt);

    deltapsi= 0.0;
    deltaeps= 0.0;
    for i in range(105,-1,-1):
        tempval= iar80[i,0]*l + iar80[i,1]*l1 + iar80[i,2]*f + iar80[i,3]*d + iar80[i,4]*omega;
        deltapsi= deltapsi + (rar80[i,0]+rar80[i,1]*ttt) * np.sin( tempval );
        deltaeps= deltaeps + (rar80[i,2]+rar80[i,3]*ttt) * np.cos( tempval );

    # --------------- find nutation parameters --------------------
    deltapsi = rem( deltapsi + ddpsi, 2.0 * np.pi );
    deltaeps = rem( deltaeps + ddeps, 2.0 * np.pi );
    trueeps  = meaneps + deltaeps;

    cospsi  = np.cos(deltapsi);
    sinpsi  = np.sin(deltapsi);
    coseps  = np.cos(meaneps);
    sineps  = np.sin(meaneps);
    costrueeps = np.cos(trueeps);
    sintrueeps = np.sin(trueeps);

    nut = np.zeros((3,3))
    nut[0,0] =  cospsi;
    nut[0,1] =  costrueeps * sinpsi;
    nut[0,2] =  sintrueeps * sinpsi;
    nut[1,0] = -coseps * sinpsi;
    nut[1,1] =  costrueeps * coseps * cospsi + sintrueeps * sineps;
    nut[1,2] =  sintrueeps * coseps * cospsi - sineps * costrueeps;
    nut[2,0] = -sineps * sinpsi;
    nut[2,1] =  costrueeps * sineps * cospsi - sintrueeps * coseps;
    nut[2,2] =  sintrueeps * sineps * cospsi + costrueeps * coseps;
    
    return deltapsi, trueeps, meaneps, omega,nut

def teme2eciNew(rteme, vteme, ateme, ttt, ddpsi, ddeps):
    """ 
    Transforms a vector from the true equator mean equinox system,
    (teme) to the mean equator mean equinox (j2000) system

    """
    ateme = np.zeros((3,1))
    prec,psia, wa, ea, xa = precess (ttt)

    deltapsi, trueeps, meaneps, omega, nut = nutation  (ttt, ddpsi, ddeps );

    # ------------------------ find eqeg ----------------------
    # rotate teme through just geometric terms 
    eqeg = deltapsi* np.cos(meaneps);

    eqeg = rem (eqeg, 2.0*np.pi);

    eqe = np.zeros((3,3))
    eqe[0,0] =  np.cos(eqeg);
    eqe[0,1] =  np.sin(eqeg);
    eqe[0,2] =  0.0;
    eqe[1,0] = -np.sin(eqeg);
    eqe[1,1] =  np.cos(eqeg);
    eqe[1,2] =  0.0;
    eqe[2,0] =  0.0;
    eqe[2,1] =  0.0;
    eqe[2,2] =  1.0;

    tm = np.matmul(np.matmul(prec, nut), eqe.T);

    reci = np.matmul(tm , rteme);
    veci = np.matmul(tm , vteme);
    aeci = np.matmul(tm , ateme);
    
    return reci, veci, aeci

def convertTEMEtoJ2000(rteme, vteme, jdate, EOPMat):
    """
    Convert Cartesian position and velocity vector
    from True Equator Mean Equinox (TEME) reference frame to J2000
    reference frame
    
    """
    _, _, dut1, _, ddpsi, ddeps, dat  = computeEOP_Celestrak( EOPMat, jdate );

    date = from_jd(np.ceil(round(jdate * 1e7,7))/ 1e7);
    year = date.year;
    mon = date.month;
    day = date.day;
    hr = date.hour;
    min = date.minute;
    sec = date.second;

    timezone = 0;

    ttt = convtime ( year, mon, day, hr, min, sec, timezone, dut1, dat );

    reci, veci, _ = teme2eciNew(np.asarray(rteme), np.asarray(vteme), np.zeros((3,1)), ttt, ddpsi, ddeps);

    return reci, veci

@jit(nopython=True,nogil=True,cache=True)
def pv2ep_jit(rr, vv, mu):
    """
    Convert eci state vector to modified equinoctial elements

    """
    radius = np.linalg.norm(rr);
    hv = np.cross(rr, vv);
    hmag = np.linalg.norm(hv);
    p = hmag**2 / mu;
    rdotv = np.dot(rr, vv);
    rzerod = rdotv / radius;
    eccen = np.cross(vv, hv);
    uhat = rr / radius;
    vhat = (radius * vv - rzerod * rr) / hmag;
    eccen = eccen / mu - uhat;

    ## unit angular momentum vector
    hhat = hv / np.linalg.norm(hv);

    ## compute kmee and hmee
    denom = 1.0 + hhat[2];
    k = hhat[0] / denom;
    h = -hhat[1] / denom;

    ## construct unit vectors in the equinoctial frame
    fhat= np.zeros((3,1))
    fhat[0] = 1.0 - k**2 + h**2;
    fhat[1] = 2.0 * k * h;
    fhat[2] = -2.0 * k;

    ghat= np.zeros((3,1))
    ghat[0] = fhat[1];
    ghat[1] = 1.0 + k**2 - h**2;
    ghat[2] = 2.0 * h;

    ssqrd = 1.0 + k**2 + h**2;

    ## normalize
    fhat = fhat / ssqrd;
    ghat = ghat / ssqrd;

    ## compute fmee and gmee
    f = np.dot(eccen, fhat);
    g = np.dot(eccen, ghat);

    # compute true longitude
    cosl = uhat[0] + vhat[1];
    sinl = uhat[1] - vhat[0];
    l = np.arctan2(sinl, cosl);

    # load modified equinoctial orbital elements array
    EP= np.zeros((6,1))
    EP[0] = p;
    EP[1] = f;
    EP[2] = g;
    EP[3] = h;
    EP[4] = k;
    EP[5] = l;
    
    return(EP)

def generateObservationsMEE(objects,TLE_jdsatepoch,obsEpochs,GM_kms,EOPMat):
    """
    Generate observations in modified equinoctial
    elements at specified observation epochs
    
    """
    nofObjects = len(objects);
    nofObs = len(obsEpochs);
    meeObs = np.zeros((6*nofObjects,nofObs));
    key_list = list(objects.keys())
    for i in range(0,nofObjects):
        for j in range(0,nofObs):
            # Observation epoch
            obsEpoch = obsEpochs[j];
            # Find nearest newer TLE
            try:
                satrecIndex = np.where(TLE_jdsatepoch.get(key_list[i])>=obsEpoch)[1][0]
            except:
                pass
            diffObsTLEEpochMinutes = (obsEpoch - objects.get(key_list[i])[0][satrecIndex].jdsatepoch) * 24*60;

            # Compute SGP4 state at epoch
            _, rtemeObs ,vtemeObs = objects.get(key_list[i])[0][satrecIndex].sgp4(obsEpoch, 0)

            # # Convert to J2000
            rj2000, vj2000 = convertTEMEtoJ2000(rtemeObs, vtemeObs, obsEpoch, EOPMat);
            meeObs[6*i:6*(i+1),j] = pv2ep_jit(rj2000,vj2000,GM_kms).flatten();
    
    return meeObs

def IERS(eop,Mjd_UTC):
    """
    Calculate IERS Earth rotation parameters
    
    """
    # linear interpolation
    mjd = (np.floor(Mjd_UTC));
    i = np.where(mjd==eop[3,:])[0][0]

    preeop = eop[:,i];
    nexteop = eop[:,i+1];
    mfme = 1440*(Mjd_UTC-np.floor(Mjd_UTC));
    fixf = mfme/1440;

    # Setting of IERS Earth rotation parameters
    # (UT1-UTC [s], TAI-UTC [s], x ["], y ["])
    UT1_UTC = preeop[6]+(nexteop[6]-preeop[6])*fixf;
    TAI_UTC = preeop[12];
    
    return UT1_UTC, TAI_UTC

def timediff(UT1_UTC,TAI_UTC):
    """
    Calculate Time Difference
    
    """
    TT_TAI  = +32.184;          # TT-TAI time difference [s]
    UTC_TAI = -TAI_UTC;         # UTC-TAI time difference [s]
    TT_UTC  = TT_TAI-UTC_TAI;   # TT-UTC time difference [s]
    return TT_UTC

def iauGmst06(uta, utb, tta, ttb):
    """
    Calculate Greenwich Mean Sidereal Time
    
    """
    # TT Julian centuries since J2000.0.
    t = ((tta - 2451545) + ttb) / 36525;
    
    # Greenwich mean sidereal time, IAU 2006.
    gmst = iauAnp(iauEra00(uta, utb) + (0.014506 +
                 (4612.156534    + (1.3915817   +
                 (-0.00000044  + (-0.000029956 +
                 (-0.0000000368 )
                 * t) * t) * t) * t) * t) * 4.848136811095359935899141e-6);

    return gmst

def iauEra00(dj1, dj2):
    """
    Calculate Earth Rotation Angle at UT1
    
    """
    # Days since fundamental epoch.
    if (dj1 < dj2):
        d1 = dj1;
        d2 = dj2;
    else:
        d1 = dj2;
        d2 = dj1;
    
    t = d1 + (d2- 2451545);
    
    # Fractional part of T (days).
    f = np.mod(d1, 1) + np.mod(d2, 1);

    # Earth rotation angle at this UT1.
    theta = iauAnp(6.283185307179586476925287 * (f + 0.7790572732640 + 0.00273781191135448 * t));

    return theta

def iauAnp(a):
    w = np.mod(a, 6.283185307179586476925287);
    if (w < 0):
        w = w + 6.283185307179586476925287;
    return w

def movmean(value,window):
    """
    Calculate moving mean based on window size
    
    """
    rollmean = np.zeros((len(value),1))
    for ik in range(len(value)):
        if ik < window/2:
            rollmean[ik] = np.sum(value[:int(ik+window/2)])/(ik+window/2)
        elif (len(value)-ik) < window/2:
            rollmean[ik] = np.sum(value[int(ik-window/2):])/(len(value)-ik+window/2)
        else:
            rollmean[ik] = np.sum(value[int(ik-window/2):int(ik+window/2)])/window
    return rollmean

def computeJB2000SWinputs(year,doy,hour,minute,sec,SOLdata,DTCdata,eopdata,spice):
    """
    Compute space weather proxies in format for JB2008 atmosphere model
    
    """
    # Input: Datetime in UTC and space weather data
    # Output: space weather proxies in format for JB2008 atmosphere model
    

    # month,day,~,~,~ = days2mdh(year,doy);
    dt_temp = datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)
    month = dt_temp.month
    day = dt_temp.day

    JD_0 = get_julian_datetime(datetime.datetime(year,month,day,hour,minute,sec))

    MJD = JD_0-2400000.5

    # READ SOLAR INDICES
    # USE 1 DAY LAG FOR F10 AND S10 FOR JB2008
    JD = np.rint(JD_0)-1; # 1 day lag (JD=MJD+2400000.5)
    i = np.where(JD==SOLdata[2,:])[0][0];
    SOL = SOLdata[:,i];
    F10 = SOL[3];
    F10B = SOL[4];
    S10 = SOL[5];
    S10B = SOL[6];

    # USE 2 DAY LAG FOR M10 FOR JB2008
    SOL = SOLdata[:,i-1];
    XM10 = SOL[7];
    XM10B = SOL[8];

    # USE 5 DAY LAG FOR Y10 FOR JB2008
    SOL = SOLdata[:,i-4];
    Y10 = SOL[9];
    Y10B = SOL[10];

    # GEOMAGNETIC STORM DTC VALUE
    i = np.where((year==DTCdata[0,:]) & (np.floor(doy)==DTCdata[1,:]))[0][0];
    ii = int(np.floor(hour)+2);
    DSTDTC1 = DTCdata[ii,i]; #DTC(ii);
    if ii >= 25: # if hour >= 23
        DSTDTC2 = DTCdata[2,i+1]; # Take first hour next day
    else:
        DSTDTC2 = DTCdata[ii+1,i]; # Take next hour same day

    DSTDTC = np.interp(minute*60+sec, [0, 3600],[DSTDTC1, DSTDTC2]);

    # CONVERT POINT OF INTEREST LOCATION (RADIANS AND KM)
    # CONVERT LONGITUDE TO RA
    UT1_UTC,TAI_UTC = IERS(eopdata,MJD);
    TT_UTC = timediff(UT1_UTC,TAI_UTC)
    DJMJD0 = 2400000.5
    DATE = get_julian_datetime(datetime.datetime(year,month,day,0,0,0)) - DJMJD0
    TIME = (60*(60*hour+minute)+sec)/86400;
    UTC = DATE+TIME;
    TT = UTC+TT_UTC/86400;
    TUT = TIME+UT1_UTC/86400;
    UT1 = DATE+TUT;
    GWRAS = iauGmst06(DJMJD0, UT1, DJMJD0, TT);

    et  = spice.spiceypy.str2et( from_jd(UTC+DJMJD0).strftime("%Y %m %d %H %M %S"))
    rr_sun = spice.spiceypy.spkezr('Sun',et,'J2000','NONE', 'Earth')[0]
    rr_sun = rr_sun[0:3];
    ra_Sun  = np.arctan2(rr_sun[1], rr_sun[0]);
    dec_Sun = np.arctan2(rr_sun[2], np.sqrt(rr_sun[0]**2+rr_sun[1]**2));

    SUN = np.zeros((2,1))
    SUN[0]  = ra_Sun;
    SUN[1]  = dec_Sun;

    return MJD,GWRAS,SUN,F10,F10B,S10,S10B,XM10,XM10B,Y10,Y10B,DSTDTC

def computeSWinputs_JB2008(jd0,jdf,eopdata,SOLdata,DTCdata,spice):
    """
    Compute space weather inputs for ROM-JB2008 model
    
    """
    # Output hourly space weather
    tt = np.arange(jd0,1/24+jdf,1/24);
    nofPoints = len(tt);

    Inputs = np.zeros((24,nofPoints));
#     Inputs = np.zeros((37,nofPoints));
    y10_future = np.zeros((1,nofPoints));
    for i in range (0,nofPoints):
        # Date and time
        jdate = tt[i];
        # [yyUTC, mmUTC, ddUTC, hhUTC, mnmnUTC, ssUTC] = datevec(jdate-1721058.5);
        dt_jdf = from_jd(np.ceil(round(jdate * 1e7,7))/ 1e7);  # End date of TLE collection window 
        yyUTC = dt_jdf.year
        mmUTC = dt_jdf.month
        ddUTC = dt_jdf.day
        hhUTC = dt_jdf.hour
        mnmnUTC = dt_jdf.minute
        ssUTC = dt_jdf.second

        # doyUTC = day(datetime(yyUTC, mmUTC, ddUTC),'dayofyear');
        doyUTC = dt_jdf.timetuple().tm_yday

        UThrs = hhUTC + mnmnUTC/60 + ssUTC/3600;
        # Get JB2008 space weather data
        _,GWRAS,SUN,F10,F10B,S10,S10B,XM10,XM10B,Y10,Y10B,DSTDTC = computeJB2000SWinputs(yyUTC,doyUTC,hhUTC,mnmnUTC,ssUTC,SOLdata,DTCdata,eopdata,spice);

        # [jdate; doy; UThrs; F10; F10B; S10; S10B; XM10; XM10B; Y10; Y10B; DSTDTC; GWRAS; SUN(1); SUN(2)]
        Inputs[0:15,i] = [jdate, doyUTC, UThrs, F10, F10B, S10, S10B, XM10, XM10B, Y10, Y10B, DSTDTC, GWRAS, SUN[0][0], SUN[1][0]];

    # Smooth DSTDTC over 12 hours
    Inputs[11,:] = movmean(Inputs[11,:],12).flatten();
    # Add future (now+1hr) space weather data
    Inputs[15,:-1] = Inputs[11,1:]; # DSTDTC
    Inputs[16,:-1] = Inputs[3,1:]; # F10
    Inputs[17,:-1] = Inputs[5,1:]; # S10
    Inputs[18,:-1] = Inputs[7,1:]; # XM10
    Inputs[19,:-1] = Inputs[9,1:]; # Y10

    # Add future (now+1hr) space weather data for last epoch
    # [yyUTC, mmUTC, ddUTC, hhUTC, mnmnUTC, ssUTC] = datevec(jdf+1/24-1721058.5);
    dt_jdf = from_jd(round(jdate+1/24,7));  # End date of TLE collection window
    yyUTC = dt_jdf.year
    mmUTC = dt_jdf.month
    ddUTC = dt_jdf.day
    hhUTC = dt_jdf.hour
    mnmnUTC = dt_jdf.minute
    ssUTC = dt_jdf.second

    # doyUTC = day(datetime(yyUTC, mmUTC, ddUTC),'dayofyear');
    doyUTC = dt_jdf.timetuple().tm_yday

    # Get JB2008 space weather for last epoch
    _,_,_,F10,_,S10,_,XM10,_,Y10,_,DSTDTC = computeJB2000SWinputs(yyUTC,doyUTC,hhUTC,mnmnUTC,ssUTC,SOLdata,DTCdata,eopdata,spice);
    Inputs[15,-1] = DSTDTC; # DSTDTC
    Inputs[16,-1] = F10; # F10
    Inputs[17,-1] = S10; # S10
    Inputs[18,-1] = XM10; # XM10
    Inputs[19,-1] = Y10; # Y10

    # Add quadratic DSTDTC terms
    Inputs[20,:] = Inputs[11,:]**2; # DSTDTC^2 (now)
    Inputs[21,:] = Inputs[15,:]**2; # DSTDTC^2 (now+1hr)
    # Add mixed terms: DSTDTC*F10
    Inputs[22,:] = np.multiply(Inputs[11,:],Inputs[3,:]); # DSTDTC*F10 (now)
    Inputs[23,:] = np.multiply(Inputs[15,:],Inputs[16,:]); # DSTDTC*F10 (now+1hr)

# # Full nonlinear Space Weather Indices
#     Inputs[15,-1] = F10; # DSTDTC
#     Inputs[16,-1] = S10; # F10
#     Inputs[17,-1] = XM10; # S10
#     Inputs[18,-1] = DSTDTC; # XM10
#     y10_future[0,-1] = Y10; # Y10

#     # and non-linear inputs (F10**2, DSTDTC**2) for current and future
#     Inputs[19,:] = Inputs[3,:]**2
#     Inputs[20,:] = Inputs[15,:]**2
#     Inputs[21,:] = Inputs[5,:]**2
#     Inputs[22,:] = Inputs[16,:]**2
#     Inputs[23,:] = Inputs[7,:]**2
#     Inputs[24,:] = Inputs[17,:]**2
#     Inputs[25,:] = Inputs[11,:]**2
#     Inputs[26,:] = Inputs[18,:]**2
    
#     # and non-linear cross terms (F10*M10, DSTDTC*M10) for current and future
#     Inputs[27,:] = np.multiply(Inputs[3,:],Inputs[5,:])
#     Inputs[28,:] = np.multiply(Inputs[15,:],Inputs[16,:])
        
#     Inputs[29,:] = np.multiply(Inputs[5,:],Inputs[7,:])
#     Inputs[30,:] = np.multiply(Inputs[16,:],Inputs[17,:])
    
#     Inputs[31,:] = np.multiply(Inputs[5,:],Inputs[11,:])
#     Inputs[32,:] = np.multiply(Inputs[16,:],Inputs[18,:])
    
#     Inputs[33,:] = np.multiply(Inputs[7,:],Inputs[11,:])
#     Inputs[34,:] = np.multiply(Inputs[17,:],Inputs[18,:])
    
#     Inputs[35,:] = np.multiply(Inputs[9,:],Inputs[11,:])
#     Inputs[36,:] = np.multiply(y10_future[0,:],Inputs[18,:])
    
    return Inputs

def generateROMdensityModel(ROMmodel,r,jd0,jdf,spice):
    """
    Generate reduced-order density model
    
    """
    TA = {}
    f = h5py.File('ROMDensityModels/JB2008_1999_2010_ROM_r100.mat','r')
    for k, v in f.items():
        TA[k] = np.array(v).T

    # Compute reduced-order dynamic density model:
    # PhiC contains continuous-time dynamic and input matrices
    # Uh contains the POD spatial modes
    # Qrom is the covariance matrix of ROM prediction error
    PhiC, Uh, Qrom = generateROM_JB2008(TA,r);

    # Compute the space weather inputs in the estimation period
    # Read Earth orientation parameters
    eopdata = inputEOP_Celestrak_Full('Data/EOP-All.txt');

    # Read space weather data: solar activity indices
    SOLdata = readSOLFSMY('Data/SOLFSMY.txt');

    # Read geomagnetic storm DTC values
    DTCdata = readDTCFILE('Data/DTCFILE.txt');
    SWinputs = computeSWinputs_JB2008(jd0,jdf+1,eopdata,SOLdata,DTCdata,spice);

    # Maximum altitude of ROM-JB2008 density model (for higher altitudes
    # density is set to zero)
    maxAtmAlt = 800;

    # Setup of ROM Modal Interpolation
    sltm = TA['localSolarTimes'][0];
    latm = TA['latitudes'][0];
    altm = TA['altitudes'][0];
    n_slt = len(sltm);
    n_lat = len(latm);
    n_alt = len(altm);

    # Mean density
    Dens_Mean = TA['densityDataMeanLog'];

    # Generate full 3D grid in local solar time, latitude and altitude
    [SLTm,LATm,ALTm]=np.meshgrid(sltm,latm,altm);

    # Generate interpolant for each POD spatial mode in Uh
    F_U = {}
    F_U[r] = [];
    for i in range(0,r):
        Uhr = np.reshape(Uh[:,i],(n_slt,n_lat,n_alt), order='F'); # i-th left singular vector on grid
        F_U[i] = RegularGridInterpolator((sltm, latm, altm), Uhr, bounds_error=False,fill_value=None); # Create interpolant of Uhr

    # Generate interpolant for the mean density in DenS_Mean
    Mr = np.reshape(Dens_Mean,(n_slt,n_lat,n_alt), order='F');
    M_U = RegularGridInterpolator((sltm, latm, altm), Mr, bounds_error=False,fill_value=None);

    # Compute dynamic and input matrices
    AC = PhiC[:r,:r]/3600;
    BC = PhiC[:r,r:]/3600;

    return AC,BC,Uh,F_U,Dens_Mean,M_U,SLTm,LATm,ALTm,maxAtmAlt,SWinputs,Qrom

def generateMLROMdensityModel(ROMmodel,r,jd0,jdf,spice):
    """
    Generate reduced-order density model
    
    """
    TA = {}
    f = h5py.File('ROMDensityModels/JB2008_1999_2010_ROM_r100.mat','r')
    for k, v in f.items():
        TA[k] = np.array(v).T

    # Compute reduced-order dynamic density model:
    # PhiC contains continuous-time dynamic and input matrices
    # Uh contains the POD spatial modes
    # Qrom is the covariance matrix of ROM prediction error
#     PhiC, encoder_model, decoder_model, Qrom = generateMLROM_JB2008_fullsw(TA,r);
    PhiC, encoder_model, decoder_model, Qrom = generateMLROM_JB2008(TA,r);
    

    # Compute the space weather inputs in the estimation period
    # Read Earth orientation parameters
    eopdata = inputEOP_Celestrak_Full('Data/EOP-All.txt');

    # Read space weather data: solar activity indices
    SOLdata = readSOLFSMY('Data/SOLFSMY.txt');

    # Read geomagnetic storm DTC values
    DTCdata = readDTCFILE('Data/DTCFILE.txt');
    SWinputs = computeSWinputs_JB2008(jd0,jdf+1,eopdata,SOLdata,DTCdata,spice);

    # Maximum altitude of ROM-JB2008 density model (for higher altitudes
    # density is set to zero)
    maxAtmAlt = 800;

    # Setup of ROM Modal Interpolation
    sltm = TA['localSolarTimes'][0];
    latm = TA['latitudes'][0];
    altm = TA['altitudes'][0];
    n_slt = len(sltm);
    n_lat = len(latm);
    n_alt = len(altm);

    # # Mean density
    # Dens_Mean = TA['densityDataMeanLog'];

    # Generate full 3D grid in local solar time, latitude and altitude
    [SLTm,LATm,ALTm]=np.meshgrid(sltm,latm,altm);

    # Compute dynamic and input matrices
    AC = PhiC[:r,:r]/3600;
    BC = PhiC[:r,r:]/3600;

    return AC,BC,encoder_model,decoder_model,SLTm,LATm,ALTm,maxAtmAlt,SWinputs,Qrom

def generateMLROMdensityModel_GITM(ROMmodel,r,jd0,jdf,spice):
    """
    Generate reduced-order density model
    
    """
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    import tensorflow as tf
    
    tf.compat.v1.disable_eager_execution()
#     tf.config.threading.set_intra_op_parallelism_threads(1)
#     tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.get_logger().setLevel('ERROR')
    
    # load encoder and decoder
    model_path = 'Model/GITM_v5'
    
    model = keras.models.load_model(model_path)
    encoder_model = Model(inputs=model.input, outputs=model.layers[13].output)
    # encoder_model.summary()

    decode_input = Input(model.layers[14].input_shape[1:])
    decoder_model = decode_input
    for layer in model.layers[14:]:
        decoder_model = layer(decoder_model)
    decoder_model = Model(inputs=decode_input, outputs=decoder_model)
    # decoder_model.summary()
    
    # Load reduced-order model files
    GTIM_rom_data = np.load('Model/GITM_rom.npz')
    
    # PhiC contains continuous-time dynamic and input matrices
    PhiC = GTIM_rom_data["PhiC"]
    
    # Qrom is the covariance matrix of ROM prediction error
    Qrom = GTIM_rom_data["Qrom"]
    
    # Generate full 3D grid in local solar time, latitude and altitude
    SLTm = GTIM_rom_data["SLTm"]
    LATm = GTIM_rom_data["LATm"]
    ALTm = GTIM_rom_data["ALTm"]
    
    # Load SW input for August
    SWinputs = GTIM_rom_data["SWinputs"]
    
    # Maximum altitude of ROM-JB2008 density model (for higher altitudes
    # density is set to zero)
    maxAtmAlt = 800;

    # Compute dynamic and input matrices
    AC = PhiC[:r,:r]/3600;
    BC = PhiC[:r,r:]/3600;

    return AC,BC,encoder_model,decoder_model,SLTm,LATm,ALTm,maxAtmAlt,SWinputs,Qrom

def getDensityJB2008llajd(lon,lat,alt,jdate,eopdata,SOLdata,DTCdata,spice):
    """
    Compute density of a particular point in atmosphere based on JB2008 density model and space weather input
    
    """
    # Date and time
    dt_jdf = from_jd(np.ceil(round(jdate * 1e7,7))/ 1e7);  # End date of TLE collection window 
    yyUTC = dt_jdf.year
    mmUTC = dt_jdf.month
    ddUTC = dt_jdf.day
    hhUTC = dt_jdf.hour
    mnmnUTC = dt_jdf.minute
    ssUTC = dt_jdf.second

    doyUTC = dt_jdf.timetuple().tm_yday

    # Get JB2008 space weather data
    MJD,GWRAS,SUN,F10,F10B,S10,S10B,XM10,XM10B,Y10,Y10B,DSTDTC = computeJB2000SWinputs(yyUTC,doyUTC,hhUTC,mnmnUTC,ssUTC,SOLdata,DTCdata,eopdata,spice);

    SAT = np.zeros((3,1))
    XLON = np.deg2rad(lon); # Lon
    SAT[0] = np.mod(GWRAS + XLON, 2*np.pi);
    SAT[1] = np.deg2rad(lat); # Lat
    SAT[2] = alt;

    YRDAY = doyUTC + ((hhUTC*3600 + mnmnUTC*60 + ssUTC) / 86400)
    _, rho = JB2008(MJD,YRDAY,SUN.flatten(),SAT.flatten(),F10,F10B,S10,S10B,XM10,XM10B,Y10,Y10B,DSTDTC)
    rho = rho * 1e9; # to kg/km^3

    return rho

@jit(nopython=True,nogil=True,cache=True)
def ep2pv_jit(EP, mu):
    """
    Convert equinoctial elements to ECI cartesian coordinates
    Based on Orbital Mechanics with MATLAB by David Eagle, 2013
    
    Inputs:
    _______
        mu      gravitational constant (km**3/sec**2)
        EP(1)   semilatus rectum of orbit (kilometers)
        EP(2)   f equinoctial element
        EP(3)   g equinoctial element
        EP(4)   h equinoctial element
        EP(5)   k equinoctial element
        EP(6)   true longitude (radians)

    Outputs:
    _______
        rr      eci position vector (kilometers)
        vv      eci velocity vector (kilometers/second)
    """
    # unload equinoctial orbital elements
    p = EP[0];
    f = EP[1];
    g = EP[2];
    h = EP[3];
    k = EP[4];
    l = EP[5];

    sqrtmup = np.sqrt(mu / p);

    cosl = np.cos(l);

    sinl = np.sin(l);

    q = 1 + f * cosl + g * sinl;

    r = p / q;

    alphasqrd = h**2 - k**2;

    ssqrd = 1 + h**2 + k**2;

    # compute eci position vector

    rr = np.asarray(r / ssqrd) * \
                np.asarray([   cosl + alphasqrd * cosl + 2 * h * k * sinl,\
                    sinl - alphasqrd * sinl + 2 * h * k * cosl,\
                    2 * (h * sinl - k * cosl)\
                ]);

    # compute eci velocity vector

    vv = np.asarray(sqrtmup  / ssqrd)  *\
                np.asarray([   - sinl - alphasqrd * sinl + 2 * h * k * cosl - g\
                    + 2 * f * h * k - alphasqrd * g,\
                    cosl - alphasqrd * cosl - 2 * h * k * sinl + f\
                        - 2 * g * h * k - alphasqrd * f,\
                    2 * (h * cosl + k * sinl + f * h + g * k)\
                ]);

    return rr, vv

@jit(nopython=True)
def computeLegendrePolynomials_jit( phi, maxdeg ):
    """
    Compute normalized associated legendre polynomials P
    
    """
    P = np.zeros((maxdeg+3, maxdeg+3, len(phi)));
    cphi = np.cos(np.pi/2-phi);
    sphi = np.sin(np.pi/2-phi);

    # force numerically zero values to be exactly zero
    cphi[np.abs(cphi)<=np.finfo(np.float64).eps] = 0;
    sphi[np.abs(sphi)<=np.finfo(np.float64).eps] = 0;

    # Seeds for recursion formula
    P[0,0,:] = 1;            # n = 0, m = 0;
    P[1,0,:] = np.sqrt(3)*cphi; # n = 1, m = 0;
    P[1,1,:] = np.sqrt(3)*sphi; # n = 1, m = 1;

    for n in range (1,maxdeg+2):
        k = n + 1;

        for m_ in range(-1,n+1):
            p = m_ + 1;
            # Compute normalized associated legendre polynomials, P, via recursion relations
            # Scale Factor needed for normalization of dUdphi partial derivative

            if (n == m_):
                P[k,k,:] = np.sqrt(2*(n+1)+1)/np.sqrt(2*(n+1))*sphi*np.reshape(P[k-1,k-1,:],phi.T.shape).T;
            elif (m_ == -1):
                P[k,p,:] = (np.sqrt(2*(n+1)+1)/(n+1))*(np.sqrt(2*(n+1)-1)*cphi*np.reshape(P[k-1,p,:],phi.T.shape).T - ((n+1)-1)/np.sqrt(2*(n+1)-3)*np.reshape(P[k-2,p,:],phi.T.shape).T);
            else:
                P[k,p,:] = np.sqrt(2*(n+1)+1)/(np.sqrt((n+1)+(m_+1))*np.sqrt((n+1)-(m_+1)))*(np.sqrt(2*(n+1)-1)*cphi*np.reshape(P[k-1,p,:],phi.T.shape).T - np.sqrt((n+1)+(m_+1)-1)*np.sqrt((n+1)-(m_+1)-1)/np.sqrt(2*(n+1)-3)*np.reshape(P[k-2,p,:],phi.T.shape).T);
    return P

@jit(nopython=True)
def computeGravity_jit(p,maxdeg,P,C,S,smlambda,cmlambda,GM,Re,r,scaleFactor):
    """
    Compute Earth gravitational acceleration in Earth-centered Earth-fixed (ECEF) coordinates
    
    """
    rRatio   = Re/r;
    rRatio_n = rRatio;

    # initialize summation of gravity in radial coordinates
    dUdrSumN      = np.ones((1,1));
    dUdphiSumN    = np.zeros((1,1));
    dUdlambdaSumN = np.zeros((1,1));

    # summation of gravity in radial coordinates
    for n in range (1,maxdeg):
        k = n+1;
        rRatio_n      = rRatio_n*rRatio;
        dUdrSumM      = np.zeros((1,1));
        dUdphiSumM    = np.zeros((1,1));
        dUdlambdaSumM = np.zeros((1,1));
        for m_ in range (-1,n+1):
            j = m_+1;
            dUdrSumM      = dUdrSumM + np.reshape(P[k,j,:],r.T.shape).T*(C[k,j]*cmlambda[:,j] + S[k,j]*smlambda[:,j]);
            dUdphiSumM    = dUdphiSumM + ( (np.reshape(P[k,j+1,:],r.T.shape).T*scaleFactor[k,j]) - p[:,2]/(np.sqrt(p[:,0]**2 + p[:,1]**2))*(m_+1)*np.reshape(P[k,j,:],r.T.shape).T)*(C[k,j]*cmlambda[:,j] + S[k,j]*smlambda[:,j]);
            dUdlambdaSumM = dUdlambdaSumM + (m_+1)*np.reshape(P[k,j,:], r.T.shape).T*(S[k,j]*cmlambda[:,j] - C[k,j]*smlambda[:,j]);

        dUdrSumN      = dUdrSumN      + dUdrSumM*rRatio_n*(k+1);
        dUdphiSumN    = dUdphiSumN    + dUdphiSumM*rRatio_n; #correct
        dUdlambdaSumN = dUdlambdaSumN + dUdlambdaSumM*rRatio_n; #correct

    # gravity in spherical coordinates
    dUdr      = -GM/(r*r)*dUdrSumN;
    dUdphi    =  GM/r*dUdphiSumN;
    dUdlambda =  GM/r*dUdlambdaSumN;

    # gravity in ECEF coordinates
    aa_grav_x = ((1/r)*dUdr - (p[:,2]/(r*r*np.sqrt(p[:,0]**2 + p[:,1]**2)))*dUdphi)*p[:,0] \
        - (dUdlambda/(p[:,0]**2 + p[:,1]**2))*p[:,1];
    aa_grav_y = ((1/r)*dUdr - (p[:,2]/(r*r*np.sqrt(p[:,0]**2 + p[:,1]**2)))*dUdphi)*p[:,1] \
        + (dUdlambda/(p[:,0]**2 + p[:,1]**2))*p[:,0];
    aa_grav_z = (1/r)*dUdr*p[:,2] + ((np.sqrt(p[:,0]**2 + p[:,1]**2))/(r*r))*dUdphi;

    aa_grav_x = aa_grav_x.T
    aa_grav_y = aa_grav_y.T
    aa_grav_z = aa_grav_z.T

    # special case for poles
    atPole = np.abs(np.arctan2(p[:,2],np.sqrt(p[:,0]**2 + p[:,1]**2)))==np.pi/2;
    if np.any(atPole):
        idx = np.where(atPole)[0]
        for ik in range(len(idx)):
            aa_grav_x[idx[ik]] = 0;
            aa_grav_y[idx[ik]] = 0;
            aa_grav_z[idx[ik]] = (1/r[idx[ik]])*dUdr[0,idx[ik]]*p[idx[ik],2];

    return aa_grav_x.flatten(), aa_grav_y.flatten(), aa_grav_z.flatten()

def gc2gd(r,yr,mth,day,hr,min,sec,dt,tf,flag):
    """
    Converts from Geocentric to Geodetic quantities.
    
    Inputs:
    _______
        r       Geocentric altitude (km)
        yr      year, e.g. 1995
        mth     month, e.g. Jan=1, Feb=2, etc.
        day     day, e.g. 1-31
        hr      hour, e.g. 0-23
        min     minutes, e.g. 0-59
        sec     seconds, e.g. 0-59
        dt      sampling interval (sec)
        tf      run time (sec)
        flag    1 to make longitude from -360 to 360 degrees
        
    Outputs:
    _______
        long    longitude (deg)
        lat     latitude (deg)
        alt     Geodetic altitude (km)
        alp     right ascension (deg)
        gst     Sidereal time (deg)
    
    """
    # Time vector
    if dt == 0:
        t = 0;
    else:
        t=np.arange(0,tf+dt,dt)#[0:dt:tf]';


    # Flattening of Earth and Earth radius constants
    f=1/298.257;
    req=6378.14;

    # Magnitude of the position
    rmag=(r[:,0]**2+r[:,1]**2+r[:,2]**2)**(0.5);

    # Altitude
    delta=np.arcsin(r[:,2]/rmag);
    alt=rmag-req*(1-f*np.sin(delta)**2-(f**2/2)*(np.sin(2*delta)**2)*(req/rmag-0.25));

    # Latitude
    sinfd=(req/rmag)*(f*np.sin(2*delta)+f*f*np.sin(4*delta)*(req/rmag-0.25));
    lat=(delta+np.arcsin(sinfd))*180/np.pi;

    # Sideral time at Greenwich
    jdate = get_julian_datetime(datetime.datetime(yr,mth,day,hr,min,sec+t));
    tdays=jdate-2415020;
    jcent=tdays/36525;
    ut=((sec+t)/60/60+min/60+hr)*360/24;
    gst=99.6910+36000.7689*jcent+0.0004*jcent*jcent+ut;

    # Longitude
    alp=np.arctan2(r[:,1],r[:,0]);
    alp=np.unwrap(alp)*180/np.pi;
    long=alp-gst;

    # Make longitude from -360 to 360 degrees
    ll=long[0];
    if (ll<0):
        for k in range (0,1000):
            if (long[0]>0): break; 
            long=long+360;
    else:
        for k in range (0,1000):
            if (long[0]<360): break;
            long=long-360;

    return long,lat,alt,alp,gst

def getDensityROM(pos,jdate,romState,r,F_U,M_U,maxAtmAlt):
    """
    Compute density using reduced-order density model
    Based on Matlab code by David Gondelach, Massachusetts Institute of Technology, 2020
    
    Inputs:
    _______
        pos         position vectors in J2000
        jdate       Julian date
        romState    reduced-order density state
        r           number of reduced order modes [integer]
        F_U         interpolant of gridded reduced-order modes
        M_U         interpolant of gridded mean density
        maxAtmAlt   maximum altitude of ROM density model
        
    Outputs:
    _______
        rho         densities at positions giving by pos
    
    """

    # Number of position vectors
    n = pos.shape[1];

    # Date and time
    # [yy, mm, dd, hh, mnmn, ss] = datevec(jdate-1721058.5); # Year, month, day, hour, minute, seconds in UTC
    # UThrs = hh + mnmn/60 + ss/3600; # Hour of day in UTC
    dt_jdf = from_jd(np.ceil(round(jdate * 1e7,7))/ 1e7);  # End date of TLE collection window 
    yy = dt_jdf.year
    mm = dt_jdf.month
    dd = dt_jdf.day
    hh = dt_jdf.hour
    mnmn = dt_jdf.minute
    ss = dt_jdf.second

    UThrs = hh + mnmn/60 + ss/3600; # Hour of day in UTC

    # # Convert ECI position to longitude, latitude and altitude
    lon,lat,alt,_,_=gc2gd(pos.T,yy,mm,dd,hh,mnmn,ss,0,0,0);
    lon = (lon+180)%360-180

    # Local solar time
    lst = UThrs+lon/15;
    lst[lst>24] = lst[lst>24]-24;
    lst[lst<0] = lst[lst<0]+24;

    # Spatial modes
    UhI = np.zeros((n,r));
    for j in range (0,r):
        UhI[:,j] = F_U[j]((lst,lat,alt));

    # Mean density
    MI = M_U((lst,lat,alt));

    # Density
    rho = 10**(np.sum(UhI.T*romState,0)+MI.T);
    rho[alt>maxAtmAlt] = 0;
#     set_trace()
    
    return rho

def getDensityROM_ml(pos,jdate,romState,r,decoder_model,maxAtmAlt,xdata_log10_mean,xdata_log10_std):
    """
    Compute density using reduced-order density model
    Based on Matlab code by David Gondelach, Massachusetts Institute of Technology, 2020
    
    Inputs:
    _______
        pos         position vectors in J2000
        jdate       Julian date
        romState    reduced-order density state
        r           number of reduced order modes [integer]
        F_U         interpolant of gridded reduced-order modes
        M_U         interpolant of gridded mean density
        maxAtmAlt   maximum altitude of ROM density model
        
    Outputs:
    _______
        rho         densities at positions giving by pos
    
    """

    # Number of position vectors
    n = pos.shape[1];

    # Date and time
    # [yy, mm, dd, hh, mnmn, ss] = datevec(jdate-1721058.5); # Year, month, day, hour, minute, seconds in UTC
    # UThrs = hh + mnmn/60 + ss/3600; # Hour of day in UTC
    dt_jdf = from_jd(np.ceil(round(jdate * 1e7,7))/ 1e7);  # End date of TLE collection window 
    yy = dt_jdf.year
    mm = dt_jdf.month
    dd = dt_jdf.day
    hh = dt_jdf.hour
    mnmn = dt_jdf.minute
    ss = dt_jdf.second

    UThrs = hh + mnmn/60 + ss/3600; # Hour of day in UTC

    # # Convert ECI position to longitude, latitude and altitude
    lon,lat,alt,_,_=gc2gd(pos.T,yy,mm,dd,hh,mnmn,ss,0,0,0);
    lon = (lon+180)%360-180

    # Local solar time
    lst = UThrs+lon/15;
    lst[lst>24] = lst[lst>24]-24;
    lst[lst<0] = lst[lst<0]+24;
    
    x1_dec = decoder_model.predict(romState[:,0:1].T) #decoder_model.predict(romState.T)
    rho_log_field = unstandard_data(x1_dec,xdata_log10_mean,xdata_log10_std)
    rho_field = 10**(rho_log_field)
    
    # # For JB2008
    # sltm = np.linspace(0,24,24)
    # latm = np.linspace(-87.5,87.5,20)
    # altm = np.linspace(100,800,36)
    
    # For GITM
    # sltm = np.linspace(0,24,24)
    # latm = np.linspace(-87.5,87.5,20)
    # altm = np.linspace(100,800,36)
    localSolarTimes = np.linspace(2,358,90)/360*24 #np.linspace(0,24,24)
    latitudes = np.linspace(-89.5,89.5,180) #np.linspace(-87.5,87.5,20)
    altitudes = np.array([100000., 101565., 103150., 104772., 106450., 108205., 110066.,
           112067., 114258., 116712., 119526., 122832., 126729., 131182.,
           136223., 141881., 148174., 155108., 162682., 170884., 179696.,
           189097., 199061., 209559., 220565., 232050., 243985., 256344.,
           269096., 282214., 295667., 309427., 323463., 337747., 352249.,
           366944., 381805., 396809., 411937., 427169., 442489., 457882.,
           473338., 488846., 504397., 519984., 535600., 551242., 566903.,
           582582.]) #np.linspace(100,800,36)
    sltm = localSolarTimes[::2]
    latm = latitudes[::2]
    altm = altitudes[::2]

    rho_fn = RegularGridInterpolator((sltm, latm, altm), rho_field[0,:,:,:], bounds_error=False,fill_value=None);
    rho = rho_fn((lst,lat,alt));
    
    rho[alt>maxAtmAlt] = 0;

    return rho


@jit(nopython=True,cache=True)
def AccelPointMass_jit(r, s, GM):
    """
    Computes the perturbational acceleration due to a point mass.
    Based on Matlab code by M. Mahooti
    
    Inputs:
    _______
        r           Satellite position vector
        s           Point mass position vector
        GM          Gravitational coefficient of point mass
        
    Outputs:
    _______
        a           Acceleration (a=d^2r/dt^2)
    
    """
    # Relative position vector of satellite w.r.t. point mass 
    d = r - s;
    # Acceleration 
    a = -GM * ( d/(np.linalg.norm(d)**3) + s/(np.linalg.norm(s)**3) );
    
    return a

@jit(nopython=True,cache=True)
def AccelSolrad_jit(r,r_Sun,AoM,Cr,P0,AU, Re):
    """
    Computes the acceleration due to solar radiation pressure
    assuming the spacecraft surface normal to the Sun direction
    (Only cylindrical shadow model is implemented)
    Based on Matlab code by M. Mahooti
    
    Inputs:
    _______
        r           Spacecraft position vector
        r_Earth     Earth position vector (Barycentric)
        r_Moon      Moon position vector (geocentric)
        r_Sun       Sun position vector (geocentric)
        r_SunSSB    Sun position vector (Barycentric)
        Area        Cross-section
        mass        Spacecraft mass
        Cr          Solar radiation pressure coefficient
        P0          Solar radiation pressure at 1 AU
        AU          Length of one Astronomical Unit
        shm         Shadow model (geometrical or cylindrical)
        
    Outputs:
    _______
        a           Acceleration (a=d^2r/dt^2)
        
    Notes:
        r, r_sun, Area, mass, P0 and AU must be given in consistent units,
        e.g. m, m^2, kg and N/m^2.
    
    """
    bcor = r-r_Sun;
    #        Satellite wrt Sun       bcor       rb 
    
    nu = Cylindrical_jit(r,r_Sun, Re)
    
    # Acceleration
    a = nu*Cr*AoM*P0*(AU*AU)*bcor/(np.linalg.norm(bcor)**3);

    return a

@jit(nopython=True,cache=True)
def Cylindrical_jit(r, r_Sun, Re):
    """
    Computes the fractional illumination of a spacecraft in the
    vicinity of the Earth assuming a cylindrical shadow model
    Based on Matlab code by M. Mahooti
    
    Inputs:
    _______
        r           Spacecraft position vector
        r_Sun       Sun position vector (m)
        
    Outputs:
    _______
        nu          Illumination factor:
                    nu=0   Spacecraft in Earth shadow
                    nu=1   Spacecraft fully illuminated by the Sun
    
    """
    
    e_Sun = r_Sun / np.linalg.norm(r_Sun);   # Sun direction unit vector
    s     = np.sum(r*e_Sun) # np.dot ( r, e_Sun );      # Projection of s/c position 
    if ( (s>0) | (np.linalg.norm(r-s*e_Sun) > Re/1000 )):
        nu = 1;
    else:
        nu = 0;
    
    return nu

def computeDerivative_PosVelBcRom(t,xp,AC,BC,SWinputs,r,noo,svs,F_U,M_U,maxAtmAlt,et0,jdate0,highFidelity,GM, Re, C_gravmodel, S_gravmodel, gravdegree, sF_gravmod):
    """
    Computes the derivatives of objects position, velocity and BC, and reduced-order state
    
    Inputs:
    _______
        t           current time: seconds since et0 [s]
        xp          state vector: position and velocity (J2000) and BC of
                    multiple objects and reduced order density state
        AC          continuous-time state transition matrix for the reduced
                    order density state
        BC          continuous-time input matrix for the reduced order density
                    state dynamics
        SWinputs    Space weather inputs
        r           number of reduced order modes [integer]
        noo         number of objects [integer]
        svs         state size per object [integer]
        F_U         interpolant of gridded reduced-order modes
        M_U         interpolant of gridded mean density
        maxAtmAlt   maximum altitude of ROM density model
        et0         initial ephemeris time (seconds since J2000 epoch)
        jdate0      initial Julian date
     
    Outputs:
    _______
        f          Time derivative of xp: dxp/dt
        
    Based on Matlab code by David Gondelach, Massachusetts Institute of Technology, 2020
    
    """
    ## LOAD KERNELS, GRAVITY MODEL, EARTH ORIENTATION PARAMETERS AND SGP4
    # Load SPICE kernels and ephemerides
    spiceypy.kclear()
    spiceypy.furnsh("Data/kernel.txt")

    # Convert state from single column to multi-column matrix
    x = np.reshape(xp,(svs*noo+r,-1), order='F');

    n = x.shape[1]

    # Date and time
    et = et0 + t; # Ephemeris time
    jdate = jdate0 + t / 86400; # Julian date

    # Space weather inputs for current time
    SWinputs_fn = interpolate.interp1d(SWinputs[0,:], SWinputs[1:,:])
    SWinputs = SWinputs_fn(jdate)

    # State derivative f=dx/dt
    f=np.zeros_like(x)

    # J2000 to ECEF transformation matrix
    xform = spiceypy.spiceypy.sxform('J2000', 'ITRF93', et ); # J2000 to ECEF transformation matrix

    # Object states in ECI
    x_eci = np.reshape(x[:-r,:],(svs,-1), order='F');
    # Object states in ECEF
    x_ecef = xform@x_eci[0:6,:]; # State in ECEF
    rr_ecef = x_ecef[0:3,:]; # Position in ECEF
    vv_ecef = x_ecef[3:6,:]; # Velocity in ECEF
    mag_v_ecef = np.sqrt( np.sum( vv_ecef**2, 0 )); # Magnitude of velocity in ECEF

    # Gravitational accelerations in ECEF [m/s^2]
    nofStates = rr_ecef.shape[1];
    partSize = 600;
    nofParts = int(np.floor( nofStates / partSize ));
    aa_grav_ecef_x = np.zeros((nofStates,1));
    aa_grav_ecef_y = np.zeros((nofStates,1));
    aa_grav_ecef_z = np.zeros((nofStates,1));

    for i in range(0,nofParts):
        aa_grav_ecef_x[(i)*partSize:(i+1)*partSize,0], aa_grav_ecef_y[(i)*partSize:(i+1)*partSize,0], aa_grav_ecef_z[(i)*partSize:(i+1)*partSize,0] = \
            computeEarthGravitationalAcceleration_jit(rr_ecef[:,(i)*partSize:(i+1)*partSize].T*1000, GM, Re, C_gravmodel, S_gravmodel, gravdegree, sF_gravmod);

    aa_grav_ecef_x[nofParts*partSize:,0], aa_grav_ecef_y[nofParts*partSize:,0], aa_grav_ecef_z[nofParts*partSize:,0] = \
            computeEarthGravitationalAcceleration_jit(rr_ecef[:,nofParts*partSize:].T*1000, GM, Re, C_gravmodel, S_gravmodel, gravdegree, sF_gravmod);

    # Gravitational accelerations in ECI [km/s^2]
    aa_grav_eci = xform[:3,:3].T @ np.vstack((np.vstack((aa_grav_ecef_x.T, aa_grav_ecef_y.T)), aa_grav_ecef_z.T)) / 1000;

    # Reduced order density state
    romState = x[-r:,:];
    romState_rep = np.repeat(romState.T.reshape(-1,1),noo,axis = 1).T.reshape(-1,r).T

    # Compute accelerations per object
    # Extract out RSO states
    a = np.reshape(np.arange(noo*7),(noo,-1))
    b = a[:,:3]
    
    x2 = x[b].transpose(1, 0, 2).reshape(3,-1)
    rho = getDensityROM(x2,jdate,romState_rep,r,F_U,M_U,maxAtmAlt).reshape(noo,-1);

    # Ballistic coefficients (BC) [m^2/(1000kg)]
    b_star = x[svs-1:-r:svs,:]

    # BC * density * velocity
    BCrhoV = np.reshape(b_star*rho,(1,-1), order = 'F') * mag_v_ecef;

    # Drag accelerations in ECEF [km/s^2]
    aa_drag_ecef = np.zeros_like(vv_ecef)
    aa_drag_ecef[0,:] = - 1/2*BCrhoV*vv_ecef[0,:]; # ECEF x-direction
    aa_drag_ecef[1,:] = - 1/2*BCrhoV*vv_ecef[1,:]; # ECEF y-direction
    aa_drag_ecef[2,:] = - 1/2*BCrhoV*vv_ecef[2,:]; # ECEF z-direction

    # Drag accelerations in ECI [km/s^2]
    aa_drag_eci = xform[:3,:3].T @ aa_drag_ecef;

    # Total accelerations in ECI [km/s^2]
    aa_grav_drag_eci = np.reshape(aa_grav_eci + aa_drag_eci, (3*noo,n), order = 'F');

    # Time derivatives of position and velocity due to velocity and gravity and drag accelerations
    b_flat = b.flatten()
    c = np.arange(aa_grav_drag_eci.shape[0])
    # Velocities in J2000 frame [km/s]
    f[b_flat,:] = x[b_flat+3,:]
    f[b_flat+3,:] = aa_grav_drag_eci[c,:]

    # Time derivative of ballistic coefficients is zero
    f[6:svs:-r,:] = 0;

    # If high fidelity, add Sun, Moon and SRP perturbations
    if highFidelity:

        ### Compute Sun Moon ###
        # Sun position in J2000 ref frame
        rr_Sun = spiceypy.spiceypy.spkezr('Sun',et,'J2000','NONE', 'Earth');
        rr_Sun = rr_Sun[0][0:3];

        # Moon position in J2000 ref frame
        rr_Moon = spiceypy.spiceypy.spkezr('Moon',et,'J2000','NONE', 'Earth');
        rr_Moon = rr_Moon[0][0:3];
        
        f = moon_sun_perturb_hf_jit(x, svs, noo, n, rr_Sun, rr_Moon, f, AC, romState, r, BC, SWinputs, Re)

    return f.flatten()


def computeDerivative_PosVelBcRom_ml(t,xp,AC,BC,SWinputs,r,noo,svs,decoder_model,maxAtmAlt,et0,jdate0,highFidelity,GM, Re, C_gravmodel, S_gravmodel, gravdegree, sF_gravmod,xdata_log10_mean,xdata_log10_std):
    """
    Computes the derivatives of objects position, velocity and BC, and reduced-order state
    
    Inputs:
    _______
        t           current time: seconds since et0 [s]
        xp          state vector: position and velocity (J2000) and BC of
                    multiple objects and reduced order density state
        AC          continuous-time state transition matrix for the reduced
                    order density state
        BC          continuous-time input matrix for the reduced order density
                    state dynamics
        SWinputs    Space weather inputs
        r           number of reduced order modes [integer]
        noo         number of objects [integer]
        svs         state size per object [integer]
        F_U         interpolant of gridded reduced-order modes
        M_U         interpolant of gridded mean density
        maxAtmAlt   maximum altitude of ROM density model
        et0         initial ephemeris time (seconds since J2000 epoch)
        jdate0      initial Julian date
     
    Outputs:
    _______
        f          Time derivative of xp: dxp/dt
        
    Based on Matlab code by David Gondelach, Massachusetts Institute of Technology, 2020
    
    """
#     ## LOAD KERNELS, GRAVITY MODEL, EARTH ORIENTATION PARAMETERS AND SGP4
#     # Load SPICE kernels and ephemerides
#     spiceypy.kclear()
#     spiceypy.furnsh("Data/kernel.txt")

    # Convert state from single column to multi-column matrix
    x = np.reshape(xp,(svs*noo+r,-1), order='F');

    n = x.shape[1]

    # Date and time
    et = et0 + t; # Ephemeris time
    jdate = jdate0 + t / 86400; # Julian date

    # Space weather inputs for current time
    SWinputs_fn = interpolate.interp1d(SWinputs[0,:], SWinputs[1:,:])
    SWinputs = SWinputs_fn(jdate)

    # State derivative f=dx/dt
    f=np.zeros_like(x)

    # J2000 to ECEF transformation matrix
    xform = spiceypy.spiceypy.sxform('J2000', 'ITRF93', et ); # J2000 to ECEF transformation matrix

    # Object states in ECI
    x_eci = np.reshape(x[:-r,:],(svs,-1), order='F');
    # Object states in ECEF
    x_ecef = xform@x_eci[0:6,:]; # State in ECEF
    rr_ecef = x_ecef[0:3,:]; # Position in ECEF
    vv_ecef = x_ecef[3:6,:]; # Velocity in ECEF
    mag_v_ecef = np.sqrt( np.sum( vv_ecef**2, 0 )); # Magnitude of velocity in ECEF

    # Gravitational accelerations in ECEF [m/s^2]
    nofStates = rr_ecef.shape[1];
    partSize = 600;
    nofParts = int(np.floor( nofStates / partSize ));
    aa_grav_ecef_x = np.zeros((nofStates,1));
    aa_grav_ecef_y = np.zeros((nofStates,1));
    aa_grav_ecef_z = np.zeros((nofStates,1));

    for i in range(0,nofParts):
        aa_grav_ecef_x[(i)*partSize:(i+1)*partSize,0], aa_grav_ecef_y[(i)*partSize:(i+1)*partSize,0], aa_grav_ecef_z[(i)*partSize:(i+1)*partSize,0] = \
            computeEarthGravitationalAcceleration_jit(rr_ecef[:,(i)*partSize:(i+1)*partSize].T*1000, GM, Re, C_gravmodel, S_gravmodel, gravdegree, sF_gravmod);

    aa_grav_ecef_x[nofParts*partSize:,0], aa_grav_ecef_y[nofParts*partSize:,0], aa_grav_ecef_z[nofParts*partSize:,0] = \
            computeEarthGravitationalAcceleration_jit(rr_ecef[:,nofParts*partSize:].T*1000, GM, Re, C_gravmodel, S_gravmodel, gravdegree, sF_gravmod);

    # Gravitational accelerations in ECI [km/s^2]
    aa_grav_eci = xform[:3,:3].T @ np.vstack((np.vstack((aa_grav_ecef_x.T, aa_grav_ecef_y.T)), aa_grav_ecef_z.T)) / 1000;

    # Reduced order density state
    romState = x[-r:,:];
    romState_rep = np.repeat(romState.T.reshape(-1,1),noo,axis = 1).T.reshape(-1,r).T

    # Compute accelerations per object
    # Extract out RSO states
    a = np.reshape(np.arange(noo*7),(noo,-1))
    b = a[:,:3]
    
    x2 = x[b].transpose(1, 0, 2).reshape(3,-1)
    rho = getDensityROM_ml(x2,jdate,romState_rep,r,decoder_model,maxAtmAlt,xdata_log10_mean,xdata_log10_std).reshape(noo,-1);

    # Ballistic coefficients (BC) [m^2/(1000kg)]
    b_star = x[svs-1:-r:svs,:]

    # BC * density * velocity
    BCrhoV = np.reshape(b_star*rho,(1,-1), order = 'F') * mag_v_ecef;

    # Drag accelerations in ECEF [km/s^2]
    aa_drag_ecef = np.zeros_like(vv_ecef)
    aa_drag_ecef[0,:] = - 1/2*BCrhoV*vv_ecef[0,:]; # ECEF x-direction
    aa_drag_ecef[1,:] = - 1/2*BCrhoV*vv_ecef[1,:]; # ECEF y-direction
    aa_drag_ecef[2,:] = - 1/2*BCrhoV*vv_ecef[2,:]; # ECEF z-direction

    # Drag accelerations in ECI [km/s^2]
    aa_drag_eci = xform[:3,:3].T @ aa_drag_ecef;

    # Total accelerations in ECI [km/s^2]
    aa_grav_drag_eci = np.reshape(aa_grav_eci + aa_drag_eci, (3*noo,n), order = 'F');

    # Time derivatives of position and velocity due to velocity and gravity and drag accelerations
    b_flat = b.flatten()
    c = np.arange(aa_grav_drag_eci.shape[0])
    # Velocities in J2000 frame [km/s]
    f[b_flat,:] = x[b_flat+3,:]
    f[b_flat+3,:] = aa_grav_drag_eci[c,:]

    # Time derivative of ballistic coefficients is zero
    f[6:svs:-r,:] = 0;

    # If high fidelity, add Sun, Moon and SRP perturbations
    if highFidelity:

        ### Compute Sun Moon ###
        # Sun position in J2000 ref frame
        rr_Sun = spiceypy.spiceypy.spkezr('Sun',et,'J2000','NONE', 'Earth');
        rr_Sun = rr_Sun[0][0:3];

        # Moon position in J2000 ref frame
        rr_Moon = spiceypy.spiceypy.spkezr('Moon',et,'J2000','NONE', 'Earth');
        rr_Moon = rr_Moon[0][0:3];
        
        f = moon_sun_perturb_hf_jit(x, svs, noo, n, rr_Sun, rr_Moon, f, AC, romState, r, BC, SWinputs, Re)

    return f.flatten()


@jit(nopython=True,cache=True)
def computeEarthGravitationalAcceleration_jit( rr_ecef, GM, Re, C, S, gravdegree, sF_gravmod):
    """
    Computes Earth gravitational acceleration using spherical harmonic expansion of Earth's
    geopotential in Earth-Centered Earth-Fixed coordinates [m/s^2].
    
    Inputs:
    _______
        rr_ecef     position in Earth-Centered Earth-Fixed (ECEF) coordinates
                    in meters [Mx3].
     
    Outputs:
    _______
        aa_grav_x   Earth gravitational acceleration in the x-direction of the
                    Earth-Centered Earth-Fixed coordinates in meters per second
                    squared [Mx1].
        aa_grav_y   Earth gravitational acceleration in the y-direction of the
                    Earth-Centered Earth-Fixed coordinates in meters per second
                    squared [Mx1].
        aa_grav_z   Earth gravitational acceleration in the z-direction of the
                    Earth-Centered Earth-Fixed coordinates in meters per second
                    squared [Mx1].
                    
    References:
        Vallado, D. A., "Fundamentals of Astrodynamics and Applications", 2001.
    
    """
    # Compute geocentric radius
    r = np.sqrt( np.sum( rr_ecef**2, 1 ));

    # Check if geocentric radius is less than equatorial radius
    if (r < Re).any():
        print('Radial position is less than Earth equatorial radius, %g.', Re);

    # Compute geocentric latitude
    phic = np.arcsin( rr_ecef[:,2]/ r );

    # Compute lambda
    lam = np.arctan2( rr_ecef[:,1], rr_ecef[:,0] );

    smlambda = np.zeros(( rr_ecef.shape[0], gravdegree+1 ));
    cmlambda = np.zeros(( rr_ecef.shape[0], gravdegree+1 ));

    slambda = np.sin(lam);
    clambda = np.cos(lam);
    smlambda[:,0] = 0;
    cmlambda[:,0] = 1;
    smlambda[:,1] = slambda;
    cmlambda[:,1] = clambda;

    for m in range(2,gravdegree+1):
        smlambda[:,m] = 2.0*clambda*smlambda[:, m-1] - smlambda[:, m-2];
        cmlambda[:,m] = 2.0*clambda*cmlambda[:, m-1] - cmlambda[:, m-2];

    # Compute normalized associated legendre polynomials
    P = computeLegendrePolynomials_jit( phic, gravdegree );

    scaleFactor = sF_gravmod;

    # Compute gravity in ECEF coordinates
    aa_grav_x, aa_grav_y, aa_grav_z = computeGravity_jit( rr_ecef, gravdegree, P, \
        C[:gravdegree+1, :gravdegree+1 ], S[:gravdegree+1, :gravdegree+1], \
        smlambda, cmlambda, GM, Re, r,scaleFactor );

    return aa_grav_x, aa_grav_y, aa_grav_z

@jit(nopython=True,cache=True)
def moon_sun_perturb_hf_jit(x, svs, noo, n, rr_Sun, rr_Moon, f, AC, romState, r, BC, SWinputs, Re):
    """
    Computes gravitational perturbation due to moon and sun
    
    """
    gravconst = 6.67259e-20; # [km^3/kg/s^2]
    GM_Sun    = gravconst*1.9891e30; # [kg]
    GM_Moon = gravconst*7.3477e22; # [kg]
    
    # Solar radiation
    AU          = 1.49597870691e8; # [km]
    L_sun       = 3.846e20; # [MW] Luminosity of the Sun -> 3.846e26 W -> W = kg m/s2 m/s -(to km)-> 1e-6 kg km/s2 km/s -> 1e-6 MW/W
    c_light     = 299792.458; # [km/s] speed of light
    P_sun       = L_sun/(4*np.pi*AU**2*c_light);
    C_R = 1.2;

    ### SRP and lunisolar perturbations ###
    for i in range (0,noo):
        aa_eci = np.zeros((3,n));
        for j in range (0,n):
            rr_sat = x[svs*(i):svs*(i)+3,j];

            # Solar radiation pressure
            b_star = x[svs*(i+1)-1,j];
            AoMSRP = b_star/2.2 * 1e-9; # Approximate area-to-mass ratio (assuming BC=Cd*AoM and Cd=2.2)
            # SRP acceleration
            aa_SRP = AccelSolrad_jit(rr_sat, rr_Sun,AoMSRP,C_R,P_sun,AU,Re);
            aa_eci[:,j] = aa_SRP;

            # Moon gravitational acceleration
            aa_Moon = AccelPointMass_jit(rr_sat,rr_Moon,GM_Moon);
            aa_eci[:,j] = aa_eci[:,j] + aa_Moon;

            # Sun gravitational acceleration
            aa_Sun = AccelPointMass_jit(rr_sat,rr_Sun,GM_Sun);
            aa_eci[:,j] = aa_eci[:,j] + aa_Sun;

        # Add SRP and lunisolar accelerations
        f[svs*(i)+3:svs*(i)+6,:] = f[svs*(i)+3:svs*(i)+6,:] + aa_eci;

    # Time derivative of reduced-order density state: dz/dt = Ac*z + Bc*u
    f[-r:,:]  = (AC @ romState).reshape(r,-1) + (BC @ SWinputs).reshape(r,-1);
    
    # Convert state derivative to single column
    f = np.reshape(f,(-1,1));
    return f

def Unscented_Transform(x0f,kappa0 = 0):
    """
    Compute weights for unscented transformation
    
    Based on Matlab code by P.M. Mehta, University of Minnesota, 2018 and
    David Gondelach, Massachusetts Institute of Technology, 2020
    
    Reference: Wan, E. A., & Van Der Merwe, R. (2001). The unscented Kalman filter, In: Kalman filtering and neural networks, pp. 221â280.
    
    """
    L=len(x0f);
    alpha = 1;
    beta = 2;
    kappa = 3 - L;
    if kappa0 != 0:
        kappa = kappa0;
    
    lambda_temp = alpha**2*(L + kappa) - L;

    W0m = lambda_temp/(L + lambda_temp) + 0j;
    W0c = lambda_temp/(L + lambda_temp)+(1-alpha**2+beta) + 0j;
    Wim = 1/(2*(L + lambda_temp));

    Wm = np.hstack((np.asarray(W0m).reshape(-1,1), Wim+np.zeros((1,2*L))));
    Wc = np.hstack((np.asarray(W0c).reshape(-1,1), Wim+np.zeros((1,2*L))));

    return Wm,Wc,L,lambda_temp

def mrdivide(A,B):
    """
    Mimic matlab mrdivide function.
    
    """
    return np.linalg.solve(B.conj().T, A.conj().T).conj().T

def wrapToPi(lam):
    """
    Wrap angle output to within -pi to pi
    
    """
    q = (lam < -np.pi) | (np.pi < lam)
    lam[q] = wrapTo2Pi(lam[q] + np.pi) - np.pi
    return lam

def wrapTo2Pi(lam):
    """
    Wrap angle output to within -2pi to 2pi
    
    """
    positiveInput = (lam > 0);
    lam = lam%(2*np.pi);
    lam[(lam == 0) & positiveInput] = 2*np.pi;
    
    return lam


def propagateState_MeeBcRom( x0_mee,t0,tf,AC,BC,SWinputs,r,nop,svs,F_U,M_U,maxAtmAlt,et0,jdate0,highFidelity,GM, Re, C_gravmodel, S_gravmodel, gravdegree, sF_gravmod):
    """
    Propagate objects and ROM density
    
    # Convert state in modified equinoctial elements to Cartesian coordinates
    # propagate states and reduced-order density and convert Cartesian states
    # back to modified equinoctial elements.
    
    Based on Matlab code by David Gondelach, Massachusetts Institute of Technology, 2020
    
    """
    x0_mee = x0_mee.reshape((nop*svs+r,-1))

    if tf == t0:
        return x0_mee

    mu = 398600.4415;

#     xx_pv = np.copy(x0_mee);
    xx_pv = np.array(x0_mee);
    for k in range (0,nop):
        for j in range (0,x0_mee.shape[1]):
            pos,vel = ep2pv_jit(x0_mee[(k)*svs:(k)*svs+6,j],mu);
            xx_pv[k*svs+0:k*svs+3,j] = pos;
            xx_pv[k*svs+3:k*svs+6,j] = vel;

    # use adams ode to solve for xf
    ode_ope = ode(computeDerivative_PosVelBcRom).set_integrator('vode',atol = 1e-10, rtol = 1e-10, method='adams')
    ode_ope.set_initial_value(xx_pv.flatten(), t0).set_f_params(AC,BC,SWinputs,r,nop,svs,F_U,M_U,maxAtmAlt,et0,jdate0,highFidelity,GM, Re, C_gravmodel, S_gravmodel, gravdegree, sF_gravmod)
    xf_out = ode_ope.integrate(tf)
    xf_pv = xf_out.reshape((nop*svs+r,-1))

    xf_mee = xf_pv;
    for k in range(0,nop):
        for j in range (0,xf_pv.shape[1]):
            pos = xf_pv[svs*(k):svs*(k)+3,j];
            vel = xf_pv[svs*(k)+3:svs*(k)+6,j];
            xf_mee[(k)*svs:(k)*svs+6,j] = pv2ep_jit(pos,vel,mu).T;

        # Make sure the difference in true longitude L of the sigma points wrt
        # the nominal true longitude L0 is minimal (i.e. L-L0 <= pi)
        # If the nominal true longitude L0 is close to pi (i.e. pi/2<L0 or L0<-pi/2) then wrap 
        # all L to [0,2pi] domain, so all difference in L <=pi (by default L is
        # on [-pi,pi] domain).
        if ((xf_mee[(k)*svs+5,0] > np.pi/2) | (xf_mee[(k)*svs+5,0] < -np.pi/2)):
            xf_mee[(k)*svs+5,:] = wrapTo2Pi(xf_mee[(k)*svs+5,:]);

    return xf_mee

def propagateState_MeeBcRom_ml( x0_mee,t0,tf,AC,BC,SWinputs,r,nop,svs,decoder_model,maxAtmAlt,et0,jdate0,highFidelity,GM,Re,C_gravmodel,S_gravmodel,gravdegree,sF_gravmod,xdata_log10_mean,xdata_log10_std):
    """
    Propagate objects and ROM density
    
    # Convert state in modified equinoctial elements to Cartesian coordinates
    # propagate states and reduced-order density and convert Cartesian states
    # back to modified equinoctial elements.
    
    Based on Matlab code by David Gondelach, Massachusetts Institute of Technology, 2020
    
    """
    x0_mee = x0_mee.reshape((nop*svs+r,-1))

    if tf == t0:
        return x0_mee

    mu = 398600.4415;

#     xx_pv = np.copy(x0_mee);
    xx_pv = np.array(x0_mee);
    for k in range (0,nop):
        for j in range (0,x0_mee.shape[1]):
            pos,vel = ep2pv_jit(x0_mee[(k)*svs:(k)*svs+6,j],mu);
            xx_pv[k*svs+0:k*svs+3,j] = pos;
            xx_pv[k*svs+3:k*svs+6,j] = vel;
    # set_trace()
    # use adams ode to solve for xf
    ode_ope = ode(computeDerivative_PosVelBcRom_ml).set_integrator('vode',atol = 1e-10, rtol = 1e-10, method='adams')
    ode_ope.set_initial_value(xx_pv.flatten(), t0).set_f_params(AC,BC,SWinputs,r,nop,svs,decoder_model,maxAtmAlt,et0,jdate0,highFidelity,GM, Re, C_gravmodel, S_gravmodel, gravdegree, sF_gravmod,xdata_log10_mean,xdata_log10_std)
    xf_out = ode_ope.integrate(tf)
    xf_pv = xf_out.reshape((nop*svs+r,-1))

    xf_mee = xf_pv;
    for k in range(0,nop):
        for j in range (0,xf_pv.shape[1]):
            pos = xf_pv[svs*(k):svs*(k)+3,j];
            vel = xf_pv[svs*(k)+3:svs*(k)+6,j];
            xf_mee[(k)*svs:(k)*svs+6,j] = pv2ep_jit(pos,vel,mu).T;

        # Make sure the difference in true longitude L of the sigma points wrt
        # the nominal true longitude L0 is minimal (i.e. L-L0 <= pi)
        # If the nominal true longitude L0 is close to pi (i.e. pi/2<L0 or L0<-pi/2) then wrap 
        # all L to [0,2pi] domain, so all difference in L <=pi (by default L is
        # on [-pi,pi] domain).
        if ((xf_mee[(k)*svs+5,0] > np.pi/2) | (xf_mee[(k)*svs+5,0] < -np.pi/2)):
            xf_mee[(k)*svs+5,:] = wrapTo2Pi(xf_mee[(k)*svs+5,:]);

    return xf_mee

def fullmee2mee(Xp,nop,svs):
    """
    Return only objects states without BCs or ROM state
    
    """
    
    mee = np.zeros((nop*6,Xp.shape[1]));
    for k in range (0,nop):
        mee[6*(k):6*(k+1),:] = Xp[svs*(k):svs*(k)+6,:];
    
    return mee

@jit(nopython=True)
def cholupdate_jit(R,x,sign):
    """
    Cholesky update equation
    
    """
    p = x.size
    x = x.T
    for k in range(p):
        if sign == '+':
            r = np.sqrt(R[k,k]**2 + x[k]**2)
        elif sign == '-':
            r = np.sqrt(R[k,k]**2 - x[k]**2)
        c = r/R[k,k]
        s = x[k]/R[k,k]
        R[k,k] = r
        if sign == '+':
            R[k,k+1:p] = (R[k,k+1:p] + s*x[k+1:p])/c
        elif sign == '-':
            R[k,k+1:p] = (R[k,k+1:p] - s*x[k+1:p])/c
        x[k+1:p]= c*x[k+1:p] - s*R[k, k+1:p]
    return R

@contextmanager
def poolcontext(*args, **kwargs):
    """
    Define pool for parallel processing
    
    """
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def UKF(X_est,Meas,time,P,RM,Q,nop,svs,r, AC, BC, SWinputs, F_U, M_U, maxAtmAlt, et0, jd0, highFidelity, GM, Re, C_gravmodel,S_gravmodel, gravdegree, sF_gravmod):
    """
    Unscented Kalman Filter to propagate space object and predicted thermospheric mass density
    
    Based on Matlab code by David Gondelach, Massachusetts Institute of Technology, 2020
    
    """

    # Unscented Filter Parameter
    # Compute the Sigma Points
    Wm,Wc,L,lam = Unscented_Transform(X_est)

    SR_Wc = np.sqrt(Wc); 
    SR_Wm = np.sqrt(Wm);

    S= np.linalg.cholesky(P).T;
    SR_R = np.sqrt(RM); # measurement noise
    SR_Q = np.sqrt(Q); # process noise
    eta = np.sqrt(L+lam);

    useMEE = True;

    m = Meas.shape[1];
    
    # Preallocate variables
    X_est_temp = np.zeros((nop*svs+r,m-1+1))
    HH = np.zeros((Meas.shape[0],nop*svs+r,m-1+1))
    Pv = np.zeros((nop*svs+r,m-1+1))

    X_est_temp[:,0:1] = X_est
    
    mu = 398600.4415
    
    # Precompile jit
    EP_temp = np.ones((6,))
    rr_temp, vv_temp = ep2pv_jit(EP_temp, mu)
    EP_temp = pv2ep_jit(rr_temp, vv_temp, mu)
    
    for i in range (0,m-1): # (0,m-1):
        print(i, m-1)
        sigv = np.real(np.hstack((eta*S, -eta*S)));
        xx = np.hstack((X_est_temp[:,i].reshape(-1,1), (sigv+np.kron(X_est_temp[:,i].reshape(-1,1),np.ones((1,2*L))))))

        # Time Update
        with poolcontext(processes=multiprocessing.cpu_count()-1 or 1) as pool:
            Xp = pool.map(partial(propagateState_MeeBcRom, t0=time[i], tf=time[i+1], AC=AC, BC=BC, SWinputs=SWinputs, r=r, nop=nop, svs=svs, F_U=F_U, M_U=M_U,maxAtmAlt=maxAtmAlt,et0=et0,jdate0=jd0,highFidelity=highFidelity,GM=GM, Re=Re, C_gravmodel=C_gravmodel, S_gravmodel=S_gravmodel, gravdegree=gravdegree, sF_gravmod=sF_gravmod), xx.T)
        Xp = np.squeeze(np.asarray(Xp).T)


        X_est_temp[:,i+1] = np.real(Wm[0,0] * Xp[:,0] + Wm[0,1] * np.sum(Xp[:,1:],axis=1));

        # Get Propagated Square Root
        _,S_minus = np.linalg.qr(np.hstack(((SR_Wc[0,1]*(Xp[:,1:]-np.kron(X_est_temp[:,i+1:i+2],np.ones((1,2*L))))), SR_Q)).T)
        S_minus = cholupdate_jit(np.real(S_minus),np.real(Wc[0,0]*(Xp[:,0]-X_est_temp[:,i+1])),'+').T

        # Measurement function
        Ym = fullmee2mee(Xp,nop,svs); # [nofMeas x nofSigma]
        ym = np.real(Wm[0,0] * Ym[:,0] + Wm[0,1] * np.sum(Ym[:,1:],axis = 1)); # [nofMeas x 1]

        DY = Ym[:,0]-ym; # [nofMeas x 1]
        DY2 = Ym[:,1:]-np.kron(ym.reshape(-1,1),np.ones((1,2*L)));

        if useMEE:
            DY[5::6] = wrapToPi(DY[5::6]); # Wrap difference in true longitude to [-pi,pi]
            DY2[5::6] = wrapToPi(DY2[5::6]); # Wrap difference in true longitude to [-pi,pi]

        # Measurement Update
        _,S_y = np.linalg.qr(np.hstack(((SR_Wc[0,1]*DY2), SR_R)).T)
        S_y = cholupdate_jit(np.real(S_y),np.real(Wc[0,0]*DY),'+').T

        # Calculate Pxy
        Pxy0 = np.real(Wc[0,0]*np.outer(Xp[:,0]-X_est_temp[:,i+1],DY.T)); # [nofStates x nofMeas]
        Pyymat = DY2;
        Pmat = Xp[:,1:]-np.kron(X_est_temp[:,i+1:i+2],np.ones((1,2*L))); # [nofStates x nofSigma-1]
        Pxy = Pxy0+Wc[0,1]*(Pmat@Pyymat.T); # [nofStates x nofMeas]

        # Measurement residual
        yres = Meas[:,i+1]-ym; # [nofMeas x 1]
        if useMEE:
            yres[5::6] = wrapToPi(yres[5::6]); # Wrap difference in true longitude to [-pi,pi]

        # Gain and Update
        KG = mrdivide(np.real(mrdivide(Pxy,S_y.T)),S_y); # [nofStates x nofMeas]
        X_est_temp[:,i+1] = X_est_temp[:,i+1] + KG@yres;
        U = KG@S_y;
#         S = np.copy(S_minus); # [nofStates x nofStates]
        S = np.array(S_minus); # [nofStates x nofStates]
        for j in range(0,len(ym)):
            S = cholupdate_jit(S.T,U[:,j],'-').T;

        HH[:,:,i+1] = (np.linalg.pinv(S@S.T)@KG@RM).T;
        Pv[:,i+1] = np.diag(S@S.T).T;
        
        if i % 3 == 0: 
            np.savez('est_variable_temp_pod_AMOS_21Obj_22_2_1.npz',X_est_temp=X_est_temp,Pv=Pv)

    return X_est_temp,Pv

def UKF_ml(X_est,Meas,time,P,RM,Q,nop,svs,r, AC, BC, SWinputs, decoder_model, maxAtmAlt, et0, jd0, highFidelity, GM, Re, C_gravmodel,S_gravmodel, gravdegree, sF_gravmod,xdata_log10_mean,xdata_log10_std):
    """
    Unscented Kalman Filter to propagate space object and predicted thermospheric mass density
    
    Based on Matlab code by David Gondelach, Massachusetts Institute of Technology, 2020
    
    """

    # Unscented Filter Parameter
    # Compute the Sigma Points
    Wm,Wc,L,lam = Unscented_Transform(X_est)

    SR_Wc = np.sqrt(Wc); 
    SR_Wm = np.sqrt(Wm);

    S= np.linalg.cholesky(P).T;
    SR_R = np.sqrt(RM); # measurement noise
    SR_Q = np.sqrt(Q); # process noise
    eta = np.sqrt(L+lam);

    useMEE = True;

    m = Meas.shape[1];
    
    # Preallocate variables
    X_est_temp = np.zeros((nop*svs+r,m-1+1))
    HH = np.zeros((Meas.shape[0],nop*svs+r,m-1+1))
    Pv = np.zeros((nop*svs+r,m-1+1))

    X_est_temp[:,0:1] = X_est
    
    mu = 398600.4415
    
    # Precompile jit
    EP_temp = np.ones((6,))
    rr_temp, vv_temp = ep2pv_jit(EP_temp, mu)
    EP_temp = pv2ep_jit(rr_temp, vv_temp, mu)
    
    for i in range (0,m-1): # (0,m-1):
        print('Hour ',i, m-1)
        sigv = np.real(np.hstack((eta*S, -eta*S)));
        xx = np.hstack((X_est_temp[:,i].reshape(-1,1), (sigv+np.kron(X_est_temp[:,i].reshape(-1,1),np.ones((1,2*L))))))

        Xp= np.zeros_like(xx)    
        for ik_xp in range(xx.shape[1]):

            Xp[:,ik_xp] = propagateState_MeeBcRom_ml(xx[:,ik_xp],t0=time[i], tf=time[i+1], AC=AC, BC=BC, SWinputs=SWinputs, r=r, nop=nop, svs=svs, decoder_model=decoder_model, maxAtmAlt=maxAtmAlt,et0=et0,jdate0=jd0,highFidelity=highFidelity,GM=GM, Re=Re, C_gravmodel=C_gravmodel, S_gravmodel=S_gravmodel, gravdegree=gravdegree, sF_gravmod=sF_gravmod,xdata_log10_mean=xdata_log10_mean,xdata_log10_std=xdata_log10_std).flatten()

        X_est_temp[:,i+1] = np.real(Wm[0,0] * Xp[:,0] + Wm[0,1] * np.sum(Xp[:,1:],axis=1));

        # Get Propagated Square Root
        _,S_minus = np.linalg.qr(np.hstack(((SR_Wc[0,1]*(Xp[:,1:]-np.kron(X_est_temp[:,i+1:i+2],np.ones((1,2*L))))), SR_Q)).T)
        S_minus = cholupdate_jit(np.real(S_minus),np.real(Wc[0,0]*(Xp[:,0]-X_est_temp[:,i+1])),'+').T

        # Measurement function
        Ym = fullmee2mee(Xp,nop,svs); # [nofMeas x nofSigma]
        ym = np.real(Wm[0,0] * Ym[:,0] + Wm[0,1] * np.sum(Ym[:,1:],axis = 1)); # [nofMeas x 1]

        DY = Ym[:,0]-ym; # [nofMeas x 1]
        DY2 = Ym[:,1:]-np.kron(ym.reshape(-1,1),np.ones((1,2*L)));

        if useMEE:
            DY[5::6] = wrapToPi(DY[5::6]); # Wrap difference in true longitude to [-pi,pi]
            DY2[5::6] = wrapToPi(DY2[5::6]); # Wrap difference in true longitude to [-pi,pi]

        # Measurement Update
        _,S_y = np.linalg.qr(np.hstack(((SR_Wc[0,1]*DY2), SR_R)).T)
        S_y = cholupdate_jit(np.real(S_y),np.real(Wc[0,0]*DY),'+').T

        # Calculate Pxy
        Pxy0 = np.real(Wc[0,0]*np.outer(Xp[:,0]-X_est_temp[:,i+1],DY.T)); # [nofStates x nofMeas]
        Pyymat = DY2;
        Pmat = Xp[:,1:]-np.kron(X_est_temp[:,i+1:i+2],np.ones((1,2*L))); # [nofStates x nofSigma-1]
        Pxy = Pxy0+Wc[0,1]*(Pmat@Pyymat.T); # [nofStates x nofMeas]

        # Measurement residual
        yres = Meas[:,i+1]-ym; # [nofMeas x 1]
        if useMEE:
            yres[5::6] = wrapToPi(yres[5::6]); # Wrap difference in true longitude to [-pi,pi]

        # Gain and Update
        KG = mrdivide(np.real(mrdivide(Pxy,S_y.T)),S_y); # [nofStates x nofMeas]
        X_est_temp[:,i+1] = X_est_temp[:,i+1] + KG@yres;
        U = KG@S_y;

        S = np.array(S_minus); # [nofStates x nofStates]
        for j in range(0,len(ym)):
            S = cholupdate_jit(S.T,U[:,j],'-').T;

        HH[:,:,i+1] = (np.linalg.pinv(S@S.T)@KG@RM).T;
        Pv[:,i+1] = np.diag(S@S.T).T;

        if i % 2 == 0: #est_variable_mlv2_fullsw_AMOS_22_2_1
            np.savez('est_variable_temp_ICIAM_6obj_2013_0Q_01RM_v2.npz',X_est_temp=X_est_temp,Pv=Pv)

    return X_est_temp,Pv
