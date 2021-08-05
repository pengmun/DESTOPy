import numpy as np
from scipy.linalg import logm

def generateROM_JB2008(TA,r):
    # generateROM_JB2008 - Compute reduced-order dynamic density model based on 
    # JB2008 density data

    ## Reduced Order Data
    Uh = TA['U100'][:,0:r];

    ## ROM timesnapshot
    X1 = TA['densityDataLogVarROM100'][0:r,0:-1];
    X2 = TA['densityDataLogVarROM100'][0:r,1:];

    ## Space weather inputs: [doy; UThrs; F10; F10B; S10; S10B; XM10; XM10B; Y10; Y10B; DSTDTC; GWRAS; SUN(1); SUN(2)]
    U1 = np.zeros((23,X1.shape[1]))
    U1[:14,:] = TA['SWdataFull'][:-1,:].T;
    # Add future values
    U1[14,:] = TA['SWdataFull'][1:,10].T; # DSTDTC
    U1[15,:] = TA['SWdataFull'][1:,2].T; # F10
    U1[16,:] = TA['SWdataFull'][1:,4].T; # S10
    U1[17,:] = TA['SWdataFull'][1:,6].T; # XM10
    U1[18,:] = TA['SWdataFull'][1:,8].T; # Y10
    # Add quadratic DSTDTC
    U1[19,:] = (TA['SWdataFull'][:-1,10]**2).T; # DSTDTC^2
    U1[20,:] = (TA['SWdataFull'][1:,10]**2).T; # DSTDTC^2
    ## Add mixed terms
    U1[21,:] = TA['SWdataFull'][:-1,10].T * TA['SWdataFull'][:-1,2].T;
    U1[22,:] = TA['SWdataFull'][1:,10].T * TA['SWdataFull'][1:,2].T;

    q = 23;

    ## DMDc

    # X2 = A*X1 + B*U1 = [A B]*[X1;U1] = Phi*Om
    Om = np.concatenate((X1,U1))

    # Phi = X2*pinv(Om)
    Phi = X2@np.linalg.pinv(Om)

    # Discrete-time dynamic and input matrix
    A = Phi[:r,:r];
    B = Phi[:r,r:];

    dth = 1;    #discrete time dt of the ROM in hours
    # Phi = [[A, B];[np.zeros((q,r)), np.eye(q)]];
    Phi = np.vstack((np.hstack((A,B)),np.hstack((np.zeros((q,r)),np.eye(q)))))
    PhiC = logm(Phi)/dth;

    ## Covariance
    X2Pred = A@X1 + B@U1; # Predict ROM state for 1hr
    errPred = X2Pred-X2; # Error of prediction w.r.t. training data
    Qrom = np.cov(errPred); # Covariance of error

    return PhiC, Uh, Qrom