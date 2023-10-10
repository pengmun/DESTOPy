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

#     ## Space weather inputs (Non-linaer)
#     U1 = np.zeros((36,X1.shape[1]))
#     U1[:14,:] = TA['SWdataFull'][:-1,:].T

#     # Include future inputs (F10, S10, M10, Y10, DSTDTC) 
#     U1[14,:] = TA['SWdataFull'][1:,2].T; # F10
#     U1[15,:] = TA['SWdataFull'][1:,4].T; # S10
#     U1[16,:] = TA['SWdataFull'][1:,6].T; # XM10
#     U1[17,:] = TA['SWdataFull'][1:,10].T; # DSTDTC

#     # and non-linear inputs (F10**2, DSTDTC**2) for current and future
#     U1[18,:] = np.multiply(TA['SWdataFull'][:-1,2].T,TA['SWdataFull'][:-1,2].T)
#     U1[19,:] = np.multiply(TA['SWdataFull'][1:,2].T,TA['SWdataFull'][1:,2].T)
#     U1[20,:] = np.multiply(TA['SWdataFull'][:-1,4].T,TA['SWdataFull'][:-1,4].T)
#     U1[21,:] = np.multiply(TA['SWdataFull'][1:,4].T,TA['SWdataFull'][1:,4].T)
#     U1[22,:] = np.multiply(TA['SWdataFull'][:-1,6].T,TA['SWdataFull'][:-1,6].T)
#     U1[23,:] = np.multiply(TA['SWdataFull'][1:,6].T,TA['SWdataFull'][1:,6].T)
    
#     U1[24,:] = np.multiply(TA['SWdataFull'][:-1,10].T,TA['SWdataFull'][:-1,10].T)
#     U1[25,:] = np.multiply(TA['SWdataFull'][1:,10].T,TA['SWdataFull'][1:,10].T)
    
#     # and non-linear cross terms (F10*M10, DSTDTC*M10) for current and future
#     U1[26,:] = np.multiply(TA['SWdataFull'][:-1,2].T,TA['SWdataFull'][:-1,4].T)
#     U1[27,:] = np.multiply(TA['SWdataFull'][1:,2].T,TA['SWdataFull'][1:,4].T)
        
#     U1[28,:] = np.multiply(TA['SWdataFull'][:-1,4].T,TA['SWdataFull'][:-1,6].T)
#     U1[29,:] = np.multiply(TA['SWdataFull'][1:,4].T,TA['SWdataFull'][1:,6].T)
    
#     U1[30,:] = np.multiply(TA['SWdataFull'][:-1,4].T,TA['SWdataFull'][:-1,10].T)
#     U1[31,:] = np.multiply(TA['SWdataFull'][1:,4].T,TA['SWdataFull'][1:,10].T)
    
#     U1[32,:] = np.multiply(TA['SWdataFull'][:-1,6].T,TA['SWdataFull'][:-1,10].T)
#     U1[33,:] = np.multiply(TA['SWdataFull'][1:,6].T,TA['SWdataFull'][1:,10].T)
    
#     U1[34,:] = np.multiply(TA['SWdataFull'][:-1,8].T,TA['SWdataFull'][:-1,10].T)
#     U1[35,:] = np.multiply(TA['SWdataFull'][1:,8].T,TA['SWdataFull'][1:,10].T) 
    
#     q = 36;

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

def generateMLROM_JB2008(TA,r):
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    import tensorflow as tf
    
    tf.compat.v1.disable_eager_execution()
#     tf.config.threading.set_intra_op_parallelism_threads(1)
#     tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.get_logger().setLevel('ERROR')
    
    # generateROM_JB2008 - Compute reduced-order dynamic density model based on 
    # JB2008 density data

    ## Reduced Order Data
    ## Uh and UH.T
    model_path = 'Model/JB2008_CNN_v2'

    model = keras.models.load_model(model_path)
    encoder_model = Model(inputs=model.input, outputs=model.layers[9].output)
    encoder_model.summary()

    decode_input = Input(model.layers[10].input_shape[1:])
    decoder_model = decode_input
    for layer in model.layers[10:]:
        decoder_model = layer(decoder_model)
    decoder_model = Model(inputs=decode_input, outputs=decoder_model)
    decoder_model.summary()

    densityDataLogVarMLROM = np.load('Model/densityDataLogVarMLROM.npy')

    ## ROM timesnapshot
    X1 = densityDataLogVarMLROM[0:r,0:-1];
    X2 = densityDataLogVarMLROM[0:r,1:];

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

    return PhiC, encoder_model, decoder_model, Qrom

def generateMLROM_JB2008_fullsw(TA,r):
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    import tensorflow as tf
    
    tf.compat.v1.disable_eager_execution()
#     tf.config.threading.set_intra_op_parallelism_threads(1)
#     tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.get_logger().setLevel('ERROR')
    
    # generateROM_JB2008 - Compute reduced-order dynamic density model based on 
    # JB2008 density data

    ## Reduced Order Data
    ## Uh and UH.T
    model_path = 'Model/JB2008_CNN_v2'

    model = keras.models.load_model(model_path)
    encoder_model = Model(inputs=model.input, outputs=model.layers[9].output)
    encoder_model.summary()

    decode_input = Input(model.layers[10].input_shape[1:])
    decoder_model = decode_input
    for layer in model.layers[10:]:
        decoder_model = layer(decoder_model)
    decoder_model = Model(inputs=decode_input, outputs=decoder_model)
    decoder_model.summary()

    densityDataLogVarMLROM = np.load('Model/densityDataLogVarMLROM.npy')

    ## ROM timesnapshot
    X1 = densityDataLogVarMLROM[0:r,0:-1];
    X2 = densityDataLogVarMLROM[0:r,1:];

    ## Space weather inputs 
    U1 = np.zeros((36,X1.shape[1]))
    U1[:14,:] = TA['SWdataFull'][:-1,:].T

    # Include future inputs (F10, S10, M10, Y10, DSTDTC) 
    U1[14,:] = TA['SWdataFull'][1:,2].T; # F10
    U1[15,:] = TA['SWdataFull'][1:,4].T; # S10
    U1[16,:] = TA['SWdataFull'][1:,6].T; # XM10
    U1[17,:] = TA['SWdataFull'][1:,10].T; # DSTDTC

    # and non-linear inputs (F10**2, DSTDTC**2) for current and future
    U1[18,:] = np.multiply(TA['SWdataFull'][:-1,2].T,TA['SWdataFull'][:-1,2].T)
    U1[19,:] = np.multiply(TA['SWdataFull'][1:,2].T,TA['SWdataFull'][1:,2].T)
    U1[20,:] = np.multiply(TA['SWdataFull'][:-1,4].T,TA['SWdataFull'][:-1,4].T)
    U1[21,:] = np.multiply(TA['SWdataFull'][1:,4].T,TA['SWdataFull'][1:,4].T)
    U1[22,:] = np.multiply(TA['SWdataFull'][:-1,6].T,TA['SWdataFull'][:-1,6].T)
    U1[23,:] = np.multiply(TA['SWdataFull'][1:,6].T,TA['SWdataFull'][1:,6].T)
    
    U1[24,:] = np.multiply(TA['SWdataFull'][:-1,10].T,TA['SWdataFull'][:-1,10].T)
    U1[25,:] = np.multiply(TA['SWdataFull'][1:,10].T,TA['SWdataFull'][1:,10].T)
    
    # and non-linear cross terms (F10*M10, DSTDTC*M10) for current and future
    U1[26,:] = np.multiply(TA['SWdataFull'][:-1,2].T,TA['SWdataFull'][:-1,4].T)
    U1[27,:] = np.multiply(TA['SWdataFull'][1:,2].T,TA['SWdataFull'][1:,4].T)
        
    U1[28,:] = np.multiply(TA['SWdataFull'][:-1,4].T,TA['SWdataFull'][:-1,6].T)
    U1[29,:] = np.multiply(TA['SWdataFull'][1:,4].T,TA['SWdataFull'][1:,6].T)
    
    U1[30,:] = np.multiply(TA['SWdataFull'][:-1,4].T,TA['SWdataFull'][:-1,10].T)
    U1[31,:] = np.multiply(TA['SWdataFull'][1:,4].T,TA['SWdataFull'][1:,10].T)
    
    U1[32,:] = np.multiply(TA['SWdataFull'][:-1,6].T,TA['SWdataFull'][:-1,10].T)
    U1[33,:] = np.multiply(TA['SWdataFull'][1:,6].T,TA['SWdataFull'][1:,10].T)
    
    U1[34,:] = np.multiply(TA['SWdataFull'][:-1,8].T,TA['SWdataFull'][:-1,10].T)
    U1[35,:] = np.multiply(TA['SWdataFull'][1:,8].T,TA['SWdataFull'][1:,10].T) 
    
    q = 36;

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

    return PhiC, encoder_model, decoder_model, Qrom

def generateMLROM_JB2008_v5(TA,r):
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    import tensorflow as tf
    
    tf.compat.v1.disable_eager_execution()
#     tf.config.threading.set_intra_op_parallelism_threads(1)
#     tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.get_logger().setLevel('ERROR')
    
    # generateROM_JB2008 - Compute reduced-order dynamic density model based on 
    # JB2008 density data

    ## Reduced Order Data
    ## Uh and UH.T
    model_path = 'Model/JB2008_CNN_v5'

    model = keras.models.load_model(model_path)
    encoder_model = Model(inputs=model.input, outputs=model.layers[7].output)
    encoder_model.summary()

    decode_input = Input(model.layers[8].input_shape[1:])
    decoder_model = decode_input
    for layer in model.layers[8:]:
        decoder_model = layer(decoder_model)
    decoder_model = Model(inputs=decode_input, outputs=decoder_model)
    decoder_model.summary()

    densityDataLogVarMLROM = np.load('Model/densityDataLogVarMLROMv5.npy')

    ## ROM timesnapshot
    X1 = densityDataLogVarMLROM[0:r,0:-1];
    X2 = densityDataLogVarMLROM[0:r,1:];

#     ## Space weather inputs: [doy; UThrs; F10; F10B; S10; S10B; XM10; XM10B; Y10; Y10B; DSTDTC; GWRAS; SUN(1); SUN(2)]
#     U1 = np.zeros((23,X1.shape[1]))
#     U1[:14,:] = TA['SWdataFull'][:-1,:].T;
#     # Add future values
#     U1[14,:] = TA['SWdataFull'][1:,10].T; # DSTDTC
#     U1[15,:] = TA['SWdataFull'][1:,2].T; # F10
#     U1[16,:] = TA['SWdataFull'][1:,4].T; # S10
#     U1[17,:] = TA['SWdataFull'][1:,6].T; # XM10
#     U1[18,:] = TA['SWdataFull'][1:,8].T; # Y10
#     # Add quadratic DSTDTC
#     U1[19,:] = (TA['SWdataFull'][:-1,10]**2).T; # DSTDTC^2
#     U1[20,:] = (TA['SWdataFull'][1:,10]**2).T; # DSTDTC^2
#     ## Add mixed terms
#     U1[21,:] = TA['SWdataFull'][:-1,10].T * TA['SWdataFull'][:-1,2].T;
#     U1[22,:] = TA['SWdataFull'][1:,10].T * TA['SWdataFull'][1:,2].T;

#     q = 23;

    ## Space weather inputs 
    U1 = np.zeros((36,X1.shape[1]))
    U1[:14,:] = TA['SWdataFull'][:-1,:].T

    # Include future inputs (F10, S10, M10, Y10, DSTDTC) 
    U1[14,:] = TA['SWdataFull'][1:,2].T; # F10
    U1[15,:] = TA['SWdataFull'][1:,4].T; # S10
    U1[16,:] = TA['SWdataFull'][1:,6].T; # XM10
    U1[17,:] = TA['SWdataFull'][1:,10].T; # DSTDTC

    # and non-linear inputs (F10**2, DSTDTC**2) for current and future
    U1[18,:] = np.multiply(TA['SWdataFull'][:-1,2].T,TA['SWdataFull'][:-1,2].T)
    U1[19,:] = np.multiply(TA['SWdataFull'][1:,2].T,TA['SWdataFull'][1:,2].T)
    U1[20,:] = np.multiply(TA['SWdataFull'][:-1,4].T,TA['SWdataFull'][:-1,4].T)
    U1[21,:] = np.multiply(TA['SWdataFull'][1:,4].T,TA['SWdataFull'][1:,4].T)
    U1[22,:] = np.multiply(TA['SWdataFull'][:-1,6].T,TA['SWdataFull'][:-1,6].T)
    U1[23,:] = np.multiply(TA['SWdataFull'][1:,6].T,TA['SWdataFull'][1:,6].T)
    
    U1[24,:] = np.multiply(TA['SWdataFull'][:-1,10].T,TA['SWdataFull'][:-1,10].T)
    U1[25,:] = np.multiply(TA['SWdataFull'][1:,10].T,TA['SWdataFull'][1:,10].T)
    
    # and non-linear cross terms (F10*M10, DSTDTC*M10) for current and future
    U1[26,:] = np.multiply(TA['SWdataFull'][:-1,2].T,TA['SWdataFull'][:-1,4].T)
    U1[27,:] = np.multiply(TA['SWdataFull'][1:,2].T,TA['SWdataFull'][1:,4].T)
        
    U1[28,:] = np.multiply(TA['SWdataFull'][:-1,4].T,TA['SWdataFull'][:-1,6].T)
    U1[29,:] = np.multiply(TA['SWdataFull'][1:,4].T,TA['SWdataFull'][1:,6].T)
    
    U1[30,:] = np.multiply(TA['SWdataFull'][:-1,4].T,TA['SWdataFull'][:-1,10].T)
    U1[31,:] = np.multiply(TA['SWdataFull'][1:,4].T,TA['SWdataFull'][1:,10].T)
    
    U1[32,:] = np.multiply(TA['SWdataFull'][:-1,6].T,TA['SWdataFull'][:-1,10].T)
    U1[33,:] = np.multiply(TA['SWdataFull'][1:,6].T,TA['SWdataFull'][1:,10].T)
    
    U1[34,:] = np.multiply(TA['SWdataFull'][:-1,8].T,TA['SWdataFull'][:-1,10].T)
    U1[35,:] = np.multiply(TA['SWdataFull'][1:,8].T,TA['SWdataFull'][1:,10].T) 
    
    q = 36;

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
    Qrom = np.cov(errPred)/10; # Covariance of error

    return PhiC, encoder_model, decoder_model, Qrom

def generateMLROM_JB2008_simple(TA,r):
    densityDataLogVarMLROM = np.load('Model/densityDataLogVarMLROM.npy')

    ## ROM timesnapshot
    X1 = densityDataLogVarMLROM[0:r,0:-1];
    X2 = densityDataLogVarMLROM[0:r,1:];

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

    return PhiC, Qrom

