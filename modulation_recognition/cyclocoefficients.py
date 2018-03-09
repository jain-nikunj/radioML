import numpy as np
import matplotlib.pyplot as plt

nfft = 16
cyclo_averaging = 8
offsets = [0,1,2,3,4,5,6,7]
zero_threshold = 1e-3

def round_to_zero(data, zero_threshold):
    '''Set entries with absolute values below threshold to zero'''
    data_rounded = data
    for k,s in enumerate(data):
        if abs(s) < zero_threshold:
            data_rounded[k] = 0
    return data_rounded
    
def compute_cyclo_fft(data, nfft):  
    '''
    Split data into blocks of size nfft and compute fft for each block       
    Input:data: length N (say), 
          nfft: size of fft
    Output: nfft x N/nfft size matrix where jth column contains fft of jth block
    '''

    data_reshape = np.reshape(data, (-1, nfft))    
    y =  np.fft.fftshift(np.fft.fft(data_reshape, axis=1), axes=1)  
#     plt.plot(data_reshape[0,:].real)
#     plt.title('Sample Data ' + str(nfft) + ' points')
#     plt.show()
    return y.T


def compute_correlation(x,y):
    '''
    Input: x and y are arrays to be correlated
    Output: inner product of x and y divided by length of x'''
    x = np.reshape(x, [-1,])
    y = np.reshape(y, [-1,])
    lenx = x.shape[0]
    corr = 0
    for i in range(lenx):
        corr += x[i]*np.conj(y[i])        
    corr /= lenx
    return corr

def compute_coefficients(cyc_fft, alphas):
    '''
    Input:nfft X cyclo_averaging size matrix where jth column is fft of a sample
    Output:specs: nfft X num_offsets size matrix where jth column contains spectral coefficients corresponding to its alpha(offset)
           scs: nfft X num_offsets size matrix where jth column contains scaled spectral coefficients corresponding to its alpha(offset)
    The spectral coefficients close to 0 are set to exactly zero before scaling to obtain scs
    '''       

    specs = np.zeros((nfft,len(alphas)), dtype=np.complex)
    scs = np.zeros((nfft,len(alphas)),dtype=np.complex)
    for alpha in alphas:
        z = np.array(np.zeros(cyc_fft.shape), dtype=np.complex)
        denom_right = np.zeros(cyc_fft.shape, dtype=np.complex)
        denom_left = np.zeros(cyc_fft.shape,dtype=np.complex)        
        for i in range(cyc_fft.shape[1]):
            x = cyc_fft[:,i]
            x_right = np.roll(x, alpha)
            x_left = np.roll(x, -alpha)
            z[:,i] = (x_right*np.conj(x_left))/cyc_fft.shape[1]                        
            
            denom_right[:,i] =(x_right*np.conj(x_right)/cyc_fft.shape[1])
            denom_left[:,i] = (x_left*np.conj(x_left)/cyc_fft.shape[1])            
            spec = np.mean(z,axis=1)   
            denom = np.sqrt(np.abs(np.mean(denom_right, axis=1))*np.abs(np.mean(denom_left,axis=1)))
           
            specs[:,alpha] = spec
            spec = round_to_zero(spec, zero_threshold)           
            sc = spec/denom
            scs[:,alpha] = sc 
            
    return specs, scs

def cyclo_stationary(data):
    cyc_fft = compute_cyclo_fft(data, nfft)  
    specs, scs = compute_coefficients(cyc_fft,alphas = offsets)
    return specs, scs    

def convert_to_1d(data_2d):
    '''Stacks columns below each other'''
    M,N= data_2d.shape
    data_1d = np.zeros((M*N,),dtype=np.complex)
    for i in range(N):
        data_1d[i*M:(i+1)*M] = data_2d[:,i]        
    return data_1d
