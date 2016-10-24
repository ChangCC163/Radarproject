import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.signal import butter, lfilter, freqz, periodogram
import matplotlib.figure as fig
import os
import operator
import math

#room_110       15000~35000
#room_110_1     15000~35000
#room_110_2     10000~30000
#room_110_3     15000~35000
#room_110_4     10000~30000

data = []
f = open('sit_90_0.5m-4.csv', 'rb')
reader = csv.reader(f)
for row in reader:
        data.append([float(num) for num in row])
f.close()
data_n = len(data)

for c in range(1,data_n,1):
        data[0].extend(data[c])

sig = np.zeros((2,len(data[0])/2),float)
i = 0
for c in range(10000, len(data[0])-1, 2):
        sig[0][i] = data[0][c]
        sig[1][i] = data[0][c+1]
        i += 1

signal_I = [x for x in sig[0] if (x != 0.) and (x>1000) and (x<3000)]
signal_Q = [x for x in sig[1] if (x != 0.) and (x>1000) and (x<3000)]
signal_I = np.array(signal_I)
signal_Q = np.array(signal_Q)
signal_Q_new = np.array(signal_Q)
n = len(signal_Q)
point2nor = 2000
nor_n = n/point2nor

########################### Power Normalized ###############################
for c in range(0,nor_n+1,1):
        if c < nor_n:
                signal_Q_new_power = [x**2 for x in signal_Q_new[(point2nor*c):(point2nor*(c+1))]]
                signal_Q_new_power = np.asarray(signal_Q_new_power)
                signal_Q_new_sqrt = math.sqrt(signal_Q_new_power.mean())
                signal_Q_new[(point2nor*c):(point2nor*(c+1))]\
                        = [x/signal_Q_new_sqrt for x in signal_Q_new[(point2nor*c):(point2nor*(c+1))]]  # power normalized 
        else:
                signal_Q_new_power = [x**2 for x in signal_Q_new[(point2nor*c):]]
                signal_Q_new_power = np.asarray(signal_Q_new_power)
                signal_Q_new_sqrt = math.sqrt(signal_Q_new_power.mean())
                signal_Q_new[(point2nor*c):]\
                        = [x/signal_Q_new_sqrt for x in signal_Q_new[(point2nor*c):]]      # power normalized
        

signal_Q_edc = [x-1.0 for x in signal_Q_new]
############################### Autocorrelation  ###############################
def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert N.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result
#################################  LPF  ################################
def butter_lowpass(cutoff, fs, order):
        nyq = 0.5*fs
        normal_cutoff = cutoff/nyq
        b,a = butter(order, normal_cutoff, btype = 'low', analog = False)
        return b,a

def butter_lowpass_filter(data, cutoff, fs, order):
        b,a = butter_lowpass(cutoff, fs, order)
        y = lfilter(b, a, data)
        return y

#################################  HPF  ################################
def butter_highpass(cutoff, fs, order):
        nyq = 0.5*fs
        normal_cutoff = cutoff/nyq
        b,a = butter(order, normal_cutoff, btype = 'high', analog = False)
        return b,a

def butter_highpass_filter(data, cutoff, fs, order):
        b,a = butter_highpass(cutoff, fs, order)
        y = lfilter(b, a, data)
        return y
############################### Autocorrelation  ###############################
def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert N.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result
################################  Find Peak  ###################################
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

##########################  FFT  ##########################################
def FFT(data, fs, N):
    T = 1.0/fs
    yfft = fft(data)
    xf = np.linspace(0.0,1.0/(2.0*T),N/2)
    y = 2.0/N*np.abs(yfft[0:N/2])
    return xf,y
###########################   Find Period   ################################
def peak_dif(x):
        i = 0
        y = []
        for c in range(0,len(x)-1,1):
                if(x[i+1]-x[i]) > 1.5*fs and (x[i+1]-x[i]) < 5.0*fs:
                        y.extend([x[i+1]-x[i]])
                i += 1
        if (y == []):
                y = [30000]
        elif (y != []):
                y = y
                
        y = np.asarray(y)
        return y

######################### Find FFT Max ###########################
def fft_max(x, fft_hz):
        index_fft, value = max(enumerate(x), key=operator.itemgetter(1))
        hz = fft_hz[index_fft]
        fft_hz_l = list(fft_hz)
        if hz < 0.15:
                index_new = fft_hz_l.index(hz)
                index_fft, value = max(enumerate(x[index_new:]), key=operator.itemgetter(1))
                hz = fft_hz[index_fft+index_new]
        else:
                pass
                
        return hz

#############################################################################
order = 5
fs = 500.0
cutoff_low = 0.1
cutoff_high = 3.0
recur_n = 500                # points to recursive 
inital_point = 10240            # calculated points one times
diff_n = n-inital_point
recur = diff_n/recur_n          # recursive times
remainder = diff_n%recur_n
period_array = np.zeros((1,recur+1),float)
total_detect = 0
success_num = 0
numtaps = 51
nyqu = fs*0.5

h = signal.firwin(numtaps, [cutoff_low, cutoff_high], window = 'hamming', pass_zero = False, nyq = nyqu)

for c in range(0,recur+1,1):
#for c in range(0,1):
        power_var = 0
        if c == 0:
                signal_Q_r = signal_Q_edc[0:inital_point-1]
                        
               
                s_lpf = np.convolve(signal_Q_r, h, mode = 'valid')

                fft_hz,s_fft = FFT(s_lpf, fs, len(s_lpf))
                hz = fft_max(s_fft, fft_hz)
                s_period_fft = hz*60.0
        
                s_auto = estimated_autocorrelation(s_lpf)
                                
##########################################  Threshold tuning #########################################
                max_index = 0
                table = np.zeros((2,20),float)
                for j in range (0,6,1):
                        i = 0.5-(j*0.05)
                        index = detect_peaks(s_auto, mph = i, mpd = 500)
                        peak_point = peak_dif(index)
                        if len(peak_point) > max_index and peak_point[0] != 30000:
                                max_index = len(peak_point)
                                threshold_num = i
                        elif len(peak_point) == max_index and peak_point[0] != 30000:
                                pass
                        else:
                                threshold_num = 0.35
                        
                #table[0][c] = power_var
                #table[1][c] = threshold_num
######################################################################################################
                index = detect_peaks(s_auto, mph = threshold_num, mpd = 500)
                peak_point = peak_dif(index)
                s_period_point = peak_point.mean()
                s_period_auto = 60.0/(s_period_point/fs)

                
                if peak_point[0] == 30000:
                        s_period = 1
                        print "Miss Detection in 0~%d" % inital_point
                elif hz > 0.7 or hz < 0.15:
                        s_period = 2
                        print "False alarm in 0~%d" % inital_point
                elif abs(s_period_auto-s_period_fft) > 5.0:
                        s_period = 1
                        print "Miss Detection in 0~%d" % inital_point
                else:  
                        s_period = (s_period_auto+s_period_fft)/2.0
                        print s_period
                        success_num += 1
                        
                period_array[0][c] = s_period
                total_detect += 1
                
        elif c == recur:
                signal_Q_r = signal_Q_edc[n-inital_point:]

                power_var = (np.array([x**2 for x in signal_Q_r])).var()
                
                s_lpf = np.convolve(signal_Q_r, h, mode = 'valid')

                fft_hz,s_fft = FFT(s_lpf, fs, len(s_lpf))
                hz = fft_max(s_fft, fft_hz)
                s_period_fft = hz*60.0

                s_auto = estimated_autocorrelation(s_lpf)
##########################################  Threshold tuning #########################################
                max_index = 0
                for j in range (0,6,1):
                        i = 0.5-(j*0.05)
                        index = detect_peaks(s_auto, mph = i, mpd = 500)
                        peak_point = peak_dif(index)
                        if len(peak_point) > max_index and peak_point[0] != 30000:
                                max_index = len(peak_point)
                                threshold_num = i
                        elif len(peak_point) == max_index and peak_point[0] != 30000:
                                pass
                        else:
                                threshold_num = 0.35
                        
                #table[0][c] = power_var
                #table[1][c] = threshold_num
######################################################################################################
                index = detect_peaks(s_auto, mph = threshold_num, mpd = 500)
                peak_point = peak_dif(index)
                s_period_point = peak_point.mean()
                s_period_auto = 60.0/(s_period_point/fs)                
                
                if peak_point[0] == 30000:
                        s_period = 1
                        print "Miss Detection at last %d" % inital_point
                elif hz > 0.7 or hz < 0.15:
                        s_period = 2
                        print "False alarm at last %d" % inital_point
                elif abs(s_period_auto-s_period_fft) > 5.0:
                         s_period = 1
                         print "Miss Detection at last %d" % inital_point
                else: 
                        s_period = (s_period_auto+s_period_fft)/2.0
                        print s_period
                        success_num += 1
                
                period_array[0][c] = s_period
                total_detect += 1

        else:
                elemt = (inital_point)+(recur_n*c)    ## last elemt
                signal_Q_r = signal_Q_edc[recur_n*c:elemt]
                            
                s_lpf = np.convolve(signal_Q_r, h, mode = 'same')

                fft_hz,s_fft = FFT(s_lpf, fs, len(s_lpf))
                hz = fft_max(s_fft, fft_hz)
                s_period_fft = hz*60.0

                s_auto = estimated_autocorrelation(s_lpf)               
##########################################  Threshold tuning #########################################
                max_index = 0
                for j in range (0,6,1):
                        i = 0.5-(j*0.05)
                        index = detect_peaks(s_auto, mph = i, mpd = 500)
                        peak_point = peak_dif(index)
                        if len(peak_point) > max_index and peak_point[0] != 30000:
                                max_index = len(peak_point)
                                threshold_num = i
                        elif len(peak_point) == max_index and peak_point[0] != 30000:
                                pass
                        else:
                                threshold_num = 0.35
                        
                #table[0][c] = power_var
                #table[1][c] = threshold_num
######################################################################################################
                index = detect_peaks(s_auto, mph = threshold_num, mpd = 500)
                peak_point = peak_dif(index)
                s_period_point = peak_point.mean()
                s_period_auto = 60.0/(s_period_point/fs)

                
                if peak_point[0] == 30000:
                        s_period = 1
                        print "Miss Detection in %d to %d" %(elemt-inital_point-1, elemt)
                elif hz > 0.7 or hz < 0.15:
                        s_periof = 2
                        print "False alarm in %d to %d" %(elemt-inital_point-1, elemt)
                elif abs(s_period_auto-s_period_fft) > 5.0:
                        s_period = 1
                        print "Miss Detection in %d to %d" %(elemt-inital_point-1, elemt)
                else:       
                        s_period = (s_period_auto+s_period_fft)/2.0
                        print s_period
                        success_num += 1
                        
                period_array[0][c] = s_period
                total_detect += 1

period_array = [x for x in period_array[0] if x!= 1 and x!= 2]
period_array = np.asarray(period_array)
print "period mean is %f " %(period_array.mean())
print "period var is %f" %(period_array.var())
print "Success Detect rate : %f%%" %(success_num*100.0/total_detect)

plt.figure(3)
plt.plot(signal_Q_new)
###################################################################################################################################
s_lpf_all = np.convolve(signal_Q_edc, h, mode = 'valid')
s_auto_all = estimated_autocorrelation(s_lpf_all)
nn = len(s_lpf_all)
hz, db = FFT(s_lpf_all, fs, nn)
plt.figure(2)
plt.plot(hz, db)
plt.figure(4)
plt.plot(s_auto_all)
plt.figure(1)
plt.plot(s_lpf_all)
plt.show()

