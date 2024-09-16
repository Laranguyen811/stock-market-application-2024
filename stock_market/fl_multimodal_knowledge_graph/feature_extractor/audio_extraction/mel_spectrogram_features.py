import numpy as np
import matplotlib.pyplot as plt
import librosa
from stock_market.fl_multimodal_knowledge_graph.data_collection.data_collection_tools.download_sounds import download_sound
import os
# This module convert audio wave to mel spectrogram features for better signals and audio processing
# Pre-emphasis is a technique used in audio preprocessing to amplify higher frequencies of an audio signal, especially useful to enhance signal-to-noise ratio
# The pre-emphasis filter is typically a 1st-order high-pass filter, with the most common form of y[n] = x[n] - alpha . x[n-1]
# Where:
# y[n]: the output signal
# x[n]: the input signal
# alpha: pre-emphasis coefficient (often between 0.9 and 1.0), determining the pre-emphasis on the higher frequencies

def pre_emphasis(y, sr):
    ''' Takes the audio signal and the sample rate amd returns the pre_emphasised audio signal.
    Inputs:
        y(np.array): A 1-D numpy array of audio signal.
        sr(int): An integer of the sample rate.
    Returns:
        np.array: A 1-D numpy array of pre-emphasised audio signal
    '''
    pre_emphasis_coeff = 0.97  # Defining the pre-emphasis coefficient

    y_preemphasised = np.append(y[0],y[1:] - pre_emphasis_coeff * y[:-1])

    #Plotting the original and pre-emphasised signals
    plt.figure(figsize=(12,6))
    plt.plot(y)
    plt.subplots(2,1,1)
    plt.title("Original Sound")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.plot(y_preemphasised)
    plt.subplots(2,1,2)
    plt.title("Pre-emphasised Signal")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    plt.close(all)

if __name__ == "__main__":
    # Example usage
    api_key = 'API_KEY'
    search_query = 'Stock market'
    save_path = './downloads'
    # Downloading the sound
    downloaded_files = download_sound(search_query,save_path)

    # Loading the first downloaded file:
    if downloaded_files:
        file_path = os.path.join(save_path,downloaded_files[0])
        y, sr = librosa.load(file_path)
        pre_emphasis(y,sr)
    else:
        print("No files downloaded.")

def framing(data: np.array,window_length: int,hop_length: int,pad_value=0) -> np.array:
    ''' Takes data from a continuous window signal and divides into small overlapping segments called frames
    Inputs:
        data (np.ndarray):np.array of dimension N >= 1.
        window_length (integer): an integer of a number of samples in each frame.
        hop_length (integer): an integer of an advance (in samples) between each window.
    Returns:
        np.array: np.array of dimension (N+1)-D with as many windows as there are complete frames that can be extracted.
    '''
    num_samples = data.shape[0]  # Assign the number of samples as the number of columns in data

    # Calculating the number of frame, including padding if needed
    num_frames = 1 + int(np.ceil((num_samples-window_length)/hop_length))  # Calculating the number of potential overlapping frames divided by the number of hops of the remaining frames. Then, calculating the ceiling of the number (the smallest integer that is greater or equal to that number) to ensure that any frame is a full frame and add 1 to account for the initial frame

    # Padding the data if necessary
    pad_length = max(0, num_frames * hop_length + window_length - num_samples)  # Calculating the amount of padding needed to ensure that all frames have the same length. Doing so by calculating the number of samples covered by frames, then adding the window length to give the total number of samples needed to include in the last frame and finally subtracting the actual number of frames to see the difference. Ensuring that the padding length is non-negative using max function.
    pad_data = np.pad(data,((0,pad_length),) + ((0,0),) * (data.ndim - 1), mode='constant', constant_values= pad_value)  # Padding any array with a specific value for each dimension with pads of a constant value (using pad_value)

    # Creating the frames using stride_tricks
    shape = (num_frames,window_length) + data.shape[1:]  # Assigning the shape to number of frames and the window length added to the shape of data from the second sample onwards
    strides = (pad_data.strides[0] * hop_length,) + pad_data.strides  # Calculating the stride (number of bytes needed to step to get to the next frame) by multiplying the stride of the first dimension with the hop length to ensure that the stride will account for hop length in the preceding one and adding the original stride on top
    frames = np.lib.stride_tricks.as_strided(pad_data,shape=shape,strides=strides)  # Creating a view of an array with a specific shape and strides, using side_tricks module to manipulate the stride of an array ( more efficient coding) based on input padded data, the desired shape and the strides specified

    return frames
def symmetric_hann_window(window_length):
    ''' Takes window length and returns a symmetric Hanning window for better Fourier analysis, smoothing and spectral leakage prevention.
    Inputs:
        window_length(integer): A string of window length in each frame
    Returns:
        array: An array of a symmetric Hann window
    '''
    return np.hanning(window_length)

def stft_magnitude(signal: np.array, fft_length: int, hop_length: int = None,window_length: int = None ) -> np.array:
    ''' Takes signal, the number of points used in the Fast Fourier Transform (FFT) computation, hop length and window length and returns an array of STFT (Short-Time Fourier Transform) magnitude.
    Inputs:
        signal (np.array): A 1D numpy array of the input time-domain signal
        fft_length (int): An integer specifying the length of the FFT window
        hop_length (int): An optional integer specifying the length of the hop
        window_length (int): An optional integer specifying the the length of the window
    Returns:
        np.array: An array of STFT (Short-Time Fourier) magnitude
    '''
    if hop_length is None:  # If the hop length is not provided
        hop_length = window_length // 2  # Setting the hop length to the window length divided by 2
    if window_length is None:  # If the window length is not provided
        window_length = fft_length  # Setting the window length to the number of points of FFT computation

    # Framing the signal
    frames = framing(signal,window_length,hop_length)

    window = symmetric_hann_window(window_length)  # Creating a symmetric Hann window from window length
    windowed_frames = frames * window  # Creating an array of frames by multiplying an array of frames with an array of a symmetric Hann window

    # Computing the FFT magnitude
    return np.abs(np.fft.rfft(windowed_frames,int(fft_length)))  # Computing the fft magnitude of each frame by computing the one-dimensional n-point discrete Fourier Transform (DFT) of a real-valued array, then finding the absolute value of it to calculate the magnitude of the resulting complex-valued FFT coefficients

if __name__ == '__main__':
    data = np.random.randn(100)
    fft_length = 256
    hop_length = 50
    window_length = 100
    magnitude = stft_magnitude(data,fft_length,hop_length,window_length)
    print(magnitude.shape)

    # Plotting the symmetric Hann window
    symmetric_hann_window = symmetric_hann_window(window_length)
    plt.plot(symmetric_hann_window)
    plt.title("Symmetric Hann Window")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def hertz_to_mel(frequencies_hertz:np.ndarray) -> np.ndarray:
    ''' Takes hertz frequencies and converts them to mel scale using HTK (Hidden Markov Model Toolkit, a software designed to build and manipulate Hidden Markov Model) formula, for better perceptual alignment, feature extraction, improved model performance and data efficiency.
    Inputs:
        frequencies_hertz(np.ndarray): np.ndarray of frequencies in hertz.
    Returns:
        np.ndarray: np.ndarray of the same size as frequencies_hertz containing corresponding values on the mel scale
    '''

    # Constants
    _MEL_HIGH_FREQUENCY_Q = 2595.0
    _MEL_BREAK_FREQUENCY_HERTZ = 700.0

    # Parameter Validation
    if np.any(frequencies_hertz < 0):
        raise ValueError("Frequencies must be non-negative")

    # Conversion
    return _MEL_HIGH_FREQUENCY_Q * np.log(1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))

def spectrogram_to_mel_matrix(num_mel_bins:int = 20, num_spectrogram_bins:int = 129, audio_sample_rate:int = 8000, lower_edge_hertz: float = 125.0,upper_edge_hertz: float = 3800.0) -> np.ndarray:
    ''' Takes the number of Mel bands, the number of spectrogram bins, the sample rate of audio input, the lower bound of the frequencies to be included in the Mel spectrum and the upper bound of the frequencies to be included in the Mel spectrum and returns a numpy ndarray of the Mel spectrogram.
    Inputs:
        num_mel_bins(int): An integer specifying the number of Mel bands.
        num_spectrogram_bins(int): An integer specifying the number of spectrogram bins.
        audio_sample_rate(int): An integer specifying the sample rate of audio input.
        lower_edge_hertz(float): A float specifying the lower bound of the frequencies to be included in the Mel.
        upper_edge_hertz(float): A float specifying the upper bound of the frequencies to be included in the Mel.
    Returns:
        np.ndarray: A numpy ndarray of the Mel spectrogram
    Raises:
        ValueError: if frequency edges are incorrectly ordered or out of range.
    '''

    # Constants
    nyquist_hertz = audio_sample_rate / 2.0  # Calculating the Nyquist frequency(the highest frequency that can be accurately represented in a digital system without introducing aliasing (distortion)), which is half the sample rate.

    # Validation to check if the frequency edges are correctly ordered and within range

    if lower_edge_hertz < 0.0:
        raise ValueError(f"Lower edge Hertz {lower_edge_hertz} must be greater than or equal to zero.")
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError(f"Lower Edge Hertz {lower_edge_hertz} >= Upper Edge Hertz {upper_edge_hertz}")
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError(f"Upper Edge Hertz {upper_edge_hertz} is greater than Nyquist {nyquist_hertz}")

    # Frequency Conversion

    spectrogram_bins_hertz = np.linspace(0.0,nyquist_hertz,num_spectrogram_bins)  # Creating an array of an evenly spaced numbers over a specified interval (frequency values) corresponding to the bins of the spectrogram to map the spectrogram bins to Mel scale.
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)  # Converting from hertz to Mel based on the array of frequency values

    # Calculating band edges
    band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),hertz_to_mel(upper_edge_hertz),num_mel_bins + 2)  # Creating an array of an evenly spaced numbers over a specified interval by converting the lower edge hertz to Mel and the upper edge hertz to Mel with the number of mel bins adding two

    # Initializing mel weights matrix
    mel_weights_matrix = np.zeros((num_spectrogram_bins,num_mel_bins))  # Creating an numpy array of zeros with the shape of the number of spectrogram bins and the number of Mel bins

    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i +3]  # Assigning lower edge mel, center mel and upper edge mel with a band edge each, with a distance of 3 Mel bins away from each other
        lower_slope = (spectrogram_bins_mel - lower_edge_mel) / (center_mel - lower_edge_mel)  # Calculating the lower slope
        upper_slope = (upper_edge_mel - spectrogram_bins_mel) / (upper_edge_mel - center_mel)  # Calculating the upper slope
        mel_weights_matrix[:,i] = np.maximum(0.0,np.minimum(lower_slope,upper_slope))  # Calculating the Mel weights for the ith Mel bin in the Mel weights matrix by finding the minimum of the lower slope and the upper slope, ensuring all numbers are non-negative

    # Ensuring that DC bin always gets a zero coefficient
    mel_weights_matrix[0, :] = 0.0

    return mel_weights_matrix

def log_mel_spectrogram(data: np.ndarray, audio_sample_rate: int = 8000, log_offset: float = 1e-6, window_length_secs: float = 0.025, hop_length_secs: float = 0.010,**kwargs) -> np.ndarray:
    ''' Takes audio waveform data and returns a log-magnitude mel-frequency.
    Inputs:
        data(np.ndarray): A numpy ndarray of audio waveform data
        audio_sample_rate(int): An integer specifying the sample rate of the audio.
        log_offset(float): A float of log offset to add to values when taking log to avoid negative infinites.
        window_length_secs(float): A float specifying the window length in seconds to analyse.
        **kwargs: additional keyword arguments to pass to spectrogram_to_mel_matrix().
    Returns:
        np.ndarray: A 2D numpy ndarray of (num_frames, num_mel_bins) containing log mel filterbank magnitude for successive rates.
    '''

    # Constants
    window_length_samples = int(round(audio_sample_rate * window_length_secs))  # Assigning the sample length of a window to the rounded integer value of the sample rate of the audio multiplied by the window length in seconds to analyse
    hop_length_samples = int(round(audio_sample_rate * hop_length_secs))  # Assigning the length of the samples in each hop to the rounded integer value of the sample rate of the audio multiplied by the window length.
    fft_length = 2 ** int(np.ceil(np.log2(window_length_samples)))  # Assigning the length of the Fourier Form to the ceiling integer (the smallest number that is greater or equal to the specified number) of the length of samples in each window log squared.

    # Calculating the STFT magnitude
    spectrogram = stft_magnitude(data,fft_length=fft_length,hop_length=hop_length_samples, window_length=window_length_samples)

    #Calculating the Mel spectrogram

    mel_spectrogram = np.dot(spectrogram,spectrogram_to_mel_matrix(num_spectrogram_bins=spectrogram.shape[1],audio_sample_rate=audio_sample_rate,**kwargs))

    # Returning the log Mel spectrogram
    return np.log(mel_spectrogram + log_offset)




