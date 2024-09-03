import numpy as np
import matplotlib.pyplot as plt
import librosa
from stock_market.fl_multimodal_knowledge_graph.data_collection.data_collection_tools.download_sounds import download_sound
import os
# Pre-emphasis is a technique used in audio preprocessing to amplify higher frequencies of an audio signal, especially useful to enhance signal-to-noise ratio
# The pre-emphasis filter is typically a 1st-order high-pass filter, with the most common form of y[n] = x[n] - alpha . x[n-1]
# Where:
# y[n]: the output signal
# x[n]: the input signal
# alpha: pre-emphasis coefficient (often between 0.9 and 1.0), determining the pre-emphasis on the higher frequencies

def pre_emphasis(y, sr):

    download_sound()
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




