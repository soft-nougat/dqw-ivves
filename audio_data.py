"""
Created on Tue Dec 21 14:06:51 2021
The audio data eda and augmentation scripts

@author: TNIKOLIC
"""

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import librosa, librosa.display
import IPython
from IPython.display import Audio
import matplotlib.pyplot as plt
import streamlit as st
from helper_functions import *
import pydub
import io
from scipy.io import wavfile
from dtw import dtw
from scipy.spatial.distance import euclidean,mahalanobis

FIG_SIZE = (15,10)

def augment_audio(augmentation_list, file):
    # load audio file with Librosa
    audio, sr = load_audio_sample(file)

    augment = Compose(augmentation_list)

    # Augment/transform/perturb the audio data
    augmented_sample = augment(samples=audio, sample_rate=sr)
    
    st.audio(ap_augmented(augmented_sample, sr))

    audio_eda(ap_augmented(augmented_sample, sr))

def compare_files(reference, comparison):
   
    y1, sr1 = librosa.load(reference) 
    y2, sr2 = librosa.load(comparison) 

    #Showing multiple plots using subplot
    mfcc1_ax = plt.subplot(1, 2, 1) 
    mfcc1 = librosa.feature.mfcc(y1,sr1)   #Computing MFCC values
    librosa.display.specshow(mfcc1)
    mfcc1_ax.set(title='Reference File MFCC')

    mfcc2_ax =plt.subplot(1, 2, 2)
    mfcc2 = librosa.feature.mfcc(y2, sr2)
    librosa.display.specshow(mfcc2)
    mfcc2_ax.set(title='Comparison File MFCC')
    st.pyplot()

    # use dtw to get the normalized distance 
    dist, _, cost, path = dtw(mfcc1.T, mfcc2.T, dist=euclidean)
    st.write("The normalized distance between the two is : ", dist)   # 0 for similar audios 

    # squeeze audio files, extract features to compare (chromagrams)
    X = librosa.feature.chroma_cens(y=y1, sr=sr1)
    Y = librosa.feature.chroma_cens(y=y2, sr=sr2)

    D, wp = librosa.sequence.dtw(X, Y, subseq=True)

    fig, ax = plt.subplots(nrows=1, sharex=True)
    img = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
                                ax=ax)
    ax.set(title='DTW cost', xlabel='Reference', ylabel='Comparison')
    ax.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    ax.legend()
    fig.colorbar(img, ax=ax)
    st.pyplot()  #To display the plots graphically

    fig1, ax1 = plt.subplots(nrows=1, sharex=True)
    ax1.plot(D[-1, :] / wp.shape[0])
    ax1.set(xlim=[0, Y.shape[1]], ylim=[0, 2],
            title='Matching cost function')
    st.pyplot()  #To display the plots graphically

def create_audio_player(file):
    audio_file = open(file, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/ogg')

def ap_augmented(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)

    return virtualfile


def prepare_audio_file(uploaded_file):
    """
    A function to prepare the audio file uploaded by the user.
    
    Input: uploaded file passed by st.file_uploader
    Output: float32 numpy 
    """
    # use pydub to quickly prepare audio segment
    a = pydub.AudioSegment.from_file(file=uploaded_file, format=uploaded_file.name.split(".")[-1])

    # split channel sounds to mono
    channel_sounds = a.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    # convert to float32 so audiomentations can augment the file
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max

    return fp_arr[:, 0], a.frame_rate

def load_audio_sample(file):
    """ 
    A simple helper function to load file with librosa and set the sample rate.

    Input: audio file
    Output: audio and sample rate 
    """
    audio, sr = librosa.load(file, sr=22050)

    return audio, sr

def audio_eda(file):
    # load audio file with Librosa
    signal, sample_rate = load_audio_sample(file)

    # WAVEFORM
    # display waveform
    plt.figure(figsize=FIG_SIZE)
    librosa.display.waveplot(signal, sample_rate, alpha=0.4)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")
    st.pyplot()

    # FFT -> power spectrum
    # perform Fourier transform
    fft = np.fft.fft(signal)

    # calculate abs values on complex numbers to get magnitude
    spectrum = np.abs(fft)

    # create frequency variable
    f = np.linspace(0, sample_rate, len(spectrum))

    # take half of the spectrum and frequency
    left_spectrum = spectrum[:int(len(spectrum)/2)]
    left_f = f[:int(len(spectrum)/2)]

    # plot spectrum
    plt.figure(figsize=FIG_SIZE)
    plt.plot(left_f, left_spectrum, alpha=0.4)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Power spectrum")
    st.pyplot()

    # STFT -> spectrogram
    hop_length = 512 # in num. of samples
    n_fft = 2048 # window in num. of samples

    # calculate duration hop length and window in seconds
    hop_length_duration = float(hop_length)/sample_rate
    n_fft_duration = float(n_fft)/sample_rate

    st.write("STFT hop length duration is: {}s".format(hop_length_duration))
    st.write("STFT window duration is: {}s".format(n_fft_duration))

    # perform stft
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

    # calculate abs values on complex numbers to get magnitude
    spectrogram = np.abs(stft)

    # display spectrogram
    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar()
    plt.title("Spectrogram")
    st.pyplot()

    # apply logarithm to cast amplitude to Decibels
    log_spectrogram = librosa.amplitude_to_db(spectrogram)

    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram (dB)")
    st.pyplot()

    # MFCCs
    # extract 13 MFCCs
    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

    # display MFCCs
    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCC coefficients")
    plt.colorbar()
    plt.title("MFCCs")
    st.pyplot()


def audio_data_app():

    st.write("Welcome to the DQW for Audio data analysis. ",
            "As unstructured data, audio input analysis for ",
            "AI models is of crucial importance. This dashboard ",
            "offers visualisation of descriptive statistics of a ",
            "audio input file uploaded in form of wav and mp3. ",
            "Futhermore, you can compare 2 audio files and generate a ",
            "report on their differences. Finally, you can augment files ",
            "one by one and download them.",
            "Please upload file(s) on the left, pick if you wish to ",
            "augment, do EDA or compare them.")

    # Side panel setup
    display_app_header(main_txt = "Step 1",
                       sub_txt= "Upload data",
                       is_sidebar=True)

    next_step = st.sidebar.radio("",
                                ('Upload 1 file', 
                                 'Upload miltiple files'))
    

    # upload 1 or 2 files
    if next_step == 'Upload 1 file':
        file_uploader = st.sidebar.file_uploader(label="", 
                                                type=[".wav", ".wave", ".flac", ".mp3", ".ogg"])

        if file_uploader:

            display_app_header(main_txt = "Step 2",
                        sub_txt= "Select action",
                        is_sidebar=True)

            action = st.sidebar.radio("",
                                    ('EDA only', 
                                    'Augmentation'))

            st.audio(file_uploader, format="audio/wav", start_time=0)
    
            if action == 'EDA only':
                
                
                audio_eda(file_uploader)

            

            else:
                augmentation_methods = st.multiselect('Select augmentation method:', 
                ['AddGaussianNoise', 
                'TimeStretch', 
                'PitchShift', 
                'Shift'])  
                
                # add p values to each method and eval parse all list elements
                # so they are pushed to global environment as audiomentation methods
                augmentation_list = [i + "(p=1.0)" for i in augmentation_methods]
                augmentation_final = [eval(i) for i in augmentation_list]

                # pass the list to augmentation
                augment_audio(augmentation_final, file_uploader)
        else:

            st.warning("Please upload a file.")
    
    else:

        reference = st.sidebar.file_uploader(label="Please upload reference file", 
                                                 type=[".wav", ".wave", ".flac", ".mp3", ".ogg"])

        if reference:

            comparison = st.sidebar.file_uploader(label="Please upload comparison file", 
                                                    type=[".wav", ".wave", ".flac", ".mp3", ".ogg"])

            if comparison:

                #file_uploader = st.sidebar.file_uploader(label="Please upload comparison file", 
                                                        #type=[".wav", ".wave", ".flac", ".mp3", ".ogg"],
                                                        #accept_multiple_files=True)
                
                st.write("Reference file")
                st.audio(reference, format="audio/wav", start_time=0)
                st.write("Comparison file")
                st.audio(comparison, format="audio/wav", start_time=0)
            
                compare_files(reference, comparison)

        else:
        
            st.warning("Please upload files.")
    
