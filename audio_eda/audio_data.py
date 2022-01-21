"""
Created on Tue Dec 21 14:06:51 2021
The audio data eda, augmentation and comparison script

@author: TNIKOLIC

References:
1 file EDA
https://github.com/musikalkemist/DeepLearningForAudioWithPython/blob/master/11-%20Preprocessing%20audio%20data%20for%20deep%20learning/code/audio_prep.py
Augmentations
https://github.com/phrasenmaeher/audio-transformation-visualization
DTW
https://librosa.org/doc/0.7.0/auto_examples/plot_music_sync.html
https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html
https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_AudioMatching.html 
Spectrum compare
https://towardsdatascience.com/a-data-scientists-approach-to-visual-audio-comparison-fa15a5d3dcef
https://github.com/QED0711/audio_analyzer
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
from scipy.spatial.distance import euclidean
import matplotlib
from audio_eda.AudioAnalyzer import *

FIG_SIZE = (15,10)

def augment_audio(augmentation_list, file):
    """
    A function to take an audio file and augment it using audiomentations.

    Arguments:
        augmentation_list - passed as an audiomentation object, user defined in app
        file - uploaded audio file

    Returns: audio file and EDA report
    """

    st.write("Augment audio file. ",
    "We use the audiomentations package to agment the uploaded audio file. ",
    "After that, we run the EDA on the new file. You can download the augmented ",
    "file by clicking the three dots in the streamlit audio player and selecting Download.")

    # load audio file with Librosa
    audio, sr = load_audio_sample(file)

    augment = Compose(augmentation_list)

    # Augment/transform/perturb the audio data
    augmented_sample = augment(samples=audio, sample_rate=sr)
    
    st.write("Augmented file")
    st.audio(ap_augmented(augmented_sample, sr))

    audio_eda(ap_augmented(augmented_sample, sr))

def audiocompare(reference, comparison):
    st.write("Performing comparison between 2 sequences. ",
    "The functions were taken from [QED0711/audio-analyser](https://github.com/QED0711/audio_analyzer/).")

    ref = AudioAnalyzer(reference, input_sr=None, fft_size=44100)
    comp = AudioAnalyzer(comparison, input_sr=None, fft_size=44100)

    compare = SpectrumCompare(ref, comp)

    st.write("Compare 2 spectrums with applied treshold. ",
    "The amplitude difference is in green, red being the treshold line. ",
    "If the green line is above red, reference has more amplitude and vice versa.")

    compare.plot_spectrum_group(frange=(20,1000), 
        ratio=True, 
        threshold=True,
        title="Spectrograms", 
        legend=("Reference", "Comparison", "Amplitude Diff.", "Threshold")
        )

    st.pyplot()

    st.write("Plot spectrum heatmap difference. ",
    "Here the amplitude difference is converted to a heatmap, hotter spots ",
    "indicating a larger difference in favor of the comparison input and vice versa.")

    compare.plot_spectrum_heatmap(
        frange=(20,1000), 
        plot_spec1=False,
        title="Reference vs Comparison"
    )

    st.pyplot()

def compare_files(reference, comparison):
    """
    A function to compare 2 files using AudioCompare and DTW.

    DTW is a module in librosa, but also a standalone package.
    We use both.

    Squeeze the audio file with librosa, extract chromagrams and 
    compute the difference. 

    Takes in 2 files and outputs a plotted difference.
    """

    st.write("Performing Dynamic Time Warping to compare 2 sequences. ",
    "DTW is a method that calculates an optimal match between two given sequences. ",
    "DTW can be used once we extract chromagrams or mfccs using librosa. ",
    "Since DTW requires some empirical studies and is specific to the UC, we have ",
    "included 2 approaches in this part of the app - using DTW and librosa.dtw.")

    y1, sr1 = librosa.load(reference) 
    y2, sr2 = librosa.load(comparison) 
    
    # Approach 1 - dtw package
    # Show reference and comparison MFCCs
    mfcc1_ax = plt.subplot(1, 2, 1) 
    mfcc1 = librosa.feature.mfcc(y1,sr1)   #Computing MFCC values
    librosa.display.specshow(mfcc1)
    mfcc1_ax.set(title='Reference File MFCC')

    mfcc2_ax =plt.subplot(1, 2, 2)
    mfcc2 = librosa.feature.mfcc(y2, sr2)
    librosa.display.specshow(mfcc2)
    mfcc2_ax.set(title='Comparison File MFCC')
    st.pyplot()

    # use dtw to get the normalized distance with euclidean argument
    # (manhattan can be used as well)
    dist, cost_matrix, acc_cost_matrix, path = dtw(mfcc1.T, mfcc2.T, dist=euclidean)
    st.write("The normalized euclidean dtw distance between the two is : ", dist, 
             ". This measure is found empirically and is use case specific.",
             "For example, [phoneme matching's treshold is 180](https://github.com/pierre-rouanet/dtw/issues/36). ",
             "For more infomation on the approach, please refer to the [DTW github page](https://github.com/pierre-rouanet/dtw).")   # 0 for similar audios 


    # https://github.com/QED0711/audio_analyzer/blob/master/main.ipynb


    # squeeze audio files, extract features to compare (chromagrams)
    X = librosa.feature.chroma_cens(y=y1, sr=sr1)
    Y = librosa.feature.chroma_cens(y=y2, sr=sr2)

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.title('Chroma Representation of Reference')
    librosa.display.specshow(X, x_axis='time',
                            y_axis='chroma')
    plt.colorbar()
    st.pyplot()

    plt.subplot(2, 1, 2)
    plt.title('Chroma Representation of Comparison')
    librosa.display.specshow(Y, x_axis='time',
                            y_axis='chroma')
    plt.colorbar()
    plt.tight_layout()
    st.pyplot()

    D, wp = librosa.sequence.dtw(X, Y, subseq=True)

    fig, ax = plt.subplots(nrows=1, sharex=True)
    img = librosa.display.specshow(D, x_axis='frames', y_axis='frames',
                                ax=ax)
    ax.set(title='DTW cost', xlabel='Reference', ylabel='Comparison')
    ax.plot(wp[:, 1], wp[:, 0], label='Optimal path', color='y')
    ax.legend()
    fig.colorbar(img, ax=ax)
    st.pyplot()  

    fig1, ax1 = plt.subplots(nrows=1, sharex=True)
    ax1.plot(D[-1, :] / wp.shape[0])
    ax1.set(xlim=[0, Y.shape[1]], ylim=[0, 2],
            title='Matching cost function')
    st.pyplot()  

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    # Plot Reference
    librosa.display.waveplot(y1, sr=sr1, ax=ax1)
    ax1.set(title='Reference')

    # Plot Comparison
    librosa.display.waveplot(y2, sr=sr2, ax=ax2)
    ax2.set(title='Comparison')

    plt.tight_layout()

    trans_figure = fig.transFigure.inverted()
    lines = []
    arrows = 30
    points_idx = np.int16(np.round(np.linspace(0, wp.shape[0] - 1, arrows)))

    hop_size = 430
    # for tp1, tp2 in zip((wp[points_idx, 0]) * hop_size, (wp[points_idx, 1]) * hop_size):
    for tp1, tp2 in wp[points_idx] * hop_size / sr1:
        # get position on axis for a given index-pair
        coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
        coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

        # draw a line
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                                    (coord1[1], coord2[1]),
                                    transform=fig.transFigure,
                                    color='r')
        lines.append(line)

    fig.lines = lines
    plt.tight_layout()
    st.pyplot()

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
    st.write("Use librosa.waveplot to diplay the [waveform](https://pudding.cool/2018/02/waveforms/) of the uploaded file. ",
             "The blue line is the data we're graphing, and it represents a sound wave. ",
             "Specifically, it's telling us about the wave's displacement, and how it changes over time.",
             "The x axis represents time in seconds, while the y axis represents amplitude (displacement).")
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
    st.write("Here we perform Fourier transformation of the input signal, ",
             "calculate magnitude and create a frequency variable to get the ",
             "power spectrum. The [power spectrum](https://en.wikipedia.org/wiki/Spectral_density) of a time series describes the",
             " distribution of power into frequency components composing that signal. ",
             "According to Fourier analysis, any physical signal can be decomposed ",
             "into a number of discrete frequencies, or a spectrum of frequencies over a ",
             "continuous range.")
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

    st.write("The Short-time Fourier transform (STFT) represents a signal ",
             "in the time-frequency domain by computing discrete Fourier ",
             "transforms (DFT) over short overlapping windows.",
             "Hop length is set to 512, while the window duration is set to 2048.")
    st.write("STFT hop length duration is: {}s".format(hop_length_duration))
    st.write("STFT window duration is: {}s".format(n_fft_duration))

    # perform stft
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

    # calculate abs values on complex numbers to get magnitude
    st.write("Use the STFT and display a spectrogram.")
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

    st.write("Use the STFT and display a spectrogram in decibels.")
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

    st.write("[Mel Frequency Cepstral coefficients (MFCCs)](https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd) ",
            "are used to represent ",
            "distinct units of sound as the shape of the human vocal tract. ",
            "For this, a Mel Scale is used to relate the percieved frequency to ",
            "the measured one, while the cepstral coefficients mesure the ",
            "rate of change in spectral bands.",
            "The color shows the feature value of each dimension at each time frame.")
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
                                 'Compare 2 files'))
    

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
            
            st.write("Uploaded file")
            st.audio(file_uploader, format="audio/wav", start_time=0)
    
            if action == 'EDA only':
                
                display_app_header(main_txt='',
                                   sub_txt='EDA results')

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
                
                st.write("Reference file")
                st.audio(reference, format="audio/wav", start_time=0)
                st.write("Comparison file")
                st.audio(comparison, format="audio/wav", start_time=0)
                
                which_comparison = st.sidebar.radio("Please pick a comparison method",
                                                    ("Spectrum Compare",
                                                    "DTW"))

                if which_comparison == "Spectrum Compare":
                    audiocompare(reference, comparison)
                else:
                    compare_files(reference, comparison)

        else:
        
            st.warning("Please upload files.")
    
