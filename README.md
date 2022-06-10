# Welcome to the DQW repository! 

This repo contains the complete DQW streamlit app code, however, the streamlit apps have been split into 5 for maintenance purposes:

- [Main Streamlit app 📊](https://share.streamlit.io/soft-nougat/dqw-ivves/app.py)
- [Tabular Data Section 🏗️](https://share.streamlit.io/soft-nougat/dqw-ivves_structured/main/app.py)
- [Audio Data Section 🎶](https://share.streamlit.io/soft-nougat/dqw-ivves_audio/main/app.py)
- [Text Data Section 📚](https://share.streamlit.io/soft-nougat/dqw-ivves_text/main/app.py)
- [Image Data Section 🖼️](https://share.streamlit.io/soft-nougat/dqw-ivves_images/main/app.py)

This application was built in the [ITEA IVVES](https://itea4.org/project/ivves.html) project and is an accelerator of the [Sogeti Quality AI Framework](https://www.sogeti.nl/nieuws/artificial-intelligence/blogs/artificial-intelligence-quality-framework), providing methods and advice on the preprocessing steps to accelerate and ensure transparency of the data preprocessing pipeline.

![The position of the DQW in the QAIF](/md_images/qaif.PNG "QAIF")

The DQW can be applied to the following data structures:
- Structured data. The most common data format used in data science, be it in finance, health, biotechnology, cybersecurity, etc. Since structured data is difficult for humans to grasp, it's very important to make it understandable prior to preparing it for training.
- Unstructured data. Unstructured data is easily understandable to humans, but needs to be thoroughly processed to use as training data for ML: Images used in computer vision algorithms such as object detection and classification; Text used in NLP models, be it for classification or sentiment analysis; Audio used in audio signal processing algorithms such as music genre recognition and automatic speech recognition.
- Synthetic data. Synthetic data evaluation is a critical step of the synthetic data generation pipeline. Validating the synthetic data training set to be used in an ML algorithm ensures model performance will not be impacted negatively. To evaluate synthetic data, you also need a portion of real data to compare it to.


The packages used in the application are in the table below.

| App section                |     Description    |     Visualisation    |     Selection    |     Package             |
|----------------------------|--------------------|----------------------|------------------|-------------------------|
|     Synthetic tabular      |          x         |           x          |                  |     [table-evaluator](https://github.com/Baukebrenninkmeijer/table-evaluator)     |
|     Tabular                |          x         |           x          |                  |     [sweetviz](https://github.com/fbdesignpro/sweetviz)            |
|     Tabular                |          x         |           x          |                  |     [pandas-profiling](https://github.com/pandas-profiling/pandas-profiling)    |
|     Tabular, text          |                    |                      |         x        |     [PyCaret](https://github.com/pycaret/pycaret)             |
|     Text                   |                    |                      |         x        |     [NLTK](https://github.com/nltk/nltk)               |
|     Text                   |                    |                      |         x        |     [SpaCy](https://github.com/explosion/spaCy)               |
|     Text                   |          x         |                      |         x        |     [TextBlob](https://github.com/sloria/TextBlob)            |
|     Text                   |          x         |           x          |                  |     [WordCloud](https://github.com/amueller/word_cloud)           |
|     Text                   |          x         |                      |         x        |     [TextStat](https://github.com/shivam5992/textstat)            |
|     Image                  |          x         |           x          |                  |     [Pillow](https://github.com/python-pillow/Pillow)              |
|     Audio                  |          x         |           x          |                  |     [librosa](https://github.com/librosa/librosa)             |
|     Audio                  |          x         |           x          |                  |     [dtw](https://github.com/pierre-rouanet/dtw)                 |
|     Audio                  |                    |                      |         x        |     [audiomentations](https://github.com/iver56/audiomentations)     |
|     Audio                  |          x         |           x          |                  |     [AudioAnalyser](https://github.com/QED0711/audio_analyzer)       |
|     Report generation      |          x         |                      |                  |     [Fpdf](https://github.com/Setasign/FPDF)                |
|     Report generation      |          x         |                      |                  |     [wkhtmltopdf](https://github.com/wkhtmltopdf/wkhtmltopdf)         |
|     Report   generation    |          x         |                      |                  |     [pdfkit](https://github.com/JazzCore/python-pdfkit)              |

## Structured (tabular) data 

Key points addressed:
- Quantitative measures – number of rows and columns. 
- Qualitative measures – column types. 
- Descriptive statistics with NumPy for numeric columns, for example, count, mean, percentiles and standard deviation. For discrete columns, count, unique, top and frequency. 
- Explore missing data. 
- Examine outliers.  
- Mitigate class imbalance.
- Compare datasets, like train, test and evaluate data.
- Evaluate synthetic datasets.
- Create a quality report.

To complete the key points, 4 subsections are created:
- One file EDA with pandas-profiling
- One file preporcessing with PyCaret
- Two file comparison with Sweetviz 
- Synthetic data evaluation with table-evaluator
- In all the sections, there is an option to download a pdf/zip of the results

## Unstructured data - text

Key points addressed:
- Frequency - Count most common words with WordCloud package in Python. This is the quickest way of seeing what the handled data contents are, in addition, it provides visualisation in form of word clouds. 
- Analyse sentiment with TextBlob in case of classification tasks. We can investigate the polarity of the text and represent it in form of bar graphs. 
- Investigate readability of data with Textstat, typically used for determining readability, complexity, and grade level of a corpus. 
- Topic analysis.
- Provide an automated preprocessing based on methods on the market.

To complete the key points, the following subsections are created:
- Preprocessing of the data with file download option
- Basic analysis methods like number of unique words, characters
- N-gram analysis with NLTK
- PoS tagging with NLTK
- NER with SpaCy
- Topic analysis with LDA, including optimal number of topics generation with u_mass coherence score

## Unstructured data - audio

Key points addressed:
- Provide EDA of audio files
- Augment audio files to showcase methods of increasing robustness of the audio dataset
- Provide methods of comparison of two files

To complete the key points, the following subsections are created:
- One file EDA with librosa
- One file augmentation with audiomentations
- Two file comparison with DTW
- Two file comparsion with audio_compare method

## Unstructured data - images

Key points addressed:
- EDA of images
- Augmentation

## Try it out yourself!

Demo files have been provided in the [/demo_data]() folder. 
Try out the app with them.

## How to run locally

1.	Installation process:

    Create virtual environment and activate it - https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/
    
    Clone or download files from this repo
    
    Run pip install -r requirements.txt
    
    Run streamlit run app.py to launch app

2.	Software dependencies:

    In requirements.txt

3.	Latest releases

    Use app.py
