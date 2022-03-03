"""
A script with the strucutured data analysis logic
Additional scripts: report_generation
"""

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from tabular_eda.te import *
from helper_functions import display_app_header, generate_zip_structured, sub_text, generate_zip_pp, open_html
from tabular_eda.report_generation import create_pdf_html, create_pdf
import sweetviz as sv
import pycaret as pyc

from sklearn import set_config
from sklearn.utils import estimator_html_repr

def structured_data_app():

    # the very necessary reference expander
    intro_text = """
    Welcome to the DQW for structured data analysis.
    Structured data analysis is an important step 
    in AI model development or Data Analysis. This app 
    offers visualisation of descriptive statistics of a 
    csv input file. 
    <br> There are 3 options you can use: 
    <li> - Analyse 1 file using <a href = ""> pandas-profiling </a>
    <li> - Peprocess 1 file with <a href = "https://github.com/pycaret/pycaret"> PyCaret </a> 
    and compare with <a href = "https://github.com/fbdesignpro/sweetviz"> Sweetviz </a> and download preprocessing pipeline
    together with the datasets used
    <li> - Compare 2 files with Sweetviz
    <li> - Analyse synthetic data with <a href = "https://github.com/Baukebrenninkmeijer/table-evaluator"> table-evaluator </a> </li>
    <br> You can download pdf files/reports at the end of each analysis.
    """
    intro = st.expander("Click here for more info on this app section and packages used")
    with intro:
        sub_text(intro_text)

    # Side panel setup
    display_app_header(main_txt = "Step 1",
                       sub_txt= "Choose type of analysis",
                       is_sidebar=True)

    selected_structure = st.sidebar.selectbox("", 
                                                ("Analyse 1 file", 
                                                "Compare 2 files",
                                                "Synthetic data comparison"))

    display_app_header(main_txt = "Step 2",
                    sub_txt= "Upload data",
                    is_sidebar=True)

    
    if selected_structure == "Analyse 1 file":

        st.session_state.data = upload_file()

        if st.session_state.data is not None:

            display_app_header(main_txt = "Step 3",
                            sub_txt= "Choose next step",
                            is_sidebar=True)

            step_3 = st.sidebar.selectbox("",
            ("None", "EDA", "Preprocess and compare"))

            if step_3 == "EDA":

                analyse_file(st.session_state.data)

            elif step_3 == "Preprocess and compare":

                st.session_state.data = preprocess(st.session_state.data)

            else:

                st.warning("Please select next step in sidebar.")
           
    if selected_structure == "Compare 2 files":
        
        sweetviz_comparison(None, None, 0, text = "Step 3")
    
    if selected_structure == "Synthetic data comparison":
        
        table_evaluator_comparison()

def upload_file():

    demo = st.sidebar.checkbox('Use demo data', value=False, help='Using the adult dataset')
    if demo:
        data = pd.read_csv('demo_data/tabular_demo.csv')
        return(data)
    else:
        data = st.sidebar.file_uploader("Upload dataset", 
                                type="csv") 

        if data:

            st.subheader('A preview of input data is below, please wait for data to be analyzed :bar_chart:')
            data = pd.read_csv(data)
            st.write(data.head(5))

            return(data)

        else:
            st.sidebar.warning("Please upload a dataset!")

            return(None)
        
            
def upload_2_files():
    """
    High level app logic when comparing 2 files
    """
    demo = st.sidebar.checkbox('Use demo data', value=False, help='Using the table-evaluator demo datasets')
    if demo:
        original = pd.read_csv('demo_data/real_test_sample.csv')
        comparison = pd.read_csv('demo_data/fake_test_sample.csv')
        indicator = 1
    else:
        original = st.sidebar.file_uploader("Upload reference dataset", 
                                            type="csv")

        if original:

            original = pd.read_csv(original)       
            indicator = 0                 

            comparison = st.sidebar.file_uploader("Upload comparison dataset", 
                                                    type="csv") 

            if comparison:                      
            
                comparison = pd.read_csv(comparison)
                indicator = 1

        else:
            st.sidebar.warning("Please upload a reference/original dataset.")
            indicator = 0
            return(None, None, indicator)

    # if data is available, continue with the app logic
    if indicator == 1:
        st.subheader('A preview of input files is below, please wait for data to be compared :bar_chart:')
        st.subheader('Reference data')
        st.write(original.head(5))
        st.subheader('Comparison data')
        st.write(comparison.head(5))

        return(original, comparison, 1)

def sweetviz_comparison(original, comparison, indicator, text, upload = True):

    """
    Function to compare test and train data with sweetviz
    """
    if upload == True:
        # call high level function and get files
        original, comparison, indicator = upload_2_files()

    # use indicator to stop the app from running
    if indicator == 1: 

        sw = sv.compare([original, "Original"], [comparison, "Comparison"])

        sw.show_html(open_browser=False, layout='vertical', scale=1.0)

        display = open("SWEETVIZ_REPORT.html", 'r', encoding='utf-8')

        source_code = display.read()

        components.html(source_code, height=1200, scrolling=True)

        create_pdf_html("SWEETVIZ_REPORT.html",
                        text,
                        "sweetviz_dqw.pdf")

        return(sw)


def table_evaluator_comparison():

    """
    The portion of structured data app dedicated to file comparison with table-evaluator
    We have 2 options, plot differences or choose categorical column to analyse
    """

    # call high level function and get files
    original, comparison, indicator = upload_2_files()

    if indicator == 1: 
        # Side panel setup
        display_app_header(main_txt = "Step 3",
                        sub_txt= "Choose table-evaluator method",
                        is_sidebar=True)

        selected_method = st.sidebar.selectbox("", 
                                                ("Plot the differences", 
                                                "Compare model performance"))

        if selected_method == "Plot the differences":

            table_evaluator = TableEvaluator(original, comparison)
            table_evaluator.visual_evaluation()

            # Side panel setup
            display_app_header(main_txt = "Step 4",
                            sub_txt= "Download pdf report",
                            is_sidebar=True)

            with st.spinner("The pdf is being generated..."):
                create_pdf(original, comparison)
            st.success('Done! Please refer to sidebar, Step 4 for download.')

            zip = generate_zip_structured(original, comparison)

            with open("pdf_files/synthetic_data/report_files_dqw.zip", "rb") as fp:
                st.sidebar.download_button(
                        "⬇️",
                    data=fp,
                    file_name="te_compare_files_dqw.zip",
                    mime="application/zip"
                )

        else:
            
            # additional analysis part -------
            # insert an additional None column to options to stop the app 
            # from running on a wrong column
            dataset_columns = original.columns
            options_columns = dataset_columns.insert(0, 'None')
            
            evaluate_col = st.selectbox('Select the target column:', 
                                        options_columns, 
                                        index=0)
        
            if evaluate_col != 'None':

                table_evaluator = TableEvaluator(original, comparison)
                evaluate = table_evaluator.evaluate(target_col = evaluate_col)

            else:

                st.sidebar.warning('Please select a categorical column to analyse.')

        
def analyse_file(data):

    """
    The portion of structured data app dedicated to 1 file analysis with pandas-profiling
    """
    
    # generate a report and save it 
    pr = data.profile_report()
    st_profile_report(pr)
    pr.to_file("pandas_prof.html")
    
    create_pdf_html("pandas_prof.html",
                    "Step 4",
                    "pandas_profiling_dqw.pdf")

    return(pr)

def preprocess(data):
    """
    Automated preprocessing of the structured dataset w/ pycaret
    """
    # show pycaret info
    pycaret_info = st.expander("Click here for more info on PyCaret methods used")
    with pycaret_info:
        text = """
        PyCaret is an exteremly useful low-code ML library. It helps automate ML workflows.
        <br>In this part of the app, you can pass a tabular dataset to the PyCaret setup function
        which runs the following preprocessing steps on your data:
        <br><li> - Missing value removal
        <li> - One-hot encoding of categorical features
        <li> - Outlier mitigation
        <li> - Target imbalance mitigation </li>
        <br> Why do we select the model we are preparing the data for? This is
        important as PyCaret's setup method works differently for supervised and unsupervised models.
        For the former, we need to specify a target column (label). For the latter, this is not neccessary.
        <br> The output of the setup function are:
        <li> - The preprocessed dataset, which you can compare with the original one using Sweetviz.
        <li> - The train and test datasets, which you can compare with each other using Sweetviz.
        """
        sub_text(text)
    
    # class column is the label - ask the user to select - not necessary for unsupervised
    model = st.selectbox('Select the type of model you are preparing data for:',
    ('None', 'Unsupervised', 'Supervised'))

    dataset_columns = data.columns
    options_columns = dataset_columns.insert(0, 'None')

    # unsupervised
    if model == 'Unsupervised':

        from pycaret.clustering import setup, get_config, save_config

        pyc_user_methods = methods_pyc(options_columns, model)

        clf_unsup = setup(data = data, 
                          silent = True, 
                          numeric_imputation = pyc_user_methods[0],
                          categorical_imputation = pyc_user_methods[1],
                          ignore_features = pyc_user_methods[2],
                          high_cardinality_features = pyc_user_methods[3],
                          high_cardinality_method = pyc_user_methods[4],
                          #remove_outliers = pyc_user_methods[6],
                          #outliers_threshold = pyc_user_methods[7],
                          normalize = pyc_user_methods[8],
                          normalize_method = pyc_user_methods[9],
                          transformation = pyc_user_methods[10],
                          transformation_method = pyc_user_methods[11]
                          )

         # save pipeline
        save_config("pdf_files/preprocessed_data/pycaret_pipeline.pkl")

        # save html of the sklearn data pipeline
        set_config(display = 'diagram')

        pipeline = get_config('prep_pipe')

        with open('prep_pipe.html', 'w') as f:  
            f.write(estimator_html_repr(pipeline))

        show_pp_file(data, get_config('X'))

    # superivised
    elif model != 'Unsupervised':

        from pycaret.classification import setup,  get_config, save_config
        
        label_col = st.selectbox('Select the label column:', 
                                    options_columns, 
                                    index=0)
  
        if label_col != 'None':
    
            pyc_user_methods = methods_pyc(options_columns, model)

            clf_sup = setup(data = data, 
                          silent = True, 
                          target = label_col, 
                          numeric_imputation = pyc_user_methods[0],
                          categorical_imputation = pyc_user_methods[1],
                          ignore_features = pyc_user_methods[2],
                          high_cardinality_features = pyc_user_methods[3],
                          high_cardinality_method = pyc_user_methods[4],
                          fix_imbalance = pyc_user_methods[5],
                          remove_outliers = pyc_user_methods[6],
                          outliers_threshold = pyc_user_methods[7],
                          normalize = pyc_user_methods[8],
                          normalize_method = pyc_user_methods[9],
                          transformation = pyc_user_methods[10],
                          transformation_method = pyc_user_methods[11]
                          )

            # save pipeline
            save_config("pdf_files/preprocessed_data/pycaret_pipeline.pkl")

            # save html of the sklearn data pipeline
            set_config(display = 'diagram')

            pipeline = get_config('prep_pipe')

            with open('prep_pipe.html', 'w') as f:  
                f.write(estimator_html_repr(pipeline))

            show_pp_file(data, get_config('X'), get_config('X_train'), get_config('X_test'),
            get_config('y'), get_config('y_train'), get_config('y_test'))
 
def show_pp_file(data, X, X_train = None, X_test = None, y = None, y_train = None, y_test = None):
    
    st.subheader("Preprocessing done! 🧼")
    st.write("A preview of data and the preprocessing pipeline is below.")
    st.write(X.head())
    open_html('prep_pipe.html', height = 400, width = 300)

    st.subheader("Compare files 👀")

    if X_train is not None:
        compare_type = st.selectbox('Select which files to compare:',
        ('Original & preprocessed', 'Train & test'))

        if compare_type == 'Original & preprocessed':

            sweetviz_comparison(data, X, 1, text = "Step 4", upload = False)
        
        else:

            sweetviz_comparison(X_train, X_test, 1, text = "Step 4", upload = False)
    else:

        st.write("Compare original and preprocessed data")
        sweetviz_comparison(data, X, 1, text = "Step 4", upload = False)

    # download files
    zip = generate_zip_pp(data, X, X_train, X_test, y, y_train, y_test)

    display_app_header(main_txt = "Step 5",
                    sub_txt= "Download preprocessed files",
                    is_sidebar=True)

    with open("pdf_files/preprocessed_data.zip", "rb") as fp:
        st.sidebar.download_button(
                "⬇️",
            data=fp,
            file_name="preprocessed_data_dqw.zip",
            mime="application/zip"
        )

def methods_pyc(columns, model):
    """
    Define whwich imputation method to run on missing values
    Define which features to ignore
    Define miscellaneous methods
    """

    st.subheader("Missing values")
    sub_text("Select imputation methods for both numerical and categorical columns.")
    imputation_num = st.selectbox("Select missing values imputation method for numerical features:",
    ("mean", "median", "zero"))

    imputation_cat = st.selectbox("Select missing values imputation method for categorical features:",
    ("constant", "mode"))

    sub_text("Select which columns to skip preprocessing for.")
    ignore = st.multiselect("Select which columns to ignore:",
    (columns), default = None)

    #if ignore != 'None':
        # if only 1 column is selected, we need to pass a list
        #if type(ignore) is str:
            #cardinal = [ignore]

    #sub_text("Select which columns you want to run ordinal one-hot encoding for. Ordinal features need to be in a specific order.")
    #sub_text("An example would be low < medium < high.")
    #ordinal = st.selectbox("Select ordinal columns:",
    #(columns))

    #ordinal_values = st.text_input("Write ordered ordinal values separated by a semi-colon (;)",
    #help="Example input: low; medium; high")
    st.subheader("Cardinal one-hot encoding")
    sub_text("Select the columns that have high cardinality, i.e., that contain variables with many levels.")
    cardinal = st.multiselect("Select cardinal columns:",
    (columns), default = None)

    cardinal_method = None

    if cardinal != 'None':
        # if only 1 column is selected, we need to pass a list
        #if type(cardinal) is str:
            #cardinal = [cardinal]

        text = """
        Below are the avaluable cardinality methods. If frequency is selected, the original value is replaced 
        with the frequency distribution. If clustering is selected, statistical attributes of data are clustered 
        and replaces the original value of the feature is replaced with the cluster label. 
        The number of clusters is determined using a combination of Calinski-Harabasz and Silhouette criteria.
        """
        sub_text(text)
        cardinal_method = st.selectbox("Select cardinal encoding method:",
        ("frequency", "clustering"))

    if model != 'Unsupervised':
        st.subheader("Resampling and bias mitigation")
        sub_text("When the training dataset has an unequal distribution of target class it can be fixed using the fix_imbalance parameter in the setup. When set to True, SMOTE (Synthetic Minority Over-sampling Technique) is used as a default method for resampling.")
        resampling = st.checkbox("Activate resampling")

        st.subheader("Outlier mitigation")
        sub_text("The Remove Outliers function in PyCaret allows you to identify and remove outliers from the dataset before training the model. Outliers are identified through PCA linear dimensionality reduction using the Singular Value Decomposition technique.")
        mitigation = st.checkbox("Activate outlier mitigation")

        mitigation_method = 0.05

        if mitigation:

            mitigation_method = st.slider("Pick mitigation threshold:",
            min_value = 0.01, max_value = 0.1, value = 0.05)
        
    else:
        resampling = None
        mitigation = None
        mitigation_method = None

    st.subheader("Normalize")
    sub_text("Normalization is a technique often applied as part of data preparation for machine learning. The goal of normalization is to rescale the values of numeric columns in the dataset without distorting differences in the ranges of values or losing information.")
    normalization = st.checkbox("Activate normalization")

    normalization_method = "zscore"

    if normalization:
       text="""
       <li> <b>Z-score</b> is a numerical measurement that describes a value's relationship to the mean of a group of values"
       <li> <b>minmax</b> scales and translates each feature individually such that it is in the range of 0 – 1
       <li> <b>maxabs</b> scales and translates each feature individually such that the maximal absolute value of each feature will be 1.0. Sparsity is left intact.
       <li> <b>robust</b> scales and translates each feature according to the Interquartile range. Use in case there's outliers.
       """
       sub_text(text)
       normalization_method = st.selectbox("Pick normalization method:",
        ("zscore", "minmax", "maxabs", "robust"))

    st.subheader("Feature Transform")
    sub_text(" Transformation changes the shape of the distribution such that the transformed data can be represented by a normal or approximate normal distribution.")
    feat_trans = st.checkbox("Activate feature transformation")

    feat_trans_method = "yeo-johnson"

    if feat_trans:
       text="""
        There are two methods available for transformation yeo-johnson and quantile.
       """
       sub_text(text)
       feat_trans_method = st.selectbox("Pick feature transformation method:",
        ("yeo-johnson", "quantile"))   

  
    return([imputation_num, imputation_cat, ignore, 
    cardinal, cardinal_method, resampling, 
    mitigation, mitigation_method, normalization, 
    normalization_method, 
    feat_trans, feat_trans_method])


def detect_unfairness(X, y, data, label_column):
    """  
    Not used currently ---

    Use the fat-forensics package to assess fairness of data
    Currently, only the accountability example is demoed

    The logic: identify sampling bias for a data set grouping for 
    a selected feature
    
    Returns: a bias estimation
    """
    # prepare data for fat forensics, numpy array needed
    X_fairness = X.to_numpy()

    # extract the labels 
    class_names = data[label_column].unique 
    feature_names = X.columns

    selected_feature_name = st.selectbox("Select a feature to measure the sampling bias:",
    (feature_names))

    selected_feature = np.where(feature_names == selected_feature_name)
    selected_feature_index = int(selected_feature[0])

    # Define grouping on the selected feature
    selected_feature_grouping = [2.5, 4.75]

    # Get counts, weights and names of the specified grouping
    grp_counts, grp_weights, grp_names = fatf_dam.sampling_bias(
        X_fairness, selected_feature_index, selected_feature_grouping)

    # Print out counts per group
    print('The counts for groups defined on "{}" feature (index {}) are:'
        ''.format(selected_feature_name, selected_feature_index))
    for g_name, g_count in zip(grp_names, grp_counts):
        is_are = 'is' if g_count == 1 else 'are'
        print('    * For the population split *{}* there {}: '
            '{} data points.'.format(g_name, is_are, g_count))

    # Get the disparity grid
    bias_grid = fatf_dam.sampling_bias_grid_check(grp_counts)

    # Print out disparity per every grouping pair
    print('\nThe Sampling Bias for *{}* feature (index {}) grouping is:'
        ''.format(selected_feature_name, selected_feature_index))
    for grouping_i, grouping_name_i in enumerate(grp_names):
        j_offset = grouping_i + 1
        for grouping_j, grouping_name_j in enumerate(grp_names[j_offset:]):
            grouping_j += j_offset
            is_not = '' if bias_grid[grouping_i, grouping_j] else ' no'

            print('    * For "{}" and "{}" groupings there >is{}< Sampling Bias.'
                ''.format(grouping_name_i, grouping_name_j, is_not))