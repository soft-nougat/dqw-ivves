from numpy.core.fromnumeric import sort
import streamlit as st
import numpy as np
import pandas as pd
from helper_functions import get_augmentations, update_augmentations, export

def show_sizes(images):
    image_sizes: np.ndarray = np.array([x[1].size for x in images]).astype(int)
    sizes = pd.DataFrame({
        "width":[x[0] for x in image_sizes], 
        "height":[x[1] for x in image_sizes]
        })
    sizes = sizes.groupby(['width','height']).size().reset_index().rename(columns={0:'counts'})
    st.vega_lite_chart(sizes, {
        'mark': {'type': 'circle', 'tooltip': True},
        'encoding': {
            'x': {'field': 'width', 'type': 'quantitative'},
            'y': {'field': 'height', 'type': 'quantitative'},
            'size': {'field': 'counts', 'type': 'quantitative'},
            'color': {'field': 'counts', 'type': 'quantitative'},
        },
    }, use_container_width = True)

def show_grid(images):
    images_per_row = st.slider("The number of images per row", step = 1, value = 4, min_value = 1, max_value = 8)
    caption_type = st.selectbox('Select image caption', ('None', 'File type', 'File name', 'File size'))
    st.subheader('A preview of the images is displayed below, please wait for data to be analysed :bar_chart:')
    n_rows = len(images) / images_per_row
    n_rows = int(np.ceil(n_rows))
    for row_num in range(n_rows):
        cols = st.columns(images_per_row)
        start = row_num*images_per_row
        end = start + images_per_row
        if end > len(images):
            end = len(images)
        for col, (detail, image) in zip(cols, images[start:end]):
            col.image(image, use_column_width=True, caption =detail[caption_type])

def show_histograms(images):
    for _, image in images:
        channels = image.split()
        
        colors = ['Red', 'Green', 'Blue']
        if len(channels) == 1:
            colors = ['Gray']
        cols = st.columns(2)
        cols[0].image(image, use_column_width=True)
        colors_df = None
        for channel, color in zip(channels, colors):
            data = {}
            data['pixel'] = np.asarray(channel).ravel()
            # data['pixel'] = np.around((data['pixel'] - data['pixel'].min()) / (data['pixel'].max() - data['pixel'].min()), decimals = 2)
            df = pd.DataFrame(data)
            df = df.groupby('pixel').size().reset_index().rename(columns={0:'counts'})
            df['color'] = color
            if colors_df is None:
                colors_df = df
            else:
                colors_df = pd.concat([colors_df, df], ignore_index=True)

        cols[1].vega_lite_chart(colors_df, {
                'title': f'Pixel intensity histogram for color channel(s)',
                'mark': {'type': 'bar', "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
                'encoding': {
                    'x': {'field': 'pixel', 'type': 'quantitative', 'title': 'Pixel value'},
                    'y': {"field": "counts", 'type': 'quantitative', 'title': 'Number of pixels'},
                    'color': {'field': 'color', "type": "nominal", "scale": {"domain": colors, "range": colors, "type": "ordinal"}}
                },
            }, use_container_width = True)

def show_channels(images):
    for _, image in images:
        channels = image.split()
        
        colors = ['Red', 'Green', 'Blue']
        if len(channels) == 1:
            colors = ['Gray']
        cols = st.columns(len(colors) + 1)
        cols[0].image(image, use_column_width=True)
        for col, channel, color in zip(cols[1:], channels, colors):
            channel = np.asarray(channel)
            col.image(channel, use_column_width=True, caption = f'{color} channel')

def augmentations(images):
    augmentation = st.selectbox("Choose augmentation method", 
                                      ("Resize", 
                                       "Grayscale",
                                       "Contrast enhancement",
                                       "Brightness enhancement",
                                       "Sharpness enhancement",
                                       "Color enhancement",
                                       "Denoise"))
    augmentations = get_augmentations()
    apply_text = 'Apply ▶️'
    revert_text = 'Revert ◀️'
    eda = 'EDA 📊'
    
    # resize section --------------------------------------
    if augmentation == "Resize":
        new_width  = st.number_input('Image width', min_value=32, max_value=2048, value=300)
        new_height = st.number_input('Image height', min_value=32, max_value=2048, value=200)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            is_applied = st.button(apply_text)
        if is_applied:
            augmentations["resize"]["width"] = new_width
            augmentations["resize"]["height"] = new_height
            update_augmentations(augmentations)
            st.success("Succesfully resized images")
        with col2:    
            is_eda = st.button(eda)
        if is_eda:
            show_eda(images)
        if augmentations["resize"]["width"] is not None and augmentations["resize"]["height"] is not None:
            with col3:
                is_reverted = st.button(revert_text)
            if is_reverted: 
                augmentations["resize"]["width"] = None
                augmentations["resize"]["height"] = None
                update_augmentations(augmentations)
    # grayscale section -------------------------------------
    elif augmentation == "Grayscale":
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            is_applied = st.button(apply_text)
        if is_applied:
            augmentations["grayscale"] = True
            update_augmentations(augmentations)
            st.success("Succesfully converted images to grayscale")
        with col2:    
            is_eda = st.button(eda)
        if is_eda:
            show_eda(images)
        if augmentations["grayscale"] == True:
            with col3:
                is_reverted = st.button(revert_text)
            if is_reverted: 
                augmentations["grayscale"] = False
                update_augmentations(augmentations)
    # contrast section ---------------------------------------
    elif augmentation == "Contrast enhancement":
        value = st.slider('Contrast level', min_value=0.1, max_value=5., value=0.5)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            is_applied = st.button(apply_text)
        if is_applied:
            augmentations["contrast"]["value"] = value
            update_augmentations(augmentations)
            st.success("Succesfully enhanced contrast of images")
        with col2:    
            is_eda = st.button(eda)
        if is_eda:
            show_eda(images)
        if augmentations["contrast"]["value"] is not None:
            with col3:
                is_reverted = st.button(revert_text)
            if is_reverted: 
                augmentations["contrast"]["value"] = None
                update_augmentations(augmentations)
    elif augmentation == "Brightness enhancement":
        value = st.slider('Brightness level', min_value=0.1, max_value=5., value=1.5)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            is_applied = st.button(apply_text)
        if is_applied:
            augmentations["brightness"]["value"] = value
            update_augmentations(augmentations)
            st.success("Succesfully enhanced brightness of images")
        with col2:    
            is_eda = st.button(eda)
        if is_eda:
            show_eda(images)
        if augmentations["brightness"]["value"] is not None:
            with col3:
                is_reverted = st.button(revert_text)
            if is_reverted: 
                augmentations["brightness"]["value"] = None
                update_augmentations(augmentations)
    # sharpness section ---------------------------------------------
    elif augmentation == "Sharpness enhancement":
        value = st.slider('Sharpness level', min_value=0.1, max_value=5., value=2.)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            is_applied = st.button(apply_text)
        if is_applied:
            augmentations["sharpness"]["value"] = value
            update_augmentations(augmentations)
            st.success("Succesfully enhanced sharpness of images")
        with col2:    
            is_eda = st.button(eda)
        if is_eda:
            show_eda(images)
        if augmentations["sharpness"]["value"] is not None:
            with col3:
                is_reverted = st.button(revert_text)
            if is_reverted: 
                augmentations["sharpness"]["value"] = None
                update_augmentations(augmentations)
    # color enhancement ------------------------------------------------
    elif augmentation == "Color enhancement":
        value = st.slider('Color level', min_value=0.1, max_value=5., value=2.)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            is_applied = st.button(apply_text)
        if is_applied:
            augmentations["color"]["value"] = value
            update_augmentations(augmentations)
            st.success("Succesfully enhanced color of images")
        with col2:    
            is_eda = st.button(eda)
        if is_eda:
            show_eda(images)
        if augmentations["color"]["value"] is not None:
            with col3:
                is_reverted = st.button(revert_text)
            if is_reverted: 
                augmentations["color"]["value"] = None
                update_augmentations(augmentations)
    # denoise section ---------------------------------------------
    elif augmentation == "Denoise":
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            is_applied = st.button(apply_text)
        if is_applied:
            augmentations["denoise"] = True
            update_augmentations(augmentations)
            st.success("Succesfully denoised images")
        with col2:    
            is_eda = st.button(eda)
        if is_eda:
            show_eda(images)
        if augmentations["denoise"] is not None:
            with col3:
                is_reverted = st.button(revert_text)
            if is_reverted: 
                augmentations["denoise"] = False
                update_augmentations(augmentations)

def export_images(images):
    try:
        export(images)
    except Exception as e:
        print(e)

def show_eda(images):
    show_grid(images)
    show_sizes(images)
    show_histograms(images)
    show_channels(images)