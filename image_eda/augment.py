import streamlit as st
from PIL import Image, ImageEnhance
import cv2
import numpy as np

def apply_augmentations(images, augmentations) -> list:
    new_images = []
    for details, image in images:
        new_image = image
        if augmentations["resize"]["width"] is not None and augmentations["resize"]["height"] is not None:
            new_image =  new_image.resize((augmentations["resize"]["width"], augmentations["resize"]["height"]), Image.ANTIALIAS)
        if augmentations["grayscale"] == True:
            new_image = new_image.convert('L')
        if augmentations["contrast"]["value"] is not None:
            new_image = ImageEnhance.Contrast(new_image).enhance(augmentations["contrast"]["value"])
        if augmentations["brightness"]["value"] is not None:
            new_image = ImageEnhance.Brightness(new_image).enhance(augmentations["brightness"]["value"])
        if augmentations["sharpness"]["value"] is not None:
            new_image = ImageEnhance.Sharpness(new_image).enhance(augmentations["sharpness"]["value"])
        if augmentations["color"]["value"] is not None:
            new_image = ImageEnhance.Color(new_image).enhance(augmentations["color"]["value"])
        if augmentations["denoise"] == True:
            if len(new_image.split()) != 3:
                new_image = Image.fromarray(cv2.fastNlMeansDenoising(np.array(new_image)))
            else:
                new_image = Image.fromarray(cv2.fastNlMeansDenoisingColored(np.array(new_image)))
        new_images.append((details, new_image))
    return new_images