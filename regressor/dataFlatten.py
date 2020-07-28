# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:12:47 2019

@author: abhinav.jhanwar
"""

from keras.preprocessing.image import ImageDataGenerator

def trainGenerator(batch_size, train_path, data_frame, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", flag_multi_class=False,
                    save_to_dir=None, target_size=(256,256), seed=1):
    '''
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        directory = train_path,
        x_col = 'field',
        y_col = 'lai',
        target_size = target_size,
        color_mode = image_color_mode,
        class_mode = 'other',
        batch_size = batch_size,
        seed = seed,
        save_to_dir = save_to_dir
        )
    return image_generator

def validationGenerator(batch_size, train_path, data_frame, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", flag_multi_class=False,
                    save_to_dir=None, target_size=(256,256), seed=1):
    '''
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        directory = train_path,
        x_col = 'field',
        y_col = 'lai',
        target_size = target_size,
        color_mode = image_color_mode,
        class_mode = 'other',
        batch_size = batch_size,
        seed = seed,
        save_to_dir = save_to_dir
        )
    
    return image_generator

