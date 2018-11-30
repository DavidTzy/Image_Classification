#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: tanzhenyu
"""
import os, sys, math, io
import numpy as np
import pandas as pd
import multiprocessing as mp
import bson
import struct
import keras
from keras.applications.xception import Xception
from keras.layers import Flatten, Dense, AveragePooling2D, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import keras.backend as K
from keras.metrics import top_k_categorical_accuracy
import math
import tensorflow as tf
from multiGPU1 import MultiGPUModel
from collections import defaultdict
from tqdm import *
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import threading
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from collections import Counter



input_path = 'your/input/path/'
output_path = "your/output/path"
train_path = os.path.join(input_path, "train.bson")
#num_train_products = 7069896
 
test_path = os.path.join(input_path, "test.bson")
#num_test_products = 1768182
 
categories_path = os.path.join(input_path,'category_names.csv')
categories_df = pd.read_csv(categories_path, index_col="category_id")
 
# Maps the category_id to an integer index. This is what we'll use to
# one-hot encode the labels.

def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat
cat2idx, idx2cat = make_category_tables()
 


categories_df["category_idx"]= pd.Series(range(len(categories_df)), index=categories_df.index)
categories_df.to_csv("categories.csv")
 



def read_bson(bson_path, num_records, with_categories):
    rows = {}
    with open(bson_path, "rb") as f:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0:
                break
            length = struct.unpack("<i", item_length_bytes)[0]
 
            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length
 
            item = bson.BSON.decode(item_data)
            product_id = item["_id"]
            num_imgs = len(item["imgs"])
 
            row = [num_imgs, offset, length]
            if with_categories:
                row += [item["category_id"]]
            rows[product_id] = row
 
            offset += length
            f.seek(offset)
 
    columns = ["num_imgs", "offset", "length"]
    if with_categories:
        columns += ["category_id"]
 
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df



train_offsets_df = read_bson(train_path, num_records=num_train_products, with_categories=True)


def make_val_set(df,max_num = 6000,threshold = 200,split_percentage=0.2, drop_percentage=0.5):
    '''max_num: the maximum number of product to keep for each category
       threshold: if the number of images of one category is below this threshold, we will do 
       some data augmentation for this category.   
    '''
    
    # Find the product_ids for each category
    category_dict = defaultdict(list)
    for ir in tqdm(df.itertuples()):
        category_dict[ir[4]].append(ir[0])
    
    
    img_dict = defaultdict(list)
    for ir in tqdm(df.itertuples()):
        img_dict[ir[4]] = img_dict.get(ir[4],0) + ir[1]
    
    train_list = []
    val_list = []
    with tqdm(total=len(df)) as pbar:
        for category_id, product_ids in category_dict.items():
            category_idx = cat2idx[category_id]

            keep_size = min(int(len(product_ids) * (1. - drop_percentage)),max_num)
                             
            if keep_size < len(product_ids) and img_dict[category_id] > threshold:
                
                product_ids = np.random.choice(product_ids, keep_size, replace=False)
            
            elif img_dict[category_id] <= threshold:
                
                add_size = (threshold-img_dict[category_id])//2
                add_ids = np.random.choice(product_ids,add_size, replace=True)
                product_ids = np.array(product_ids + list(add_ids))

                
             
            val_size = int(len(product_ids) * split_percentage)
            if val_size > 0:
                val_ids = np.random.choice(product_ids, val_size, replace=False)
            else:
                val_ids = []

            # Create a new row for each image.
            for product_id in product_ids:
                row = [product_id, category_idx]
                for img_idx in range(df.loc[product_id, "num_imgs"]):
                    if product_id in val_ids:
                        val_list.append(row + [img_idx])
                    else:
                        train_list.append(row + [img_idx])
                pbar.update()
                
    columns = ["product_id", "category_idx", "img_idx"]
    train_df = pd.DataFrame(train_list, columns=columns)
    val_df = pd.DataFrame(val_list, columns=columns)   
    return train_df, val_df

train_images_df, val_images_df = make_val_set(train_offsets_df,max_num = 6000,threshold = 200, split_percentage=0.1, 
                                              drop_percentage=0.5)

train_images_df.to_csv("train_images.csv")
val_images_df.to_csv("val_images.csv")
categories_df = pd.read_csv("categories.csv", index_col=0)


cat2idx, idx2cat = make_category_tables()
train_offsets_df = pd.read_csv("train_offsets.csv", index_col=0)
train_images_df = pd.read_csv("train_images.csv", index_col=0)
val_images_df = pd.read_csv("val_images.csv", index_col=0)



#It creates batches of images (and their one-hot encoded labels) directly from the BSON file. 
class BsonGenerator(Iterator):
    def __init__(self, bson_file, images_df, offsets_df, num_class,
                 image_data_generator, lock, target_size=(180, 180), 
                 with_labels=True, batch_size=32, shuffle=False, seed=None):
        self.images_df = images_df
        self.offsets_df = offsets_df
        self.with_labels = with_labels
        self.samples = len(images_df)
        self.num_class = num_class
        self.file = bson_file      
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.image_shape = self.target_size + (3,)
        Iterator.__init__(self.samples, batch_size, shuffle, seed)# initilize for Iterator
        self.lock = lock

    def _get_batches_of_transformed_samples(self, index_array):
       
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        if self.with_labels:
            batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())

        for i, j in enumerate(index_array):
            # Protect file and dataframe access with a lock.
            with self.lock:
                image_row = self.images_df.iloc[j]
                product_id = image_row["product_id"]
                offset_row = self.offsets_df.loc[product_id]

                # Read this product's data from the BSON file.
                self.file.seek(offset_row["offset"])
                item_data = self.file.read(offset_row["length"])

            # Grab the image from the product.
            item = bson.BSON.decode(item_data)
            img_idx = image_row["img_idx"]
            bson_img = item["imgs"][img_idx]["picture"]

            # Load the image.
            img = load_img(io.BytesIO(bson_img), target_size=self.target_size)

            # Preprocess the image.
            x = img_to_array(img)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)

            # Add the image and the label to the batch (one-hot encoded).
            batch_x[i] = x
            if self.with_labels:
                batch_y[i, image_row["category_idx"]] = 1

        if self.with_labels:
            return batch_x, batch_y
        else:
            return batch_x

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array[0])

    
    
    
train_bson_file = open(train_path, "rb")

lock = threading.Lock()

num_train_images = len(train_images_df)
num_val_images = len(val_images_df)
batch_size = 128
# Tip: use ImageDataGenerator for data augmentation and preprocessing.
train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.xception.preprocess_input,
                                   shear_range=0.2,
                                   zoom_range=0.1,
                                   rotation_range=180.,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)

train_gen = BsonGenerator(train_bson_file, train_images_df, train_offsets_df, 
                         num_classes, train_datagen, lock,
                         batch_size=batch_size, shuffle=True)


val_datagen = ImageDataGenerator(preprocessing_function=keras.applications.xception.preprocess_input,
                                 shear_range=0.2,
                                zoom_range=0.1,
                                rotation_range=180.,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                horizontal_flip=True)


val_gen = BsonGenerator(train_bson_file, val_images_df, train_offsets_df,
                       num_classes, val_datagen, lock,
                       batch_size=batch_size, shuffle=True)


last_set_layer = 96  # value is based on based model selected.
models_savename = "./models/xception"
classnames  = pd.read_csv('your/path/...')


img_width = 180
img_height = 180


'''Xception Model'''

model0 = Xception(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(img_width, img_height, 3))
x = model0.output
x = GlobalAveragePooling2D(name='avg_pool')(x)

x = Dropout(0.2)(x)

x = Dense(len(classnames), activation='softmax', name='predictions')(x)
model = Model(model0.input, x) 

#transfer learning


#first train the top added layer weights
for layer in model0.layers:
    layer.trainable = False



top_weights_path = os.path.join(output_path, 'top_model_weights_xception.h5')
callbacks= [ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True)]

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['acc'])

model.fit_generator(generator=train_gen,
                    steps_per_epoch=num_train_images//batch_size,
                    callbacks=callbacks,
                    validation_data=val_gen,
                    epochs=20,
                    validation_steps=num_val_images//batch_size)



# then train the layers after the last_set layer
model.load_weights(top_weights_path)


for layer in model.layers[:last_set_layer]:
        layer.trainable = False
for layer in model.layers[last_set_layer:]:
        layer.trainable = True



model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00025), 
              metrics=['acc'])
model.fit_generator(generator=train_gen,
                    steps_per_epoch=num_train_images//batch_size,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=val_gen,
                    epochs=30,
                    validation_steps=num_val_images//batch_size)



#non-transfer

#train the whole model
for layer in model.layers:
        layer.trainable = True
# Note you need to recompile the whole thing. Otherwise you are not traing first layers    
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00025), 
              metrics=['acc'])



#model.fit_generator(generator=train_gen,
#                    steps_per_epoch=100,
#                    verbose=1,
#                    callbacks=callbacks,
#                    validation_data=val_gen,
#                    epochs=30,
#                    validation_steps=40)
#
#


#Here we use decreasing learning rate since it is proved recently to have better results

init_epochs = 0


for i in range(10):
    # gradually decrease the learning rate
    K.set_value(model.optimizer.lr, 0.95 * K.get_value(model.optimizer.lr))
    start_epoch = (i * 2)
    epochs = ((i + 1) * 2)    
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=num_train_images//batch_size,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        initial_epoch=start_epoch + init_epochs,
                        epochs=epochs + init_epochs,
                        validation_steps=num_val_images//batch_size)
model.save('xception.h5')



'''InceptionV3_Model(The same as Xception model setting)'''


#here we have to change the generator's preprocessing part to inception_v3's, as remembered it is the same as Xception preprocessing
#function, but it is good habit to change it each time when using a new model(don't forget!)

train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.inception_v3.preprocess_input,
                                   shear_range=0.2,
                                   zoom_range=0.1,
                                   rotation_range=180.,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)

train_gen = BsonGenerator(train_bson_file, train_images_df, train_offsets_df, 
                         num_classes, train_datagen, lock,
                         batch_size=batch_size, shuffle=True)


val_datagen = ImageDataGenerator(preprocessing_function=keras.applications.inception_v3.preprocess_input,
                                 shear_range=0.2,
                                zoom_range=0.1,
                                rotation_range=180.,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                horizontal_flip=True)


val_gen = BsonGenerator(train_bson_file, val_images_df, train_offsets_df,
                       num_classes, val_datagen, lock,
                       batch_size=batch_size, shuffle=True)



model1 = InceptionV3(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(img_width, img_height, 3))
x = model1.output
x = GlobalAveragePooling2D(name='avg_pool')(x)

x = Dropout(0.2)(x)

x = Dense(len(classnames), activation='softmax', name='predictions')(x)
model = Model(model1.input, x) 

#transfer learning

#first train the top added layer weights
for layer in model0.layers:
    layer.trainable = False



top_weights_path = os.path.join(output_path, 'top_model_weights_inception.h5')
callbacks= [ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True)]

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['acc'])

model.fit_generator(generator=train_gen,
                    steps_per_epoch = num_train_images//batch_size,
                    callbacks = callbacks,
                    validation_data = val_gen,
                    epochs = 20,
                    validation_steps=num_val_images//batch_size)



# then train the layers after the last_set layer
model.load_weights(top_weights_path)


for layer in model.layers[:last_set_layer]:
        layer.trainable = False
for layer in model.layers[last_set_layer:]:
        layer.trainable = True



model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00025), 
              metrics=['acc'])

model.fit_generator(generator=train_gen,
                    steps_per_epoch=num_train_images//batch_size,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=val_gen,
                    epochs=30,
                    validation_steps=num_val_images//batch_size)



#non-transfer

#train the whole model
for layer in model.layers:
        layer.trainable = True
# Note you need to recompile the whole thing. Otherwise you are not traing first layers    
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00025), 
              metrics=['acc'])


init_epochs = 0


for i in range(10):
    # gradually decrease the learning rate
    K.set_value(model.optimizer.lr, 0.95 * K.get_value(model.optimizer.lr))
    start_epoch = (i * 2)
    epochs = ((i + 1) * 2)    
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=num_train_images//batch_size,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=val_gen,
                        initial_epoch=start_epoch + init_epochs,
                        epochs=epochs + init_epochs,
                        validation_steps=num_val_images//batch_size)
    
model.save('InceptionV3.h5')




'''Resnet50'''

from keras.applications.resnet50 import ResNet50   



train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input,
                                   shear_range=0.2,
                                   zoom_range=0.1,
                                   rotation_range=180.,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)

train_gen = BsonGenerator(train_bson_file, train_images_df, train_offsets_df, 
                         num_classes, train_datagen, lock,
                         batch_size=batch_size, shuffle=True)


val_datagen = ImageDataGenerator(preprocessing_function=keras.applications.resnet50.preprocess_input,
                                 shear_range=0.2,
                                zoom_range=0.1,
                                rotation_range=180.,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                horizontal_flip=True)


val_gen = BsonGenerator(train_bson_file, val_images_df, train_offsets_df,
                       num_classes, val_datagen, lock,
                       batch_size=batch_size, shuffle=True)






model2 = ResNet50(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(img_width, img_height, 3))
x = model2.output
x = GlobalAveragePooling2D(name='avg_pool')(x)

x = Dropout(0.2)(x)

x = Dense(len(classnames), activation='softmax', name='predictions')(x)
model = Model(model2.input, x) 

#transfer learning


for i, layer in enumerate(model2.layers):
    print(i,layer.name)

last_set_layer = 103

#first train the top added layer weights
for layer in model2.layers:
    layer.trainable = False



top_weights_path_resnet50 = os.path.join(output_path, 'top_model_weights_resnet50.h5')
callbacks= [ModelCheckpoint(top_weights_path_resnet50, monitor='val_acc', verbose=1, save_best_only=True)]

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['acc'])

model.fit_generator(generator=train_gen,
                    steps_per_epoch=num_train_images//batch_size,
                    callbacks=callbacks,
                    validation_data=val_gen,
                    epochs=20,
                    validation_steps=num_val_images//batch_size)



# then train the layers after the last_set layer
model.load_weights(top_weights_path_resnet50)


for layer in model.layers[:last_set_layer]:
        layer.trainable = False
for layer in model.layers[last_set_layer:]:
        layer.trainable = True


# here we use a very small learning rate to fine tuning the weights
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00025), 
              metrics=['acc'])


model.fit_generator(generator=train_gen,
                    steps_per_epoch=num_train_images//batch_size,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=val_gen,
                    epochs=30,
                    validation_steps=num_val_images//batch_size)









#here just use one example of xception to illustrate the prediction part 
model = load_model("/Users/tanzhenyu/Downloads/xception.hdf5")

submission_df = pd.read_csv(output_path + "sample_submission.csv")
submission_df.head()

test_datagen = ImageDataGenerator(preprocessing_function=keras.applications.xception.preprocess_input)
data = bson.decode_file_iter(open(test_path, "rb"))


#mean methos
with tqdm(total=num_test_products) as pbar:
    for c, d in enumerate(data):
        product_id = d["_id"]
        num_imgs = len(d["imgs"])

        batch_x = np.zeros((num_imgs, 180, 180, 3), dtype=K.floatx())

        for i in range(num_imgs):
            bson_img = d["imgs"][i]["picture"]

            # Load and preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=(180, 180))
            x = img_to_array(img)
            x = test_datagen.random_transform(x)
            x = test_datagen.standardize(x)

            # Add the image to the batch.
            batch_x[i] = x

        prediction = model.predict(batch_x, batch_size=num_imgs)
        
        avg_pred = prediction.mean(axis=0)
        cat_idx = np.argmax(avg_pred)
        submission_df.iloc[c]["category_id"] = idx2cat[cat_idx]
        pbar.update()



#voting methods
with tqdm(total=num_test_products) as pbar:
    for c, d in enumerate(data):
        product_id = d["_id"]
        num_imgs = len(d["imgs"])

        batch_x = np.zeros((num_imgs, 180, 180, 3), dtype=K.floatx())

        for i in range(num_imgs):
            bson_img = d["imgs"][i]["picture"]

            # Load and preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=(180, 180))
            x = img_to_array(img)
            x = test_datagen.random_transform(x)
            x = test_datagen.standardize(x)

            # Add the image to the batch.
            batch_x[i] = x

        prediction = model.predict(batch_x, batch_size=num_imgs)
        cat_idx = np.argmax(prediction.mean(axis=0))
        if num_imgs in [3,4]:
            prediction_arg = np.argmax(prediction, axis=1)
            for key, value in Counter(prediction_arg).items():
                if value >=2:
                    cat_idx = key 
        submission_df.iloc[c]["category_id"] = idx2cat[cat_idx]
        pbar.update()


submission_df.to_csv("my_submission.csv.gz", compression="gzip", index=False)

    
    
    
    
