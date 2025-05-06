import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

image_path = 'C:/Screen_Detection/Mobile_App/Moire/data/'

data = DataLoader.from_folder(image_path)
train_data, test_data = data.split(0.80)


model = image_classifier.create(train_data, model_spec=model_spec.get('efficientnet_lite0'), validation_data=test_data, epochs=50)

#loss, accuracy = model.evaluate(test_data)
history = model.evaluate(test_data)
#print(history)
#confusion_matrix = model.confusion_matrix(validation_data)
#print(confusion_matrix)
#model.export(export_dir='tf_model/')

#classes=model.index_to_label
#ds = test_data.gen_dataset()

#test_labels=[label for i, (image, label) in enumerate(ds.take(len(ds)))]
#test_labels=[labs[0].numpy() for labs in test_labels]

#test_pred=model.predict_top_k(ds)
#test_pred=[classes.index(pred[0][0]) for pred in test_pred]
#cm = tf.math.confusion_matrix(test_labels, test_pred, num_classes=2)
#print(cm)