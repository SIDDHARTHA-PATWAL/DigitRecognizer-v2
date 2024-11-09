import tensorflow as tf
import tf2onnx


model = tf.keras.models.load_model('handwritten_model.h5')

onnx_model = tf2onnx.convert.from_keras(model)


onnx_model = onnx_model[0]  

import onnx
onnx.save_model(onnx_model, 'handwritten_model.onnx')
