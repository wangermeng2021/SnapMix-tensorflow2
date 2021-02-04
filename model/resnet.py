
import tensorflow as tf
from utils.layers import concat_max_average_pool,rename_layer
from utils.model_preprocess import normalize_input
import cv2
class ResNet():
    def __init__(self, depth=50, classes=10, concat_max_and_average_pool=False,pretrain='imagenet'):
        self.type = depth
        self.classes = classes
        self.concat_max_and_averal_pool = concat_max_and_average_pool
        self.pretrain = pretrain
        self.keras_model = self.create_model()
    def preprocess(self, x, size=None, normalization_mode='caffe'):
        if size != None:
            x = tf.image.resize(x, size)
        return normalize_input(x, mode=normalization_mode)
    def create_model(self):
        input_layer = tf.keras.Input(shape=(None, None, 3))
        if self.type == 50:
            x = tf.keras.applications.ResNet50(include_top=False,weights=self.pretrain)(input_layer)
        elif self.type == 101:
            x = tf.keras.applications.ResNet101(include_top=False,weights=self.pretrain)(input_layer)

        if self.concat_max_and_averal_pool:
            x = concat_max_average_pool(pool_size=(2, 2),name="last_conv")(x)
        else:
            x = rename_layer(name="last_conv")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.classes, activation='softmax',name="predictions")(x)

        return tf.keras.Model(inputs=input_layer, outputs=x)

