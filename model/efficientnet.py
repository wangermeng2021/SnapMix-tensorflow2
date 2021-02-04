
import tensorflow as tf
from utils.layers import concat_max_average_pool,rename_layer
from utils.model_preprocess import normalize_input
import cv2
class EfficientNet():
    def __init__(self, type='B0', classes=10, concat_max_and_average_pool=False,pretrain='imagenet'):
        self.type = type
        self.classes = classes
        self.concat_max_and_average_pool = concat_max_and_average_pool
        self.efficientnet_input_size = {'B0':224, 'B1':240, 'B2':260, 'B3':300, 'B4':380, 'B5':456,'B6':528, 'B7':600}
        self.pretrain = pretrain
        self.keras_model = self.create_model()
    def preprocess(self, x, size=None, normalization_mode=None):
        if size == None:
            x = tf.image.resize(x,(self.efficientnet_input_size[self.type],self.efficientnet_input_size[self.type]))
        return normalize_input(x, mode=normalization_mode)
    def create_model(self):
        input_layer = tf.keras.Input(shape=(None, None, 3))
        if self.type == 'B0':
            x = tf.keras.applications.EfficientNetB0(include_top=False, weights=self.pretrain)(input_layer)
        elif self.type == 'B1':
            x = tf.keras.applications.EfficientNetB1(include_top=False, weights=self.pretrain)(input_layer)
        elif self.type == 'B2':
            x = tf.keras.applications.EfficientNetB2(include_top=False, weights=self.pretrain)(input_layer)
        elif self.type == 'B3':
            x = tf.keras.applications.EfficientNetB3(include_top=False, weights=self.pretrain)(input_layer)
        elif self.type == 'B4':
            x = tf.keras.applications.EfficientNetB4(include_top=False, weights=self.pretrain)(input_layer)
        elif self.type == 'B5':
            x = tf.keras.applications.EfficientNetB5(include_top=False, weights=self.pretrain)(input_layer)
        elif self.type == 'B6':
            x = tf.keras.applications.EfficientNetB6(include_top=False, weights=self.pretrain)(input_layer)
        elif self.type == 'B7':
            x = tf.keras.applications.EfficientNetB7(include_top=False, weights=self.pretrain)(input_layer)

        if self.concat_max_and_average_pool:
            x = concat_max_average_pool(pool_size=(2, 2),name="last_conv")(x)
        else:
            x = rename_layer(name="last_conv")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.classes, activation='softmax',name="predictions")(x)

        return tf.keras.Model(inputs=input_layer, outputs=x)



