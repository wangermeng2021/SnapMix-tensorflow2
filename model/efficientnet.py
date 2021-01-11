
import tensorflow as tf
from utils.layers import concat_max_average_pool,rename_layer
from utils.model_preprocess import normalize_input
import cv2
class EfficientNet():
    def __init__(self, type='B0', classes=10, concat_max_and_averal_pool=False,pretrain='imagenet'):
        self.type = type
        self.classes = classes
        self.concat_max_and_averal_pool = concat_max_and_averal_pool
        self.efficientnet_input_size = {'B0':224, 'B1':240, 'B2':260, 'B3':300, 'B4':380, 'B5':456,'B6':528, 'B7':600}
        self.pretrain = pretrain

    def preprocess(self, x, size=None, normalization_mode=None):
        if size == None:
            x = cv2.resize(x, (self.efficientnet_input_size[self.type],self.efficientnet_input_size[self.type]),interpolation=cv2.INTER_LINEAR)
        return normalize_input(x, mode=normalization_mode)
    def get_model(self):
        input_layer = tf.keras.Input(shape=(None, None, 3))
        if self.type == 'B0':
            x = tf.keras.applications.EfficientNetB0(include_top=False, weights=self.pretrain)(input_layer)
        elif self.depth == 'B1':
            x = tf.keras.applications.EfficientNetB1(include_top=False, weights=self.pretrain)(input_layer)
        elif self.depth == 'B2':
            x = tf.keras.applications.EfficientNetB2(include_top=False, weights=self.pretrain)(input_layer)
        elif self.depth == 'B3':
            x = tf.keras.applications.EfficientNetB3(include_top=False, weights=self.pretrain)(input_layer)
        elif self.depth == 'B4':
            x = tf.keras.applications.EfficientNetB4(include_top=False, weights=self.pretrain)(input_layer)
        elif self.depth == 'B5':
            x = tf.keras.applications.EfficientNetB5(include_top=False, weights=self.pretrain)(input_layer)
        elif self.depth == 'B6':
            x = tf.keras.applications.EfficientNetB6(include_top=False, weights=self.pretrain)(input_layer)
        elif self.depth == 'B7':
            x = tf.keras.applications.EfficientNetB7(include_top=False, weights=self.pretrain)(input_layer)

        if self.concat_max_and_averal_pool:
            x = concat_max_average_pool(pool_size=(2, 2),name="last_conv")(x)
        else:
            x = rename_layer(name="last_conv")(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(self.classes, activation='softmax',name="predictions")(x)

        return tf.keras.Model(inputs=input_layer, outputs=x)



