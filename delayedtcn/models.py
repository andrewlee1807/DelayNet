#  Copyright (c) 2022 Andrew
#  Email: andrewlee1807@gmail.com

import inspect

import tensorflow as tf
from tensorflow.keras import layers


class DelayedLayer(layers.Layer):
    """
    Delayed BlockLayer:
    ------- 1DConvolution
    ------- ReLU
    ------- Dropout
    ------- 1DConvolution
    ------- ReLU
    ------- Dropout
    ------- Skip Connection
    ------- ReLU
    """

    def __init__(self,
                 nb_stride=3,
                 nb_filters=64,
                 kernel_size=3,
                 padding='causal',
                 dropout_rate=0.0,
                 use_weight_norm=False,
                 use_skip_connections=False,
                 init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                 name="DelayedLayer", **kwargs):
        super(DelayedLayer, self).__init__(name=name, **kwargs)

        self.nb_filters = nb_filters
        self.use_weight_norm = use_weight_norm
        self.use_skip_connections = use_skip_connections
        self.layers = []

        for k in range(2):
            name = 'conv1D_{}'.format(k)
            if k == 0:
                conv = layers.Conv1D(filters=self.nb_filters,
                                     kernel_size=kernel_size,
                                     strides=nb_stride,
                                     padding=padding,
                                     name=name)
            else:
                conv = layers.Conv1D(filters=self.nb_filters,
                                     kernel_size=kernel_size,
                                     dilation_rate=1,
                                     padding=padding,
                                     name=name)

            self.layers.append(conv)

            if self.use_weight_norm:
                from tensorflow_addons.layers import WeightNormalization
                self.layers.append(WeightNormalization)
            self.layers.append(layers.Activation('relu', name='Act_Delay_Layer'))
            # self.layers.append(layers.Dropout(rate=dropout_rate))

        # 1x1 conv to match the shapes (channel dimension).
        # make and build this layer separately because it directly uses input_shape.
        self.shape_match_conv = layers.Conv1D(
            filters=self.nb_filters,
            kernel_size=1,
            padding='same',
            name='matching_conv1D'
        )
        #
        # self.batch1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation('relu', name='Act_Delay_Block')
        self.drop1 = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=None, **kwargs):
        x_org, x = inputs  # [original input, input after generating]
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)
        # skip connections
        if self.use_skip_connections:
            x = layers.add([self.shape_match_conv(x_org), x], name='Add_Res')
            x_out = self.ac1(x)
        else:
            x_out = x
        return [x, x_out]  # [skip_connection, output]


class StrideLayer(layers.Layer):
    """
    Dilated BlockLayer:
    ------- 1DConvolution: dilation_rate=2^0
    ------- ReLU
    ------- Dropout
    ------- 1DConvolution: dilation_rate=2^1
    ------- ReLU
    ------- Dropout
    ------- Skip Connection
    ------- ReLU
    """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=3,
                 dilation_rate=1,
                 padding='causal',
                 dropout_rate=0.0,
                 use_weight_norm=False,
                 init=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
                 name="DilatedLayer", **kwargs):
        super(StrideLayer, self).__init__(name=name, **kwargs)

        self.nb_filters = nb_filters
        self.dilation_rate = dilation_rate
        self.use_weight_norm = use_weight_norm
        self.layers = []

        for k in range(6):
            conv = layers.Conv1D(filters=self.nb_filters,
                                 kernel_size=kernel_size,
                                 dilation_rate=2 ** k,
                                 padding=padding,
                                 name=name)

            self.layers.append(conv)

            if self.use_weight_norm:
                from tensorflow_addons.layers import WeightNormalization
                self.layers.append(WeightNormalization)
            self.layers.append(layers.Activation('relu', name='Act_Dilated_Layer'))
            self.layers.append(layers.Dropout(rate=dropout_rate))

        # self.batch1 = layers.BatchNormalization(axis=-1)
        self.ac1 = layers.Activation('relu', name='Act_Stride_Block')
        self.drop1 = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        for layer in self.layers:
            training_flag = 'training' in dict(inspect.signature(layer.call).parameters)
            x = layer(x, training=training) if training_flag else layer(x)

        inputs_x = layers.add([inputs, x], name='Add_Res')
        x_out = self.ac1(inputs_x)
        return [x, x_out]  # [skip_connection, output]


# Idea 1: Preprocessing input, stride connection apply then
class Model1(tf.keras.Model):
    def __init__(self,
                 list_stride=(7, 1),
                 list_dilation=(1, 1),
                 nb_filters=64,
                 kernel_size=6,
                 nb_stacks=1,
                 padding='causal',
                 target_size=24,
                 use_skip_connections=True,
                 dropout_rate=0.0):
        self.nb_filters = nb_filters
        self.list_stride = list_stride
        self.list_dilation = list_dilation
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_skip_connections = use_skip_connections
        self.nb_stacks = nb_stacks

        super(Model1, self).__init__()
        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        assert padding in ['causal', 'same']

        # base layer
        self.delayed_block = DelayedLayer(nb_stride=self.list_stride[0],
                                          nb_filters=self.nb_filters,
                                          kernel_size=self.kernel_size,
                                          padding=self.padding,
                                          init=init,
                                          use_skip_connections=self.use_skip_connections,
                                          dropout_rate=dropout_rate,
                                          name="DelayedLayer")

        self.skip_connections = None

        self.dilated_blocks = []

        # res layer
        for i in range(self.nb_stacks - 1):
            self.dilated_blocks.append(
                StrideLayer(nb_filters=self.nb_filters,
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            init=init,
                            dropout_rate=dropout_rate,
                            name=f"DilatedLayer_{i}")
            )

        self.slicer_layer = layers.Lambda(lambda tt: tt[:, -1, :], name='Slice_Output')
        self.final_ac = layers.Activation('relu', name='Act_Final')

        self.dense1 = layers.Dense(units=128)
        self.dense = layers.Dense(units=target_size)

    def call(self, inputs, training=True):
        x_org, x = inputs  # [original input, input after preprocessing]
        self.skip_connections = []

        x_res, x = self.delayed_block([x_org, x])  # x_res, x_out
        self.skip_connections.append(x_res)

        for dilated_block in self.dilated_blocks:
            x_res, x = dilated_block(x)  # x_res, x_out
            self.skip_connections.append(x_res)

        if self.use_skip_connections and len(self.dilated_blocks) > 0:
            x = layers.add(self.skip_connections, name='Add_Skip_Connections')
            x = self.final_ac(x)

        x = self.slicer_layer(x)
        x = self.dense1(x)
        x = self.dense(x)
        # x = self.reshape(x)
        return x

    def summary(self, x):
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name="DelayNet1")
        return model.summary()