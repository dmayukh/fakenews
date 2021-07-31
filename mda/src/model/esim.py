import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *

"""
    Keras ESIM, an implementation of Chen et. al. Enhanced LSTM for Natural Language Inference for the task of natural language inference.
    https://arxiv.org/abs/1609.06038
    Reference: https://github.com/dzdrav/kerasESIM
"""
class esim:
    def __init__(self, embedding_matrix, vocab_size = 8000, embedding_dim=300, alignment_dense_dim=300, final_dense_dim=100):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.alignment_dense_dim = alignment_dense_dim
        self.final_dense_dim = final_dense_dim
        self.embedding_matrix = embedding_matrix
        self.lstm_units = self.embedding_dim


    def build_model(self):

        inp1 = keras.Input(shape=(None,), name="hypothesis")
        inp2 = keras.Input(shape=(None,), name="evidence")

        embedding_hyp_layer = Embedding(
            input_dim=self.vocab_size + 1,
            output_dim=self.embedding_dim,
            weights=[self.embedding_matrix],
            trainable=False)
        embedding_evi_layer = Embedding(
            input_dim=self.vocab_size + 1,
            output_dim=self.embedding_dim,
            weights=[self.embedding_matrix],
            trainable=False)

        x_hyp = embedding_hyp_layer(inp1)
        x_hyp = tf.keras.layers.Dropout(0.5)(x_hyp)

        x_evi = embedding_evi_layer(inp2)
        x_evi = tf.keras.layers.Dropout(0.5)(x_evi)

        # Encoder = Bidirectional(CuDNNLSTM(units=300, return_sequences=True, kernel_initializer='RandomNormal'))

        lstm_layer1 = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(tf.keras.layers.LSTMCell(self.lstm_units), return_sequences=True))(
            x_hyp)
        lstm_layer2 = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(tf.keras.layers.LSTMCell(self.lstm_units), return_sequences=True))(
            x_evi)

        # lstm_layer1 = Encoder(x_hyp)
        # lstm_layer2 = Encoder(x_evi)

        F_p, F_h = lstm_layer1, lstm_layer2
        Eph = keras.layers.Dot(axes=(2, 2))([F_h, F_p])  # [batch_size, Hsize, Psize]
        Eh = Lambda(lambda x: keras.activations.softmax(x))(Eph)  # [batch_size, Hsize, Psize]
        Ep = keras.layers.Permute((2, 1))(Eph)  # [batch_size, Psize, Hsize)
        Ep = Lambda(lambda x: keras.activations.softmax(x))(Ep)  # [batch_size, Psize, Hsize]

        # 4, Normalize score matrix, encoder premises and get alignment
        PremAlign = keras.layers.Dot((2, 1))([Ep, lstm_layer2])  # [-1, Psize, dim]
        HypoAlign = keras.layers.Dot((2, 1))([Eh, lstm_layer1])  # [-1, Hsize, dim]
        mm_1 = keras.layers.Multiply()([lstm_layer1, PremAlign])
        mm_2 = keras.layers.Multiply()([lstm_layer2, HypoAlign])
        sb_1 = keras.layers.Subtract()([lstm_layer1, PremAlign])
        sb_2 = keras.layers.Subtract()([lstm_layer2, HypoAlign])

        # concat [a_, a~, a_ * a~, a_ - a~], isto za b_, b~
        PremAlign = keras.layers.Concatenate()([lstm_layer1, PremAlign, sb_1, mm_1, ])  # [batch_size, Psize, 2*unit]
        HypoAlign = keras.layers.Concatenate()([lstm_layer2, HypoAlign, sb_2, mm_2])  # [batch_size, Hsize, 2*unit]

        # inputs = tf.keras.Input(shape=(10, 128, 128, 3))
        # conv_2d_layer = tf.keras.layers.Conv2D(64, (3, 3))
        # outputs = tf.keras.layers.TimeDistributed(conv_2d_layer)(inputs)
        # outputs.shape

        # ff layer w/RELU activation
        Compresser = tf.keras.layers.TimeDistributed(Dense(self.alignment_dense_dim,
                                                           kernel_regularizer=l2(0.0),
                                                           bias_regularizer=l2(0.0),
                                                           activation='relu'),
                                                           name='Compresser')

        PremAlign = Compresser(PremAlign)
        HypoAlign = Compresser(HypoAlign)

        # 5, Final biLST < Encoder + Softmax Classifier
        # Decoder = Bidirectional(CuDNNLSTM(units=100, return_sequences=True, kernel_initializer='RandomNormal'),
        #                         name='finaldecoder')  # [-1,2*units]

        Decoder = tf.keras.layers.Bidirectional(tf.keras.layers.RNN(tf.keras.layers.LSTMCell(self.lstm_units), return_sequences=True),
                                                name='finaldecoder')

        PremAlign = Dropout(0.5)(PremAlign)
        HypoAlign = Dropout(0.5)(HypoAlign)
        final_p = Decoder(PremAlign)
        final_h = Decoder(HypoAlign)

        AveragePooling = tf.keras.layers.GlobalAveragePooling1D()
        MaxPooling = tf.keras.layers.GlobalMaxPooling1D()

        # AveragePooling = Lambda(lambda x: K.mean(x, axis=1)) # outs [-1, dim]
        # MaxPooling = Lambda(lambda x: K.max(x, axis=1)) # outs [-1, dim]
        avg_p = AveragePooling(final_p)
        avg_h = AveragePooling(final_h)
        max_p = MaxPooling(final_p)
        max_h = MaxPooling(final_h)
        # concat of avg and max pooling for hypothesis and premise
        Final = keras.layers.Concatenate()([avg_p, max_p, avg_h, max_h])
        # dropout layer
        Final = Dropout(0.5)(Final)
        # ff layer w/tanh activation
        Final = Dense(self.final_dense_dim,
                      kernel_regularizer=l2(0.0),
                      bias_regularizer=l2(0.0),
                      name='dense300_',
                      activation='tanh')(Final)

        # last dropout factor
        factor = 1
        # if self.LastDropoutHalf:
        #     factor = 2
        Final = Dropout(0.5 / factor)(Final)

        # softmax classifier
        Final = Dense(3,
                      activation='softmax',
                      name='judge300_')(Final)
        model = tf.keras.Model(inputs=[inp1, inp2], outputs=Final)

        LearningRate = 4e-4
        GradientClipping = 10.0

        # Optimizer = keras.optimizers.Adam(lr = LearningRate,
        #             clipnorm = GradientClipping)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model
