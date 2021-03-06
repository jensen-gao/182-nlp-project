from transformers import TFDistilBertPreTrainedModel, TFDistilBertMainLayer
from coral import CORAL
from transformers.modeling_tf_utils import get_initializer
import tensorflow as tf


class TFDistilBertForOrdinalRegression(TFDistilBertPreTrainedModel):
    def __init__(self, config, as_features=False, use_layer_norm=True, *inputs, **kwargs):
        super(TFDistilBertForOrdinalRegression, self).__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.use_layer_norm = use_layer_norm

        self.distilbert = TFDistilBertMainLayer(config, name="distilbert", trainable=not as_features)
        self.pre_classifier = tf.keras.layers.Dense(
            config.dim,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="relu",
            name="pre_classifier",
        )
        self.classifier = CORAL(config.num_labels, kernel_initializer=get_initializer(config.initializer_range),
                                name="classifier")
        self.dropout = tf.keras.layers.Dropout(config.seq_classif_dropout)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, **kwargs):
        distilbert_output = self.distilbert(inputs, **kwargs)

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        if self.use_layer_norm:
            pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output, training=kwargs.get("training", False))  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)
        return logits
