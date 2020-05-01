import tensorflow as tf
import transformers
import pickle
import os
from tf_distilbert_for_ordinal_regression import TFDistilBertForOrdinalRegression
from data_utils import load_data
from metrics import *


config = transformers.DistilBertConfig.from_pretrained('models/final/', num_labels=4)
strategy = tf.distribute.MirroredStrategy()
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

with strategy.scope():
    model = TFDistilBertForOrdinalRegression.from_pretrained('models/final/', config=config)
    model.compile(optimizer='adam', loss=loss, metrics=[pred_accuracy, pred_abs_error])

batch_size_per_replica = 16
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

test_dataset = load_data(split='test', batch_size=batch_size)
original_test_dataset = load_data(split='original_test', batch_size=batch_size)

test_results = model.evaluate(test_dataset)
original_test_results = model.evaluate(original_test_dataset)
results = {'test_results': test_results, 'original_test_results': original_test_results}

with open(os.path.join('test_results', 'results.pickle'), 'wb') as f:
    pickle.dump(results, f)
