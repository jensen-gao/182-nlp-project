import tensorflow as tf
import transformers
import pickle
import os
import argparse
from bert_models.tf_distilbert_for_ordinal_regression import TFDistilBertForOrdinalRegression
from bert_models.tf_distilbert_for_classification import TFDistilBertForClassification
from data_utils import load_data
from metrics import *

parser = argparse.ArgumentParser()
parser.add_argument('--version', '-v', type=str, default='final',
                    help='Version of model to load.')
parser.add_argument('--pretrain', '-p', action='store_true', help='Whether to use the model pretrained on the corpus')
parser.add_argument('--ordinal', '-o', action='store_true',
                    help='Whether to use ordinal regression instead of classification.')
parser.add_argument('--as_features', '-f', action='store_false',
                    help='Whether to freeze the BERT layers and use them only as features instead of fine-tuning.')
args = parser.parse_args()

model_path = os.path.join('models', args.version)
strategy = tf.distribute.MirroredStrategy()

if args.ordinal:
    model_type = TFDistilBertForOrdinalRegression
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    num_labels = 4
    metrics = [ord_pred_accuracy, ord_pred_abs_error]

else:
    model_type = TFDistilBertForClassification
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    num_labels = 5
    metrics = ['accuracy', pred_abs_error]

config = transformers.DistilBertConfig.from_pretrained(model_path, num_labels=num_labels)

with strategy.scope():
    model = model_type.from_pretrained(model_path, config=config)
    model.compile(optimizer='adam', loss=loss, metrics=metrics)

batch_size_per_replica = 16
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

original_test_dataset = load_data(split='original_test', ordinal=args.ordinal, batch_size=batch_size)
perturbed_test_dataset = load_data(split='perturbed_test', ordinal=args.ordinal, batch_size=batch_size)

original_test_results = model.evaluate(original_test_dataset)
perturbed_test_results = model.evaluate(perturbed_test_dataset)
results = {'original_test_results': original_test_results, 'perturbed_test_results': perturbed_test_results}

with open(os.path.join('test_results', 'results.pickle'), 'wb') as f:
    pickle.dump(results, f)
