import os
import tensorflow as tf
import transformers
import argparse
from bert_models.tf_distilbert_for_ordinal_regression import TFDistilBertForOrdinalRegression
from bert_models.tf_distilbert_for_classification import TFDistilBertForClassification
from metrics import *


parser = argparse.ArgumentParser()
parser.add_argument('--version', '-v', type=str, default='final',
                    help='Version of model, used for naming the saved files.')
parser.add_argument('--pretrain', '-p', action='store_true', help='Whether to use the model pretrained on the corpus')
parser.add_argument('--ordinal', '-o', action='store_true',
                    help='Whether to use ordinal regression instead of classification.')
parser.add_argument('--as_features', '-f', action='store_true',
                    help='Whether to freeze the BERT layers and use them only as features instead of fine-tuning.')
parser.add_argument('--ckpt_path', '-c', type=str, default='final',
                    help='Path to checkpoin files.')
args = parser.parse_args()


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

if args.pretrain:
    config = transformers.DistilBertConfig.from_pretrained('pretrained/', num_labels=num_labels)
else:
    config = transformers.DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

if args.pretrain:
    model = model_type.from_pretrained('pretrained/', config=config, as_features=args.as_features, from_pt=True)
else:
    model = model_type.from_pretrained('distilbert-base-uncased', config=config, as_features=args.as_features)

model.compile(optimizer='adam', loss=loss, metrics=metrics)
model.load_weights(args.ckpt_path)

save_dir = os.path.join('models', args.version)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
model.save_pretrained(save_dir)
