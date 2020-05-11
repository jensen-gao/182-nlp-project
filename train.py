import os
import tensorflow as tf
from data_utils import load_data
import transformers
import pickle
import argparse
from bert_models.tf_distilbert_for_ordinal_regression import TFDistilBertForOrdinalRegression
from bert_models.tf_distilbert_for_classification import TFDistilBertForClassification
from metrics import *
from math import ceil


parser = argparse.ArgumentParser()
parser.add_argument('--version', '-v', type=str, default='final',
                    help='Version of model, used for naming the saved files.')
parser.add_argument('--pretrain', '-p', action='store_true', help='Whether to use the model pretrained on the corpus')
parser.add_argument('--ordinal', '-o', action='store_true',
                    help='Whether to use ordinal regression instead of classification.')
parser.add_argument('--as_features', '-f', action='store_true',
                    help='Whether to freeze the BERT layers and use them only as features instead of fine-tuning.')
parser.add_argument('--loss_weights', '-lw', nargs='+', default=[1, 1, 1, 1, 1],
                    help='Loss weights for each possible star label')
parser.add_argument('--layer_norm', '-n', action='store_true',
                    help='Whether to use layer normalization before the classification output')
parser.add_argument('--batch_size', '-b', type=int, default=16,
                    help='Batch size to use for training')
args = parser.parse_args()

strategy = tf.distribute.MirroredStrategy()

def make_weighted_loss(loss_type, weights, ord=False):
    """
    Returns a weighted version of `loss_fn`.

    Parameters
    ==========
    loss_type : Class[tf.keras.losses.Loss]
        The class of the original loss function

    weights : list
        A list of weights, one for each possible label.

    ord : bool
        If true, assumes that the labels are ordinal labels.
    """
    weights = tf.constant(weights)
    loss_fn = loss_type(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def loss(y_true, y_pred):
        indices = tf.cast(tf.reduce_sum(y_true, axis=1) if ord else tf.constant(y_true), tf.int32)
        losses = loss_fn(y_true, y_pred)
        weighted_losses = tf.gather(weights, indices) * losses
        return tf.reduce_mean(weighted_losses)
    
    return loss

assert len(args.loss_weights) == 5, "Must have exactly 5 loss weights (one for each star rating)"
loss_weights = [float(i) for i in args.loss_weights]
weighted_loss = (loss_weights != [1, 1, 1, 1, 1])

SparseCategoricalCrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
WeightedSparseCategoricalCrossentropy = make_weighted_loss(tf.keras.losses.SparseCategoricalCrossentropy, loss_weights, ord=False)

BinaryCrossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
WeightedBinaryCrossentropy = make_weighted_loss(tf.keras.losses.BinaryCrossentropy, loss_weights, ord=True)

if weighted_loss:
    print(f"Using loss weights {loss_weights}")

if args.ordinal:
    model_type = TFDistilBertForOrdinalRegression
    loss = WeightedBinaryCrossentropy if weighted_loss else BinaryCrossentropy
    num_labels = 4
    metrics = [ord_pred_accuracy, ord_pred_abs_error]
else:
    model_type = TFDistilBertForClassification
    loss = WeightedSparseCategoricalCrossentropy if weighted_loss else SparseCategoricalCrossentropy
    num_labels = 5
    metrics = ['accuracy', pred_abs_error]

if args.pretrain:
    config = transformers.DistilBertConfig.from_pretrained('pretrained/', num_labels=num_labels)
else:
    config = transformers.DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=num_labels)

with strategy.scope():
    if args.pretrain:
        model = model_type.from_pretrained('pretrained/', config=config, as_features=args.as_features, use_layer_norm=args.layer_norm, from_pt=True)
    else:
        model = model_type.from_pretrained('distilbert-base-uncased', config=config, as_features=args.as_features, use_layer_norm=args.layer_norm)
    model.compile(optimizer='adam', loss=loss, metrics=metrics)


epochs = 4
batch_size_per_replica = args.batch_size
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

train_dataset = load_data(split='train', ordinal=args.ordinal, batch_size=batch_size)
valid_dataset = load_data(split='valid', ordinal=args.ordinal, batch_size=batch_size)

num_examples = ceil(747010 / batch_size)
num_training_steps = num_examples * epochs
num_warmup_steps = num_training_steps // 10
base_learning_rate = 2e-5
current_epoch = 0


def learning_rate_schedule(batch, logs):
    step = batch + (current_epoch * num_examples)
    if step < num_warmup_steps:
        new_lr = base_learning_rate * float(step + 1) / float(max(1, num_warmup_steps))
    else:
        new_lr = base_learning_rate * max(0.0, float(num_training_steps - step) /
                                          float(max(1, num_training_steps - num_warmup_steps)))
    tf.keras.backend.set_value(model.optimizer.lr, new_lr)


lr_callback = tf.keras.callbacks.LambdaCallback(on_batch_begin=learning_rate_schedule)


def record_epoch(epoch, logs):
    global current_epoch
    current_epoch = epoch


epoch_callback = tf.keras.callbacks.LambdaCallback(on_epoch_begin=record_epoch)

checkpoint_dir = os.path.join('checkpoints', args.version)
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

history = model.fit(train_dataset, epochs=epochs, callbacks=[epoch_callback, checkpoint_callback, lr_callback],
                    validation_data=valid_dataset)

save_dir = os.path.join('models', args.version)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
model.save_pretrained(save_dir)

train_history_dir = os.path.join('train_history', args.version)
if not os.path.isdir(train_history_dir):
    os.mkdir(train_history_dir)
with open(os.path.join(train_history_dir, 'train_history.pickle'), 'wb') as f:
    pickle.dump(history.history, f)

