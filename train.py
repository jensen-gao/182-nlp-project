import os
import tensorflow as tf
from data_utils import load_data
import transformers
import pickle
import argparse
from tf_distilbert_for_ordinal_regression import TFDistilBertForOrdinalRegression
from tf_distilbert_for_classification import TFDistilBertForClassification
from metrics import *
from math import ceil


parser = argparse.ArgumentParser()
parser.add_argument('--version', '-v', type=str, default='final',
                    help='Version of model, used for naming the saved files.')
parser.add_argument('--pretrain', '-p', action='store_true', help='Whether to use the model pretrained on the corpus')
parser.add_argument('--ordinal', '-o', action='store_true',
                    help='Whether to use ordinal regression instead of classification.')
parser.add_argument('--as_features', '-f', action='store_false',
                    help='Whether to freeze the BERT layers and use them only as features instead of fine-tuning.')
args = parser.parse_args()

if args.pretrain:
    config = transformers.DistilBertConfig.from_pretrained('pretrained/', num_labels=4)
else:
    config = transformers.DistilBertConfig.from_pretrained('distilbert-base-uncased', num_labels=4)

strategy = tf.distribute.MirroredStrategy()
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

if args.ordinal:
    model_type = TFDistilBertForOrdinalRegression
else:
    model_type = TFDistilBertForClassification

with strategy.scope():
    if args.pretrain:
        model = model_type.from_pretrained('pretrained/', config=config, as_features=args.as_features, from_pt=True)
    else:
        model = model_type.from_pretrained('distilbert-base-uncased', config=config, as_features=args.as_features)
    model.compile(optimizer='adam', loss=loss, metrics=[pred_accuracy, pred_abs_error])

epochs = 4
batch_size_per_replica = 16
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

train_dataset = load_data(split='train', ordinal=args.ordinal, batch_size=batch_size)
valid_dataset = load_data(split='valid', ordinal=args.ordinal, batch_size=batch_size)

num_examples = ceil(747010 / batch_size)
num_training_steps = num_examples * epochs
num_validation_steps = ceil(106714 / batch_size)
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
                    validation_data=valid_dataset, validation_steps=num_validation_steps)

save_dir = os.path.join('models', args.version)
model.save_pretrained(save_dir)

train_history_dir = os.path.join('train_history', args.version)
with open(os.path.join(train_history_dir, 'train_history.pickle'), 'wb') as f:
    pickle.dump(history.history, f)

