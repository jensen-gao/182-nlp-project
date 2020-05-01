import os
import tensorflow as tf
from data_utils import load_data
import transformers
import pickle
from tf_distilbert_for_ordinal_regression import TFDistilBertForOrdinalRegression
from metrics import *
from math import ceil


version = 'final'
config = transformers.DistilBertConfig.from_pretrained('pretrained/', num_labels=4)
strategy = tf.distribute.MirroredStrategy()
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

with strategy.scope():
    model = TFDistilBertForOrdinalRegression.from_pretrained('pretrained/', config=config, from_pt=True)
    model.compile(optimizer='adam', loss=loss, metrics=[pred_accuracy, pred_abs_error])

epochs = 4
batch_size_per_replica = 16
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

train_dataset = load_data(split='train', batch_size=batch_size)
valid_dataset = load_data(split='valid', batch_size=batch_size)

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

checkpoint_dir = os.path.join('checkpoints', version)
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

history = model.fit(train_dataset, epochs=epochs, callbacks=[epoch_callback, checkpoint_callback, lr_callback],
                    validation_data=valid_dataset, validation_steps=num_validation_steps)

save_dir = os.path.join('models', version)
model.save_pretrained(save_dir)

train_history_dir = os.path.join('train_history', version)
with open(os.path.join(train_history_dir, 'train_history.pickle'), 'wb') as f:
    pickle.dump(history.history, f)

