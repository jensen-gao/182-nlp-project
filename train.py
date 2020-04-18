import os
import tensorflow as tf
from data_utils import load_data
import transformers
import pickle
from tf_distilbert_for_ordinal_regression import TFDistilBertForOrdinalClassification
from tensorflow.keras import backend as K


config = transformers.DistilBertConfig.from_pretrained('pretrained/', num_labels=4)
strategy = tf.distribute.MirroredStrategy()
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def pred_accuracy(y_true, y_pred, threshold=0.0):
    y_pred = K.sum(K.cast(y_pred > threshold, y_pred.dtype), axis=-1)
    y_true = K.sum(y_true, axis=-1)
    return K.mean(K.equal(y_true, y_pred))


def pred_abs_error(y_true, y_pred, threshold=0.0):
    y_pred = K.sum(K.cast(y_pred > threshold, y_pred.dtype), axis=-1)
    y_true = K.sum(y_true, axis=-1)
    return K.mean(K.abs(y_pred - y_true))


with strategy.scope():
    model = TFDistilBertForOrdinalClassification.from_pretrained('pretrained/', config=config, from_pt=True)
    model.compile(optimizer='adam', loss=loss, metrics=[pred_accuracy, pred_abs_error])

epochs = 4
batch_size_per_replica = 16
batch_size = batch_size_per_replica * strategy.num_replicas_in_sync

#train_dataset = load_data(split='train', batch_size=batch_size)
valid_dataset = load_data(split='valid', batch_size=batch_size)
breakpoint()

num_training_steps = int(tf.data.experimental.cardinality(train_dataset)) * epochs
num_validation_steps = int(tf.data.experimental.cardinality(valid_dataset))
num_warmup_steps = num_training_steps // 10
base_learning_rate = 2e-5


def learning_rate_schedule(batch, logs):
    if batch < num_warmup_steps:
        new_lr = base_learning_rate * float(batch + 1) / float(max(1, num_warmup_steps))
    else:
        new_lr = base_learning_rate * max(0.0, float(num_training_steps - batch) /
                                          float(max(1, num_training_steps - num_warmup_steps)))
    tf.keras.backend.set_value(model.optimizer.lr, new_lr)


lr_callback = tf.keras.callbacks.LambdaCallback(on_batch_begin=learning_rate_schedule)

checkpoint_dir = 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

#for i in range(1, 5):
#    with strategy.scope():
#        model.load_weights('checkpoints/ckpt_' + str(i))
#    metrics = model.evaluate(valid_dataset)
#    with open('eval_metrics_' + str(i) + '.pickle', 'wb') as f:
#        pickle.dump(metrics, f)
#    print(metrics)

history = model.fit(train_dataset, epochs=epochs, callbacks=[checkpoint_callback, lr_callback], validation_data=valid_dataset, validation_steps=num_validation_steps)
model.save_pretrained('models')
with open('train_history.pickle', 'wb') as f:
    pickle.dump(history.history, f)

