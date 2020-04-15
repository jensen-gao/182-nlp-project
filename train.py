import os
import tensorflow as tf
from data_utils import load_data
import transformers
from tf_distilbert_for_ordinal_regression import TFDistilBertForOrdinalClassification

config = transformers.DistilBertConfig.from_pretrained('pretrained/', num_labels=4)
model = TFDistilBertForOrdinalClassification.from_pretrained('pretrained/', config=config, from_pt=True)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)

epochs = 1
batch_size = 16

train_dataset = load_data(split='train', batch_size=batch_size, weighted=True)
valid_dataset = load_data(split='valid', batch_size=batch_size)

num_training_steps = int(tf.data.experimental.cardinality(train_dataset)) * epochs
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

model.fit(train_dataset, epochs=epochs, callbacks=[checkpoint_callback, lr_callback], validation_data=valid_dataset)
model.save_pretrained('models')
