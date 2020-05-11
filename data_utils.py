import json_lines
import os
import math
import tensorflow as tf
import pickle
from transformers import DistilBertTokenizer
import random

MAX_LENGTH = 384


def split_data(load_path='data', save_path='data/split_data'):
    """
    Shuffles and splits the data in `load_path` into training, validation, and test sets.
    """
    # Original dataset of review text
    with open(os.path.join(load_path, 'yelp.txt'), 'r') as f:
        text = f.readlines()
    # Same reviews, but translated into french and back
    with open(os.path.join(load_path, 'paraphrased_yelp.txt'), 'r') as f:
        paraphrased_text = f.readlines()
    # Labels (star counts)
    with open(os.path.join(load_path, 'stars.pickle'), 'rb') as f:
        stars = pickle.load(f)
    assert len(text) == len(paraphrased_text) == len(stars)

    n_data = len(text)
    random.seed(123456)

    # Shuffle the training data and labels
    combined = list(zip(text, paraphrased_text, stars))
    random.shuffle(combined)
    text, paraphrased_text, stars = zip(*combined)

    # Split into training (70%), validation (10%), and test (20%) sets
    train_len = math.floor(0.7 * n_data)
    valid_len = math.floor(0.1 * n_data)
    train_text = text[:train_len] + paraphrased_text[:train_len]
    valid_text = text[train_len: train_len + valid_len] + paraphrased_text[train_len: train_len + valid_len]
    original_test_text = text[train_len + valid_len:]
    perturbed_test_text = paraphrased_text[train_len + valid_len:]

    # Write train/validation/test data into txt files
    with open(os.path.join(save_path, 'train_text.txt'), 'w') as f:
        f.writelines(train_text)
    with open(os.path.join(save_path, 'valid_text.txt'), 'w') as f:
        f.writelines(valid_text)
    with open(os.path.join(save_path, 'original_test_text.txt'), 'w') as f:
        f.writelines(original_test_text)
    with open(os.path.join(save_path, 'perturbed_test_text.txt'), 'w') as f:
        f.writelines(perturbed_test_text)

    # Save the star counts to a pickle file
    train_stars = 2 * stars[:train_len] # labels repeated once for the paraphrased reviews
    valid_stars = 2 * stars[train_len: train_len + valid_len]
    test_stars = stars[train_len + valid_len:]
    pickle.dump(train_stars, open(os.path.join(save_path, 'train_stars.pickle'), 'wb'))
    pickle.dump(valid_stars, open(os.path.join(save_path, 'valid_stars.pickle'), 'wb'))
    pickle.dump(test_stars, open(os.path.join(save_path, 'test_stars.pickle'), 'wb'))


def process_data(load_path='data/split_data', save_path='data/datasets'):
    """
    Preprocesses the training/validation/test data from text files in `load_path`, then saves serialized versions as 
    TFRecords in `save_path`.
    """
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    with open(os.path.join(load_path, 'train_text.txt'), 'r') as f:
        train_text = f.readlines()
    train_stars = pickle.load(open(os.path.join(load_path, 'train_stars.pickle'), 'rb'))
    assert len(train_text) == len(train_stars)
    _write_dataset(train_text, train_stars, tokenizer, 123, save_path, 'train.tfrecord')

    with open(os.path.join(load_path, 'valid_text.txt'), 'r') as f:
        valid_text = f.readlines()
    valid_stars = pickle.load(open(os.path.join(load_path, 'valid_stars.pickle'), 'rb'))
    assert len(valid_text) == len(valid_stars)
    _write_dataset(valid_text, valid_stars, tokenizer, 456, save_path, 'valid.tfrecord')

    test_stars = pickle.load(open(os.path.join(load_path, 'test_stars.pickle'), 'rb'))

    with open(os.path.join(load_path, 'original_test_text.txt'), 'r') as f:
        original_test_text = f.readlines()
    assert len(original_test_text) == len(test_stars)
    _write_dataset(original_test_text, test_stars, tokenizer, 789, save_path, 'original_test.tfrecord')

    with open(os.path.join(load_path, 'perturbed_test_text.txt'), 'r') as f:
        perturbed_test_text = f.readlines()
    assert len(perturbed_test_text) == len(test_stars)
    _write_dataset(perturbed_test_text, test_stars, tokenizer, 789, save_path, 'perturbed_test.tfrecord')


def _write_dataset(text, stars, tokenizer, seed, save_path, filename):
    """
    Encodes and stores the data in `text` and `stars` as TFRecord files. This will save to two files:
    1) `{save_path}/{filename}`     for data with normal classification labels
    2)`{save_path}/ord_{filename}` for data with ordinal regression labels
    """
    input_ids, attention_masks = _encode_data(text, tokenizer)

    # Normal classification labels (0-4)
    labels = [int(star - 1) for star in stars]

    # Labels for ordinal regression.
    # 1 star : [0, 0, 0, 0]
    # 2 stars: [1, 0, 0, 0]
    # 3 stars: [1, 1, 0, 0]
    # 4 stars: [1, 1, 1, 0]
    # 5 stars: [1, 1, 1, 1]
    ord_labels = [[1] * (int(star) - 1) + [0] * (4 - int(star) + 1) for star in stars] 

    data = list(zip(input_ids, attention_masks, labels, ord_labels))
    random.seed(seed)
    random.shuffle(data)
    writer = tf.io.TFRecordWriter(os.path.join(save_path, filename))
    ord_writer = tf.io.TFRecordWriter(os.path.join(save_path, 'ord_' + filename))
    for input_ids, attention_mask, label, ord_label in data:
        feature = {
            'input_ids': _create_int_feature(input_ids),
            'attention_mask': _create_int_feature(attention_mask),
            'label': _create_int_feature(label)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

        feature['label'] = _create_int_feature(ord_label)
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        ord_writer.write(example.SerializeToString())


def _encode_data(text, tokenizer, max_length=MAX_LENGTH):
    """
    Tokenizes and encodes the data in `text`. Returns the encoded input ids and attention mask (to identify padded tokens).
    """
    encoding = tokenizer.batch_encode_plus(text, max_length=max_length, pad_to_max_length=True, return_attention_masks=True)
    input_ids = encoding['input_ids']
    attention_masks = encoding['attention_mask']
    return input_ids, attention_masks


def _create_int_feature(values):
    try:
        values = list(values)
    except TypeError:
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def load_data(split='train', ordinal=True, path='data/datasets', buffer_size=50000, batch_size=32, max_length=MAX_LENGTH):
    """
    Loads data from a TFRecord file and returns it as a batched TFRecordDataset.
    """
    filename = split + '.tfrecord'
    if ordinal:
        filename = 'ord_' + filename
    dataset = tf.data.TFRecordDataset(os.path.join(path, filename))
    dataset = dataset.map(lambda x: _decode_record(x, ordinal, max_length))
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset


def _decode_record(record, ordinal=True, max_length=MAX_LENGTH):
    if ordinal:
        label_len = 4
    else:
        label_len = 1
    features = {
        'input_ids': tf.io.FixedLenFeature([max_length], tf.int64),
        'attention_mask': tf.io.FixedLenFeature([max_length], tf.int64),
        'label': tf.io.FixedLenFeature([label_len], tf.int64)
    }
    record = tf.io.parse_single_example(record, features)
    label = record.pop('label')
    return record, label


def split_reviews(path='data/yelp_review_training_dataset.jsonl', save_path='data'):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text = []
    stars = []
    with open(path, 'rb') as f:
        for item in json_lines.reader(f):
            text.append(tokenizer.basic_tokenizer._clean_text(item['text']) + '\n')
            stars.append(item['stars'])
    with open(os.path.join(save_path, 'yelp.txt'), 'w') as f:
        f.writelines(text)
    with open(os.path.join(save_path, 'stars.pickle'), 'wb') as f:
        pickle.dump(stars, f)


def combine_paraphrased(path='back_translate/back_trans_data/paraphrase', save_path='data/paraphrased_yelp.txt'):
    text = []
    for i in range(10):
        with open(os.path.join(path, 'file_' + str(i) + '_of_10.json'), 'rb') as f:
            text.extend(f.readlines())
    with open(save_path, 'wb') as f:
        f.writelines(text)


if __name__ == "__main__":
    split_data()
    process_data()
