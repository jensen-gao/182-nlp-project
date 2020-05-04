import json_lines
import os
import math
import tensorflow as tf
import pickle
from transformers import DistilBertTokenizer
import random


def split_data(load_path='data', save_path='data'):
    with open(os.path.join(load_path, 'yelp.txt'), 'r') as f:
        text = f.readlines()
    with open(os.path.join(load_path, 'paraphrased_yelp.txt'), 'r') as f:
        paraphrased_text = f.readlines()
    stars = []
    with open(os.path.join(load_path, 'yelp_review_training_dataset.jsonl'), 'rb') as f:
        for item in json_lines.reader(f):
            stars.append(item['stars'])
    assert len(text) == len(paraphrased_text) == len(stars)
    n_data = len(text)
    random.seed(123456)
    combined = list(zip(text, paraphrased_text, stars))
    random.shuffle(combined)
    text, paraphrased_text, stars = zip(*combined)
    train_len = math.floor(0.7 * n_data)
    valid_len = math.floor(0.1 * n_data)
    train_text = text[:train_len] + paraphrased_text[:train_len]
    valid_text = text[train_len: train_len + valid_len] + paraphrased_text[train_len: train_len + valid_len]
    test_text = text[train_len + valid_len:] + paraphrased_text[train_len + valid_len:]
    with open(os.path.join(save_path, 'train_text.txt'), 'w') as f:
        f.writelines(train_text)
    with open(os.path.join(save_path, 'valid_text.txt'), 'w') as f:
        f.writelines(valid_text)
    with open(os.path.join(save_path, 'test_text.txt'), 'w') as f:
        f.writelines(test_text)
    with open(os.path.join(save_path, 'original_test_text.txt'), 'w') as f:
        f.writelines(text[train_len + valid_len:])

    train_stars = 2 * stars[:train_len]
    valid_stars = 2 * stars[train_len: train_len + valid_len]
    test_stars = 2 * stars[train_len + valid_len:]
    pickle.dump(train_stars, open(os.path.join(save_path, 'train_stars.pickle'), 'wb'))
    pickle.dump(valid_stars, open(os.path.join(save_path, 'valid_stars.pickle'), 'wb'))
    pickle.dump(test_stars, open(os.path.join(save_path, 'test_stars.pickle'), 'wb'))
    pickle.dump(stars[train_len + valid_len:], open(os.path.join(save_path, 'original_test_stars.pickle'), 'wb'))


def process_data(load_path='data', save_path='data', ordinal=True):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    with open(os.path.join(load_path, 'train_text.txt'), 'r') as f:
        train_text = f.readlines()
    train_stars = pickle.load(open(os.path.join(load_path, 'train_stars.pickle'), 'rb'))
    assert len(train_text) == len(train_stars)
    save_name = 'train.tfrecord'
    if ordinal:
        save_name = 'ord_' + save_name
    _write_dataset(train_text, train_stars, tokenizer, 123, os.path.join(save_path, save_name), ordinal)

    with open(os.path.join(load_path, 'valid_text.txt'), 'r') as f:
        valid_text = f.readlines()
    valid_stars = pickle.load(open(os.path.join(load_path, 'valid_stars.pickle'), 'rb'))
    assert len(valid_text) == len(valid_stars)
    save_name = 'valid.tfrecord'
    if ordinal:
        save_name = 'ord_' + save_name
    _write_dataset(valid_text, valid_stars, tokenizer, 456, os.path.join(save_path, save_name), ordinal)

    with open(os.path.join(load_path, 'test_text.txt'), 'r') as f:
        test_text = f.readlines()
    test_stars = pickle.load(open(os.path.join(load_path, 'test_stars.pickle'), 'rb'))
    assert len(test_text) == len(test_stars)
    save_name = 'test.tfrecord'
    if ordinal:
        save_name = 'ord_' + save_name
    _write_dataset(test_text, test_stars, tokenizer, 789, os.path.join(save_path, save_name), ordinal)

    with open(os.path.join(load_path, 'original_test_text.txt'), 'r') as f:
        original_test_text = f.readlines()
    original_test_stars = pickle.load(open(os.path.join(load_path, 'original_test_stars.pickle'), 'rb'))
    assert len(original_test_text) == len(original_test_stars)
    save_name = 'original_test.tfrecord'
    if ordinal:
        save_name = 'ord_' + save_name
    _write_dataset(original_test_text, original_test_stars, tokenizer, 789,
                   os.path.join(save_path, save_name), ordinal)


def _write_dataset(text, stars, tokenizer, seed, save_path, ordinal=True):
    input_ids, attention_masks = _encode_data(text, tokenizer)
    if ordinal:
        labels = [[1] * (int(star) - 1) + [0] * (4 - int(star) + 1) for star in stars]
    else:
        labels = [int(star - 1) for star in stars]
    data = list(zip(input_ids, attention_masks, labels))
    random.seed(seed)
    random.shuffle(data)
    writer = tf.io.TFRecordWriter(save_path)
    for input_ids, attention_mask, label in data:
        feature = {
            'input_ids': _create_int_feature(input_ids),
            'attention_mask': _create_int_feature(attention_mask),
            'label': _create_int_feature(label)
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())


def _encode_data(text, tokenizer):
    encoding = tokenizer.batch_encode_plus(text, max_length=384, pad_to_max_length=True, return_attention_masks=True)
    input_ids = encoding['input_ids']
    attention_masks = encoding['attention_mask']
    return input_ids, attention_masks


def _create_int_feature(values):
    try:
        values = list(values)
    except TypeError:
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def load_data(split='train', ordinal=True, path='data', buffer_size=50000, batch_size=32):
    filename = split + '.tfrecord'
    if ordinal:
        filename = 'ord_' + filename
    dataset = tf.data.TFRecordDataset(os.path.join(path, filename))
    dataset = dataset.map(lambda x: _decode_record(x, ordinal))
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    return dataset


def _decode_record(record, ordinal=True):
    if ordinal:
        label_len = 4
    else:
        label_len = 1
    features = {
        'input_ids': tf.io.FixedLenFeature([384], tf.int64),
        'attention_mask': tf.io.FixedLenFeature([384], tf.int64),
        'label': tf.io.FixedLenFeature([label_len], tf.int64)
    }
    record = tf.io.parse_single_example(record, features)
    label = record.pop('label')
    return record, label


def write_reviews_to_txt(path='data/yelp_review_training_dataset.jsonl', save_path='data/yelp.txt'):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text = []
    with open(path, 'rb') as f:
        for item in json_lines.reader(f):
            text.append(tokenizer.basic_tokenizer._clean_text(item['text']) + '\n')
    with open(save_path, 'w') as f:
        f.writelines(text)


def combine_paraphrased(path='back_translate/back_trans_data/paraphrase', save_path='data/paraphrased_yelp.txt'):
    text = []
    for i in range(10):
        with open(os.path.join(path, 'file_' + str(i) + '_of_10.json'), 'rb') as f:
            text.extend(f.readlines())
    with open(save_path, 'wb') as f:
        f.writelines(text)


if __name__ == "__main__":
    # split_data()
    process_data()
