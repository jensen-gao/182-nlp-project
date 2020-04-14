import json_lines
import os
import math
import numpy as np
import tensorflow as tf
import pickle
from transformers import DistilBertTokenizer


def process_data(path, save_dir='data'):
    text = []
    stars = []
    with open(path, 'rb') as f:
        for item in json_lines.reader(f):
            text.append(item['text'])
            stars.append(item['stars'])

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    token_ids = np.array(tokenizer.batch_encode_plus(text, max_length=512, pad_to_max_length=True)['input_ids'])
    stars = np.array(stars)
    np.random.seed(123456)
    order = np.random.permutation(len(token_ids))
    token_ids = token_ids[order]
    stars = stars[order]

    train_len = math.floor(0.6 * len(token_ids))
    valid_len = math.floor(0.2 * len(token_ids))

    train_ids = token_ids[:train_len]
    train_stars = stars[:train_len]
    train_data = (train_ids, train_stars)
    pickle.dump(train_data, open(os.path.join(save_dir, 'train.pickle'), 'wb'))

    valid_ids = token_ids[train_len: train_len + valid_len]
    valid_stars = stars[train_len: train_len + valid_len]
    valid_data = (valid_ids, valid_stars)
    pickle.dump(valid_data, open(os.path.join(save_dir, 'valid.pickle'), 'wb'))

    test_ids = token_ids[train_len + valid_len:]
    test_stars = stars[train_len + valid_len:]
    test_data = (test_ids, test_stars)
    pickle.dump(test_data, open(os.path.join(save_dir, 'test.pickle'), 'wb'))


def load_data(split='train', path='data', buffer_size=10000, batch_size=24, weighted=False):
    data = pickle.load(open(os.path.join(path, split + '.pickle'), 'rb'))
    stars = np.array([[1] * (int(star) - 1) + [0] * (4 - int(star) + 1) for star in data[1]])
    data = tf.data.Dataset.from_tensor_slices((data[0], stars))
    if split == 'train':
        data = data.shuffle(buffer_size).batch(batch_size)
        if weighted:
            n = len(stars)
            weights = np.sum(stars, axis=0)
            weights = np.maximum(weights, n - weights)
            weights = np.sqrt(weights)
            weights /= np.max(weights)
            data = data.map(lambda x, y: (x, y, weights))
        return data
    data = data.batch(batch_size)
    return data


def write_reviews_to_txt(path, save_path='data/yelp.txt'):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text = []
    with open(path, 'rb') as f:
        for item in json_lines.reader(f):
            text.append(tokenizer.basic_tokenizer._clean_text(item['text']) + '\n')
    with open(save_path, 'w') as f:
        f.writelines(text)


if __name__ == "__main__":
    process_data('yelp_review_training_dataset.jsonl')

