import json_lines
import os
import math
import numpy as np
import tensorflow as tf
import pickle
from transformers import DistilBertTokenizer


def process_data(path='data'):
    with open(os.path.join(path, 'yelp.txt'), 'r') as f:
        text = f.readlines()
    with open(os.path.join(path, 'yelp.txt'), 'r') as f:
        paraphrased_text = f.readlines()
    stars = []
    with open(os.path.join(path, 'yelp_review_training_dataset.jsonl'), 'rb') as f:
        for item in json_lines.reader(f):
            stars.append(item['stars'])
    assert len(text) == len(paraphrased_text) == len(stars)
    np.random.seed(123456)
    n_data = len(text)
    order = np.random.permutation(n_data)
    train_len = math.floor(0.6 * n_data)
    valid_len = math.floor(0.2 * n_data)
    train_indices = order[:train_len]
    valid_indices = order[train_len: train_len + valid_len]
    test_indices = order[train_len + valid_len:]

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokens = tokenizer.batch_encode_plus(text)['input_ids']
    tokens = np.array([seq + [0] * (512 - len(seq)) if len(seq) <= 512 else seq[:129] + seq[-383:] for seq in tokens])
    paraphrased_tokens = tokenizer.batch_encode_plus(paraphrased_text)['input_ids']
    paraphrased_tokens = np.array([seq + [0] * (512 - len(seq)) if len(seq) <= 512 else seq[:129] + seq[-383:]
                                   for seq in paraphrased_tokens])
    stars = np.array(stars)

    train_tokens = np.concatenate((tokens[train_indices], paraphrased_tokens[train_indices]), axis=0)
    train_stars = np.tile(stars[train_indices], 2)
    train_data = (train_tokens, train_stars)
    pickle.dump(train_data, open(os.path.join(path, 'train.pickle'), 'wb'))

    valid_tokens = np.concatenate((tokens[valid_indices], paraphrased_tokens[valid_indices]), axis=0)
    valid_stars = np.tile(stars[valid_indices], 2)
    valid_data = (valid_tokens, valid_stars)
    pickle.dump(valid_data, open(os.path.join(path, 'valid.pickle'), 'wb'))

    test_tokens = np.concatenate((tokens[test_indices], paraphrased_tokens[test_indices]), axis=0)
    test_stars = np.tile(stars[test_indices], 2)
    test_data = (test_tokens, test_stars)
    pickle.dump(test_data, open(os.path.join(path, 'test.pickle'), 'wb'))


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


def write_reviews_to_txt(path='data/yelp_review_training_dataset.jsonl', save_path='data/yelp.txt'):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text = []
    with open(path, 'rb') as f:
        for item in json_lines.reader(f):
            text.append(tokenizer.basic_tokenizer._clean_text(item['text']) + '\n')
    with open(save_path, 'w') as f:
        f.writelines(text)


if __name__ == "__main__":
    write_reviews_to_txt()
    process_data()

