import json_lines
import os
import math
import numpy as np
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
    text = text[:100]
    paraphrased_text = paraphrased_text[:100]
    stars = stars[:100]
    n_data = len(text)
    random.seed(123456)
    combined = list(zip(text, paraphrased_text, stars))
    random.shuffle(combined)
    text, paraphrased_text, stars = zip(*combined)
    train_len = math.floor(0.6 * n_data)
    valid_len = math.floor(0.2 * n_data)
    train_text = text[:train_len] + paraphrased_text[:train_len]
    valid_text = text[train_len: train_len + valid_len] + paraphrased_text[train_len: train_len + valid_len]
    test_text = text[train_len + valid_len:] + paraphrased_text[train_len + valid_len:]
    with open(os.path.join(save_path, 'train_text.txt'), 'w') as f:
        f.writelines(train_text)
    with open(os.path.join(save_path, 'valid_text.txt'), 'w') as f:
        f.writelines(valid_text)
    with open(os.path.join(save_path, 'test_text.txt'), 'w') as f:
        f.writelines(test_text)
    train_stars = 2 * stars[:train_len]
    valid_stars = 2 * stars[train_len: train_len + valid_len]
    test_stars = 2 * stars[train_len + valid_len:]
    pickle.dump(train_stars, open(os.path.join(save_path, 'train_stars.pickle'), 'wb'))
    pickle.dump(valid_stars, open(os.path.join(save_path, 'valid_stars.pickle'), 'wb'))
    pickle.dump(test_stars, open(os.path.join(save_path, 'test_stars.pickle'), 'wb'))


def process_data(load_path='data', save_path='data'):
    with open(os.path.join(load_path, 'train_text.txt'), 'r') as f:
        train_text = f.readlines()
    with open(os.path.join(load_path, 'valid_text.txt'), 'r') as f:
        valid_text = f.readlines()
    with open(os.path.join(load_path, 'test_text.txt'), 'r') as f:
        test_text = f.readlines()
    train_stars = pickle.load(open(os.path.join(load_path, 'train_stars.pickle'), 'rb'))
    valid_stars = pickle.load(open(os.path.join(load_path, 'valid_stars.pickle'), 'rb'))
    test_stars = pickle.load(open(os.path.join(load_path, 'test_stars.pickle'), 'rb'))

    assert len(train_text) == len(train_stars)
    assert len(valid_text) == len(valid_stars)
    assert len(test_text) == len(test_stars)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_ids, train_masks = encode_data(train_text, tokenizer)
    valid_ids, valid_masks = encode_data(valid_text, tokenizer)
    test_ids, test_masks = encode_data(test_text, tokenizer)

    train_data = (train_ids, train_masks, train_stars)
    pickle.dump(train_data, open(os.path.join(save_path, 'train.pickle'), 'wb'))
    valid_data = (valid_ids, valid_masks, valid_stars)
    pickle.dump(valid_data, open(os.path.join(save_path, 'valid.pickle'), 'wb'))
    test_data = (test_ids, test_masks, test_stars)
    pickle.dump(test_data, open(os.path.join(save_path, 'test.pickle'), 'wb'))


def encode_data(text, tokenizer):
    encoding = tokenizer.batch_encode_plus(text, max_length=512, pad_to_max_length=True, return_attention_masks=True)
    ids = encoding['input_ids']
    masks = encoding['attention_mask']
    return ids, masks


def load_data(split='train', path='data', buffer_size=10000, batch_size=16, weighted=False):
    data = pickle.load(open(os.path.join(path, split + '.pickle'), 'rb'))
    stars = [[1] * (int(star) - 1) + [0] * (4 - int(star) + 1) for star in data[2]]
    data = tf.data.Dataset.from_tensor_slices(({'input_ids': data[0], 'attention_mask': data[1]}, stars))
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
