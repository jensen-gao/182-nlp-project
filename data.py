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
    text = text[:100]
    paraphrased_text = paraphrased_text[:100]
    stars = stars[:100]
    np.random.seed(123456)
    n_data = len(text)
    order = np.random.permutation(n_data)
    train_len = math.floor(0.6 * n_data)
    valid_len = math.floor(0.2 * n_data)
    train_indices = order[:train_len]
    valid_indices = order[train_len: train_len + valid_len]
    test_indices = order[train_len + valid_len:]

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoding = tokenizer.batch_encode_plus(text, max_length=512, pad_to_max_length=True, return_attention_masks=True)
    ids = np.array(encoding['input_ids'])
    masks = np.array(encoding['attention_mask'])
    paraphrased_encoding = tokenizer.batch_encode_plus(paraphrased_text, max_length=512, pad_to_max_length=True,
                                                       return_attention_masks=True)
    paraphrased_ids = np.array(paraphrased_encoding['input_ids'])
    paraphrased_masks = np.array(paraphrased_encoding['attention_mask'])
    stars = np.array(stars)

    write_data(os.path.join(path, 'train.pickle'), ids, paraphrased_ids, masks, paraphrased_masks, stars, train_indices)
    write_data(os.path.join(path, 'valid.pickle'), ids, paraphrased_ids, masks, paraphrased_masks, stars, valid_indices)
    write_data(os.path.join(path, 'test.pickle'), ids, paraphrased_ids, masks, paraphrased_masks, stars, test_indices)


def write_data(path, ids, paraphrased_ids, masks, paraphrased_masks, stars, indices):
    combined_ids = np.concatenate((ids[indices], paraphrased_ids[indices]), axis=0)
    combined_masks = np.concatenate((masks[indices], paraphrased_masks[indices]), axis=0)
    combined_stars = np.tile(stars[indices], 2)
    combined_data = (combined_ids, combined_masks, combined_stars)
    pickle.dump(combined_data, open(path, 'wb'))


def load_data(split='train', path='data', buffer_size=10000, batch_size=24, weighted=False):
    data = pickle.load(open(os.path.join(path, split + '.pickle'), 'rb'))
    stars = np.array([[1] * (int(star) - 1) + [0] * (4 - int(star) + 1) for star in data[2]])
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


if __name__ == "__main__":
    write_reviews_to_txt()
    process_data()

