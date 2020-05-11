import json, sys, pickle
import numpy as np
import tensorflow as tf
import transformers
from transformers import DistilBertTokenizer
from bert_models.tf_distilbert_for_ordinal_regression import TFDistilBertForOrdinalRegression
from bert_models.tf_distilbert_for_classification import TFDistilBertForClassification

class_models = []
ord_models = []

config = transformers.DistilBertConfig.from_pretrained('models/pretrain_nofeat_class_epoch2', num_labels=5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model = TFDistilBertForClassification.from_pretrained('models/pretrain_nofeat_class_epoch2', config=config)
model.compile(optimizer='adam', loss=loss)
class_models.append(model)

config2 = transformers.DistilBertConfig.from_pretrained('models/ord_4', num_labels=4)
loss2 = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model2 = TFDistilBertForOrdinalRegression.from_pretrained('models/ord_4', config=config2)
model2.compile(optimizer='adam', loss=loss2)
ord_models.append(model2)

config3 = transformers.DistilBertConfig.from_pretrained('models/pretrain_nofeat_class', num_labels=5)
model3 = TFDistilBertForClassification.from_pretrained('models/pretrain_nofeat_class', config=config)
model3.compile(optimizer='adam', loss=loss)
class_models.append(model3)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class_stars = []
ord_stars = []
class_stars2 = []

def eval(text):
	# This is where you call your model to get the number of stars output
    encoding = tokenizer.encode_plus(text, max_length=384)
    input_ids = encoding['input_ids']
    input_ids = np.expand_dims(input_ids, 0)

    prediction = model.predict_on_batch(input_ids)
    class_stars.append(np.argmax(prediction) + 1)

    prediction = model2.predict_on_batch(input_ids)
    ord_stars.append(np.sum(prediction > 0) + 1)

    prediction = model3.predict_on_batch(input_ids)
    class_stars2.append(np.argmax(prediction) + 1)

def evaluate(predicted, actual):
    correct = 1 - np.count_nonzero(np.subtract(predicted, actual)) / len(actual)
    mae = np.sum(np.abs(np.subtract(predicted, actual))) / len(predicted)
    return correct, mae

def analyze(predicted, actual):
    predicted_stats = np.zeros((5, 5))
    total_true = np.zeros(5)
    for i in range(len(predicted)):
        prediction = int(predicted[i])
        true = int(actual[i])
        predicted_stats[true - 1][prediction - 1] += 1
        total_true[true - 1] += 1
    return np.transpose(np.transpose(predicted_stats) / total_true)

def calculate_data_star(percentages, star):
    average = 0
    mae = 0
    for i in range(5):
        average += percentages[i] * (i + 1)
        mae += np.abs(i + 1 - star) * percentages[i]
    return average, mae

def compute_stats(type_stars, stars):
    correct, mae = evaluate(type_stars, stars)
    print("Percent Correct: ", correct)
    print("MAE: ", mae)
    print()

    percent_chart = analyze(type_stars, stars)
    star = 1
    for row in percent_chart:
        print("Star " + str(star) + " prediction breakdown: ")
        print(row)
        average_star, mae_star = calculate_data_star(row, star)
        print("Average Prediction:", average_star)
        print("MAE for " + str(star) +  ":", mae_star)
        print()
        star += 1

def distribution(stars):
    total = np.zeros(5)
    for star in stars:
        total[int(star) - 1] += 1
    return total / len(stars)

if len(sys.argv) == 3:
    valid_text = sys.argv[1]
    valid_stars = sys.argv[2]

    with open(valid_text, "r") as f1:
        for line in f1:
            review = line.rstrip('\n')
            eval(review)
    
    stars = []
    with open(valid_stars, "rb") as f2:
        stars = pickle.load(f2)

    assert len(class_stars) == len(stars), "Files are not of the same size"
    print("Stars Distibution: ", distribution(stars))
    print()

    print("Ensemble BS Round: ")
    compute_stats(np.round(np.add(class_stars, ord_stars) / 2), stars)
    
    print("Class Epoch 1: ")
    compute_stats(class_stars2, stars)
    
    print("2 Class normal round: ")
    compute_stats(np.floor(np.add(class_stars, class_stars2) / 2 + 0.5), stars)
    
    print("2 Class bs round: ")
    compute_stats(np.round(np.add(class_stars, class_stars2) / 2), stars)

    print("Ensemble 3: ")
    compute_stats(np.floor(np.add(np.add(class_stars, ord_stars), class_stars2) / 3 + 0.5), stars)
else:
    print("Wrong Number of Args")
