import json, sys, jsonlines
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

#config2 = transformers.DistilBertConfig.from_pretrained('models/ord_4', num_labels=4)
#loss2 = tf.keras.losses.BinaryCrossentropy(from_logits=True)
#model2 = TFDistilBertForOrdinalRegression.from_pretrained('models/ord_4', config=config2)
#model2.compile(optimizer='adam', loss=loss2)
#ord_models.append(model2)

config3 = transformers.DistilBertConfig.from_pretrained('models/pretrain_nofeat_class', num_labels=5)
model3 = TFDistilBertForClassification.from_pretrained('models/pretrain_nofeat_class', config=config3)
model3.compile(optimizer='adam', loss=loss)
class_models.append(model3)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

class_stars = []
ord_stars = []
epoch1_stars = []
epoch2_stars = []

def eval(text):
    # This is where you call your model to get the number of stars output
    encoding = tokenizer.encode_plus(text, max_length=384)
    input_ids = encoding['input_ids']
    input_ids = np.expand_dims(input_ids, 0)

    #temp_class = []
    temp_logits = np.ones(5)
    first = False
    for model in class_models:
        prediction = model.predict_on_batch(input_ids)
        #temp_class.append(np.argmax(prediction) + 1)
        temp_logits *= np.exp(prediction)[0] / np.sum(np.exp(prediction))
        if first:
            epoch1_stars.append(np.argmax(prediction) + 1)
        else:
            epoch2_stars.append(np.argmax(prediction) + 1)
            first = True
    class_prediction = np.argmax(temp_logits) + 1
    #class_prediction = np.floor(np.mean(temp_class) + 0.5)
    class_stars.append(class_prediction)

	#temp_ord = []
	#for model in ord_models:
	#	prediction = model2.predict_on_batch(input_ids)
	#	temp_ord.append(np.sum(prediction > 0) + 1)
	#ord_prediction = np.floor(np.mean(temp_ord) + 0.5)
	#ord_stars.append(ord_prediction)

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
    print("Percent Correct: ", correct, "MAE: ", mae)
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
    distribution = np.zeros(5)
    for star in stars:
        distribution[int(star) - 1] += 1
    return distribution / len(stars)


if len(sys.argv) > 1:
    for i in range(1, len(sys.argv)):
        dataset = sys.argv[i]

        print("Dataset: ", dataset)
        stars = []
        with jsonlines.open(dataset) as f:
            for line in f.iter():
                eval(line['text'])
                stars.append(line['stars'])

        assert len(stars) == len(class_stars), "Mismatch Datasets"

        print("Stars Distibution: ", distribution(stars))
        print()

        #print("Epoch 1 ")
        #compute_stats(epoch1_stars, stars)

        print("Epoch 2: ")
        compute_stats(epoch2_stars, stars)

        print("Ensemble: ")
        compute_stats(class_stars, stars)
        
        print("Diff betweeen 1, 2: ")
        compute_stats(epoch1_stars, epoch2_stars)

        class_stars.clear()
        ord_stars.clear()
        epoch1_stars.clear()
        epoch2_stars.clear()

else:
	print("No validation file given")


