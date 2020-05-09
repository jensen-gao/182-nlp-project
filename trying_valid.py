import json, sys
import numpy as np
import tensorflow as tf
import transformers
from transformers import DistilBertTokenizer
from bert_models.tf_distilbert_for_ordinal_regression import TFDistilBertForOrdinalRegression
from bert_models.tf_distilbert_for_classification import TFDistilBertForClassification

class_models = []
ord_models = []

config = transformers.DistilBertConfig.from_pretrained('models/pretrain_nofeat_class_epoch2/', num_labels=5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model = TFDistilBertForClassification.from_pretrained('models/pretrain_nofeat_class_epoch2/', config=config)
model.compile(optimizer='adam', loss=loss)
class_models.append(model)

config2 = transformers.DistilBertConfig.from_pretrained('models/ord_4/', num_labels=4)
loss2 = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model2 = TFDistilBertForOrdinalRegression.from_pretrained('models/ord_4/', config=config2)
model2.compile(optimizer='adam', loss=loss2)
ord_models.append(model2)

#config5 = transformers.DistilBertConfig.from_pretrained('models/pretrain_nofeat_class_epoch3/', num_labels=5)
#model5 = TFDistilBertForClassification.from_pretrained('models/pretrain_nofeat_class_epoch3/', config=config5)
#model5.compile(optimizer='adam', loss=loss)
#class_models.append(model5)

#config6 = transformers.DistilBertConfig.from_pretrained('models/ord_3/', num_labels=4)
#model6 = TFDistilBertForOrdinalRegression.from_pretrained('models/ord_3/', config=config6)
#model6.compile(optimizer='adam', loss=loss2)
#ord_models.append(model6)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def eval(text):
	# This is where you call your model to get the number of stars output
	encoding = tokenizer.encode_plus(text, max_length=384)
	input_ids = encoding['input_ids']
	input_ids = np.expand_dims(input_ids, 0)

	total_stars = []
	for model in class_models:
		prediction = model.predict_on_batch(input_ids)
		total_stars.append(np.argmax(prediction) + 1)
		#total_stars.append(np.argmax(prediction) + 1)

	for model in ord_models:
		prediction = model.predict_on_batch(input_ids)
		total_stars.append(np.sum(prediction > 0) + 1)
		#total_stars.append(np.sum(prediction > 0) + 1)

	stars = np.floor(np.mean(total_stars) + 0.5)
	return int(stars)


if len(sys.argv) == 2:
	valid_text = sys.argv[1]
	with open("output_data.jsonl", "w") as fw:
		with open(valid_text, "r") as f1:
			for line in f1:
				review = line.rstrip('\n')
				fw.write(json.dumps({"predicted_stars": eval(review)})+"\n")
	print("Output prediction file written")
else:
	print("No validation file given")


