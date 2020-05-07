import json, sys
import numpy as np
import tensorflow as tf
import transformers
from transformers import DistilBertTokenizer
from bert_models.tf_distilbert_for_ordinal_regression import TFDistilBertForOrdinalRegression
from bert_models.tf_distilbert_for_classification import TFDistilBertForClassification

class_models = []
ord_models = []

config = transformers.DistilBertConfig.from_pretrained('models/bert_feat_class/', num_labels=5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model = TFDistilBertForClassification.from_pretrained('models/bert_feat_class/', config=config)
model.compile(optimizer='adam', loss=loss)
class_models.append(model)

config2 = transformers.DistilBertConfig.from_pretrained('models/bert_feat_ord/', num_labels=4)
loss2 = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model2 = TFDistilBertForOrdinalRegression.from_pretrained('models/bert_feat_ord/', config=config2)
model2.compile(optimizer='adam', loss=loss2)
ord_models.append(model2)

config3 = transformers.DistilBertConfig.from_pretrained('models/bert_nofeat_class/', num_labels=5)
model3 = TFDistilBertForClassification.from_pretrained('models/bert_nofeat_class/', config=config3)
model3.compile(optimizer='adam', loss=loss)
class_models.append(model3)

config4 = transformers.DistilBertConfig.from_pretrained('models/bert_nofeat_ord/', num_labels=4)
model4 = TFDistilBertForOrdinalRegression.from_pretrained('models/bert_nofeat_ord/', config=config4)
model4.compile(optimizer='adam', loss=loss2)
ord_models.append(model4)

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

	for model in ord_models:
		prediction = model.predict_on_batch(input_ids)
		total_stars.append(np.sum(prediction > 0) + 1)

	stars = np.floor(np.mean(total_stars) + 0.5)
	return int(stars)


if len(sys.argv) > 1:
	validation_file = sys.argv[1]
	with open("output.jsonl", "w") as fw:
		with open(validation_file, "r") as fr:
			for line in fr:
				review = json.loads(line)
				fw.write(json.dumps({"review_id": review['review_id'], "predicted_stars": eval(review['text'])})+"\n")
	print("Output prediction file written")
else:
	print("No validation file given")


