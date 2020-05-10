import json, sys
import numpy as np
import tensorflow as tf
import transformers
from transformers import DistilBertTokenizer
from bert_models.tf_distilbert_for_classification import TFDistilBertForClassification
from bert_models.tf_distilbert_for_ordinal_regression import TFDistilBertForOrdinalRegression


class_path = 'models/final_class'
ord_path = 'models/final_ord'

class_config = transformers.DistilBertConfig.from_pretrained(class_path, num_labels=5)
ord_config = transformers.DistilBertConfig.from_pretrained(ord_path, num_labels=4)
class_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
ord_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
class_model = TFDistilBertForClassification.from_pretrained(class_path, config=class_config)
ord_model = TFDistilBertForOrdinalRegression.from_pretrained(ord_path, config=ord_config)
class_model.compile(optimizer='adam', loss=class_loss)
ord_model.compile(optimizer='adam', loss=ord_loss)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def eval(text):
	# This is where you call your model to get the number of stars output
	encoding = tokenizer.encode_plus(text, max_length=384)
	input_ids = encoding['input_ids']
	input_ids = np.expand_dims(input_ids, 0)
	class_prediction = class_model.predict_on_batch(input_ids)
	ord_prediction = ord_model.predict_on_batch(input_ids)
	class_stars = np.argmax(class_prediction) + 1
	ord_stars = np.sum(ord_prediction > 0) + 1
	return int(np.round((class_stars + ord_stars) / 2))


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
