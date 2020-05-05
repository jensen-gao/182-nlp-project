import json, sys
import numpy as np
import tensorflow as tf
import transformers
from transformers import DistilBertTokenizer
from bert_models.tf_distilbert_for_ordinal_regression import TFDistilBertForOrdinalRegression

config = transformers.DistilBertConfig.from_pretrained('models/final/', num_labels=4)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model = TFDistilBertForOrdinalRegression.from_pretrained('models/final/', config=config)
model.compile(optimizer='adam', loss=loss)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def eval(text):
	# This is where you call your model to get the number of stars output
	encoding = tokenizer.encode_plus(text, max_length=384)
	input_ids = encoding['input_ids']
	prediction = model.predict([input_ids])
	stars = np.sum(prediction > 0) + 1
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