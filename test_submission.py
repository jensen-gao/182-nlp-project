import json, sys
import numpy as np
import tensorflow as tf
import transformers
from transformers import DistilBertTokenizer
from bert_models.tf_distilbert_for_classification import TFDistilBertForClassification


model_1_path = 'models/epoch_1'
model_2_path = 'models/epoch_2'
model_3_path = 'models/epoch_3'

model_1_config = transformers.DistilBertConfig.from_pretrained(model_1_path, num_labels=5)
model_2_config = transformers.DistilBertConfig.from_pretrained(model_2_path, num_labels=5)
model_3_config = transformers.DistilBertConfig.from_pretrained(model_3_path, num_labels=5)
class_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model_1 = TFDistilBertForClassification.from_pretrained(model_1_path, config=model_1_config)
model_2 = TFDistilBertForClassification.from_pretrained(model_2_path, config=model_2_config)
model_3 = TFDistilBertForClassification.from_pretrained(model_3_path, config=model_3_config)
model_1.compile(optimizer='adam', loss=class_loss)
model_2.compile(optimizer='adam', loss=class_loss)
model_3.compile(optimizer='adam', loss=class_loss)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def eval(text):
	# This is where you call your model to get the number of stars output
	encoding = tokenizer.batch_encode_plus(text, max_length=384, pad_to_max_length=True, return_attention_masks=True)
	input_ids = np.array(encoding['input_ids'])
	attention_masks = np.array(encoding['attention_mask'])
	prediction_1 = model_1.predict_on_batch([input_ids, attention_masks])
	prediction_2 = model_2.predict_on_batch([input_ids, attention_masks])
	prediction_3 = model_3.predict_on_batch([input_ids, attention_masks])
	stars_1 = np.argmax(prediction_1, axis=1) + 1
	stars_2 = np.argmax(prediction_2, axis=1) + 1
	stars_3 = np.argmax(prediction_3, axis=1) + 1
	return np.round(stars_1 + stars_2 + stars_3) / 3

if len(sys.argv) > 1:
	validation_file = sys.argv[1]
	with open("output.jsonl", "w") as fw:
		with open(validation_file, "r") as fr:
			review_ids = []
			text = []
			for line in fr:
				review = json.loads(line)
				review_ids.append(review['review_id'])
				text.append(review['text'])
				if len(review_ids) >= 16:
					results = eval(text)
					for i in range(len(review_ids)):
						fw.write(json.dumps({"review_id": review_ids[i], "predicted_stars": int(results[i])})+"\n")
					review_ids = []
					text = []
	print("Output prediction file written")
else:
	print("No validation file given")
