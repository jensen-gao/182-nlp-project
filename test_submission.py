import json, sys
import numpy as np
import tensorflow as tf
import transformers
from transformers import DistilBertTokenizer
from bert_models.tf_distilbert_for_classification import TFDistilBertForClassification


model_1_path = 'models/epoch_1'
model_2_path = 'models/epoch_2'

model_1_config = transformers.DistilBertConfig.from_pretrained(model_1_path, num_labels=5)
model_2_config = transformers.DistilBertConfig.from_pretrained(model_2_path, num_labels=5)
class_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model_1 = TFDistilBertForClassification.from_pretrained(model_1_path, config=model_1_config)
model_2 = TFDistilBertForClassification.from_pretrained(model_2_path, config=model_2_config)
model_1.compile(optimizer='adam', loss=class_loss)
model_2.compile(optimizer='adam', loss=class_loss)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


def softmax(x):
    subtract = np.max(x, axis=1)
    subtract = np.expand_dims(subtract, 1)
    e_x = np.exp(x - subtract)
    divide = np.sum(e_x, axis=1)
    divide = np.expand_dims(divide, 1)
    return e_x / divide


def eval(text):
    # This is where you call your model to get the number of stars output
    encoding = tokenizer.batch_encode_plus(text, max_length=384, pad_to_max_length=True, return_attention_masks=True)
    input_ids = np.array(encoding['input_ids'])
    attention_masks = np.array(encoding['attention_mask'])
    logits_1 = model_1.predict_on_batch([input_ids, attention_masks])
    logits_2 = model_2.predict_on_batch([input_ids, attention_masks])
    pred_1 = softmax(logits_1)
    pred_2 = softmax(logits_2)
    pred = pred_1 * pred_2
    return np.argmax(pred, axis=1) + 1


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
            if review_ids:
                results = eval(text)
                for i in range(len(review_ids)):
                    fw.write(json.dumps({"review_id": review_ids[i], "predicted_stars": int(results[i])}) + "\n")
    print("Output prediction file written")
else:
    print("No validation file given")
