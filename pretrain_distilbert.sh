python run_language_modeling.py --output_dir=pretrained --model_type=distilbert \
--model_name_or_path=distilbert-base-uncased --do_train --train_data_file=data/combined_yelp_train.txt --mlm \
--block_size=512 --line_by_line --per_gpu_eval_batch_size 16
