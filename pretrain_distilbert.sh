python run_language_modeling.py --output_dir=pretrained/ --model_type=distilbert \
--model_name_or_path=distilbert-base-uncased --do_train --train_data_file=data/train_text.txt --mlm \
--block_size=512 --line_by_line --per_gpu_train_batch_size=16 --do_eval --evaluate_during_training \
--eval_data_file=data/eval_text.txt --per_gpu_eval_batch_size=16 --max_steps=100000 \
--warmup_steps=10000 --logging_steps=5000 --save_steps=5000
