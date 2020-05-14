First run ```python data_utils.py``` to preprocess the data.
Then run ```python train.py``` with appropriate command line arguments to train the model.
- ```-v FOLDER``` will save checkpoints at ```checkpoints/FOLDER``` and the final model at ```models/FOLDER```.
- ```-p``` will use the DistilBERT model pretrained with in-domain MLM instead of the default original base model.
- ```-o``` will train an ordinal regression model instead of the default classification model.
- ```-f``` will freeze the pretrained layers instead of training the entire model end-to-end.
- ```-e NUM_EPOCHS``` will set the number of training epochs to NUM_EPOCHS (default 4).
- ```-n``` will add layer normalization before the final classification output.
- ```-b BATCHES_PER_GPU``` will set the batch size per GPU (total batch size is BATCHES_PER_GPU x # of GPUs).
