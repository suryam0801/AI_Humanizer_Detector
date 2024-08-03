Repository Overview: 
- (main.py) endpoint for humanizing and detecting AI Text
- (humanize.py) first gets instructions to humanize the text and then humanizes the text using finetuned gpt 4o mini model
- (detect_ai.py) uses custom trained BERT model to detect AI score of the text
- (finetune_openai.py) finetunes gpt 40 mini model on the training data
- (colab_train_model.py) trains the custom BERT model on the training data. Make sure to run in google colab using T4 GPU for faster training - its free
- (winston.py) contains the call to winston API to detect AI score of the text
- (TrainingData) contains the training data for the custom trained BERT model and the finetuned gpt 40 mini model
- (Data) contains the source data of the training data used to train the custom trained BERT model and the finetuned gpt 40 mini model