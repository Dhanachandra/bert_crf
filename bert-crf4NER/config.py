class Config(object):	
	apr_dir = '../model/'
	data_dir = '../corpus/'
	model_name = 'model_4.pt'
	epoch = 5
	bert_model = 'bert-base-cased'
	lr = 5e-5
	eps = 1e-8
	batch_size = 16
	mode = 'prediction' # for prediction mode = "prediction"
	training_data = 'eng.train.txt'
	val_data = 'eng.testa.txt'
	test_data = 'eng.testa.txt'
	test_out = 'test_prediction.csv'
	raw_prediction_output = 'raw_prediction.csv'
