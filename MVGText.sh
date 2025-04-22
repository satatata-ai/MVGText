nohup python MVGText.py --test_dataset ag  --target_model wordLSTM --target_model_path lstm --test_len 1000 >> MVGText_lstm_ag.log 2>&1 &
nohup python MVGText.py --test_dataset mr  --target_model wordLSTM --target_model_path lstm --test_len 1000 >> MVGText_lstm_mr.log 2>&1 &
nohup python MVGText.py --test_dataset yahoo  --target_model wordLSTM --target_model_path lstm --test_len 1000 >> MVGText_lstm_yahoo.log 2>&1 &
nohup python MVGText.py --test_dataset yelp  --target_model wordLSTM --target_model_path lstm --test_len 1000 >> MVGText_lstm_yelp.log 2>&1 &
nohup python MVGText.py --test_dataset imdb  --target_model wordLSTM --target_model_path lstm --test_len 1000 >> MVGText_lstm_imdb.log 2>&1 &

nohup python MVGText.py --test_dataset ag  --target_model bert --target_model_path bert --test_len 1000 >> MVGText_bert_ag.log 2>&1 &
nohup python MVGText.py --test_dataset mr  --target_model bert --target_model_path bert --test_len 1000 >> MVGText_bert_mr.log 2>&1 &
nohup python MVGText.py --test_dataset yahoo  --target_model bert --target_model_path bert --test_len 1000 >> MVGText_bert_yahoo.log 2>&1 &
nohup python MVGText.py --test_dataset yelp  --target_model bert --target_model_path bert --test_len 1000 >> MVGText_bert_yelp.log 2>&1 &
nohup python MVGText.py --test_dataset imdb  --target_model bert --target_model_path bert --test_len 1000 >> MVGText_bert_imdb.log 2>&1 &

nohup python MVGText.py --test_dataset ag  --target_model wordCNN --target_model_path cnn --test_len 1000 >> MVGText_cnn_ag.log 2>&1 &
nohup python MVGText.py --test_dataset mr  --target_model wordCNN --target_model_path cnn --test_len 1000 >> MVGText_cnn_mr.log 2>&1 &
nohup python MVGText.py --test_dataset yahoo  --target_model wordCNN --target_model_path cnn --test_len 1000 >> MVGText_cnn_yahoo.log 2>&1 &
nohup python MVGText.py --test_dataset yelp  --target_model wordCNN --target_model_path cnn --test_len 1000 >> MVGText_cnn_yelp.log 2>&1 &
nohup python MVGText.py --test_dataset imdb  --target_model wordCNN --target_model_path cnn --test_len 1000 >> MVGText_cnn_imdb.log 2>&1 &


nohup python MVGText_nli.py --test_dataset snli  --target_model bert --target_model_path bert --data_size 1000 >> MVGText_bert_snli.log 2>&1 &
nohup python MVGText_nli.py --test_dataset mnli  --target_model bert --target_model_path bert --data_size 1000 >> MVGText_bert_mnli.log 2>&1 &
nohup python MVGText_nli.py --test_dataset mnli_mismatched  --target_model bert --target_model_path bert --data_size 1000 >> MVGText_bert_mnli_mismatched.log 2>&1 &
nohup python MVGText_nli.py --test_dataset mnli_matched  --target_model bert --target_model_path bert --data_size 1000 >> MVGText_bert_mnli_matched.log 2>&1 &


