for model in cnn rnn cnn_rnn
do
  python example/example_eeg.py train.cuda=True train.epochs=1 test=True n_parallel=1 train.model=$model \
         train.model.model_name=$model hydra.verbose=ml train.model.return_prob=True
done