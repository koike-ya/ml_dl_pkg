for model in cnn rnn cnn_rnn
do
  python example_eeg.py --cuda --epochs 20 --test --model-type $model --amp --tensorboard --return-prob --n-parallel 2
done