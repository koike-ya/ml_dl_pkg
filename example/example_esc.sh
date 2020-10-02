python example_esc.py --cuda --epochs 20 --test --model-type panns --amp --tensorboard --return-prob \
--cv-name group --n-splits 5 --batch-size 16 --transform logmel --window-size 0.3 --window-stride 0.04 \
--checkpoint-path ./cnn14.pth