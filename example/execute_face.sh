for model in vgg16 resnet resnext
do
  python example_face.py --cuda --epochs 150 --test --model-type $model --amp --tensorboard --return-prob --n-parallel 2
done

for model in vgg16 resnet resnext
do
  python example_face.py --cuda --epochs 150 --test --model-type $model --amp --tensorboard --return-prob --pretrained --n-parallel 2
done