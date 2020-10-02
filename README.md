# What's this repository for?
To boost baseline model construction and incremental hypothesis testing speed.

You can use this library for both machine learning and deep learning tasks by changing only some arguments.

# Requirements
- cuda >= 10.0 (for GPU users)

# Setup
## Virtual environment
```
conda create -n ml_dl_pkg python=3.7
source activate ml_dl_pkg
```

### For mac user
Please install xgboost from source
```
https://xgboost.readthedocs.io/en/latest/build.html#building-on-osx

```

## Apex(for GPU users)
ref: https://github.com/NVIDIA/apex
```
cd ../
git clone https://github.com/NVIDIA/apex;cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../ml_dl_pkg
```

## For CPU users
```
mkdir --parent ../apex/amp
```

## Installation
```
pip install -r requirements.txt
python setup.py install
```

# Example
```
mkdir input/eeg;unzip 'input/*.zip' -d input/eeg/;
python example.py --transform spectrogram
```