# About DeepDTA: deep drug-target binding affinity prediction

The approach used in this work is the modeling of protein sequences and compound 1D representations (SMILES) with convolutional neural networks (CNNs) to predict the binding affinity value of drug-target pairs.

![Figure](https://github.com/hkmztrk/DeepDTA/blob/master/docs/figures/deepdta.PNG)
# Installation

## Data

## Requirements

You'll need to install following in order to run the codes.

*  [Python 3.4 <=](https://www.python.org/downloads/)
*  [Keras 2.x](https://pypi.org/project/Keras/)
*  [Tensorflow 1.x](https://www.tensorflow.org/install/)
*  numpy
*  matplotlib

You have to place "data" folder under "source" directory. 

# Usage
```
python run_experiments.py --num_windows 32 \
                          --seq_window_lengths 8 12 \
                          --smi_window_lengths 4 8 \
                          --batch_size 256 \
                          --num_epoch 100 \
                          --max_seq_len 1000 \
                          --max_smi_len 100 \
                          --dataset_path 'data/kiba/' \
                          --problem_type 1 \
                          --log_dir 'logs/'


```





**For citation:**

```
@article{ozturk2018deepdta,
  title={DeepDTA: deep drug--target binding affinity prediction},
  author={{\"O}zt{\"u}rk, Hakime and {\"O}zg{\"u}r, Arzucan and Ozkirimli, Elif},
  journal={Bioinformatics},
  volume={34},
  number={17},
  pages={i821--i829},
  year={2018},
  publisher={Oxford University Press}
}
```
