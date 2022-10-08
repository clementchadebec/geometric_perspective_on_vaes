# This is the official implementation of ["A Geometric Perspective on Variational Autoencoders"](https://arxiv.org/abs/2209.07370) (NeurIPS 2022)

This code uses a version of **python3.6**. 

**Note**: The method should be soon added to [`pythae`](https://github.com/clementchadebec/benchmark_VAE).

To install requirement run

```bash 
pip install -r requirements.txt
```

## Data folders

The data must be located in `data_folders`:

### MNIST
The provided code requires a file `mnist_32x32.npz` to be located in `data_folders/mnist/`.
The data must be in the range [0, 255] and loadable as follows:
```python
    import numpy as np
    mnist_digits = np.load(args.path_to_train)
    train_data = mnist_digits['x_train'] # data of shape 60000x32x32x1 in [0-255]
    train_targets = mnist_digits['y_train'] # corresponding labels
```

In `data_folders/mnist/test_folder` must be located 10k test images in `.png` format used for metric
computation

### CIFAR10
The provided code requires a file `cifar_10.npz` to be located in `data_folders/cifar/`.
The data must be in the range of [0, 255] and lodable as follows:
```python
    import numpy as np
    cifar_data = np.load(args.path_to_train)
    train_data = cifar_data['x_train'] # data of shape 50000x32x32x3 in [0-255]
    train_targets = cifar_data['y_train'] # corresponding labels
```

In `data_folders/cifar/test_folder` must be located 10k test images in `.png` format used for metric
computation

### Celeba
The provided code requires a file `train_data.pt` to be located in `data_folders/celeba/`. The data 
must be a big tensor of shape n_samplesx3x64x64 in the range [0, 1] and loadable as follows:

```python
    import torch
    train_data = torch.load(os.path.join(args.path_to_train, 'train_data.pt')) # data of shape 162770x64x64x3 in the range of [0-1]
    val_data = torch.load(os.path.join(args.path_to_train, 'val_data.pt')) # data of shape 19867x64x64x3 in the range of [0-1]
```

In `data_folders/celeba/test/test` must be located the test images in `.png` format used for metric
computation

### SVHN

The provided code requires a file `train_32x32.mat` to be located in `data_folders/svhn/`.
The data must be in the rnage [0, 255] and loadable as follows:

```python
    from scipy.io import loadmat
    svnh_digits = loadmat(args.path_to_train)['X'] # data of shape 32x32x3x73257 in the range of [0-255]
    svnh_targets = loadmat(args.path_to_train)['y'] # corresponding labels
```

In `data_folders/svhn/test_folder` must be located the test images in `.png` format used for metric
computation.


### OASIS
The provided code requires a file `OASIS.npz` to be located in `data_folders/oasis/`. The data must be in the range of [0, 255] and you must ensure that each data image has a maximum voxel value of 255 and a minimum of 0. The data must be loadable as follows

```python
    import numpy as np
    oasis_data = np.load(args.path_to_train)
    train_data = oasis_data['x_train'] # data of shape 416x208x176x1 in the range of [0-255]
    train_targets = torch.tensor(oasis_data['y_train'] # corresponding targets
```

## Performing experiments

The commandines to train a model, generate new data and compute the metrics are available in 
`models_to_train.sh`.


## Reference

```bibtex
@article{chadebec2022geometric,
  title={A Geometric Perspective on Variational Autoencoders},
  author={Chadebec, Cl{\'e}ment and Allassonni{\`e}re, St{\'e}phanie},
  journal={arXiv preprint arXiv:2209.07370},
  year={2022}
}
```
