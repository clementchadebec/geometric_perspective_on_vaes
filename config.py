from dataclasses import dataclass

@dataclass
class VAE_config:
    input_dim: int =  None
    model_name: str = "VAE"
    architecture: str= 'convnet'
    n_channels: int = None
    latent_dim: int = None
    beta: float =1
    device: str = 'cuda'
    cuda: bool= True
    dynamic_binarization: bool= False
    dataset: str=None

@dataclass
class RHVAE_config:
    input_dim: int =  None
    model_name: str = "RHVAE"
    architecture: str= 'convnet'
    n_channels: int = None
    latent_dim: int = None
    beta: float =1
    n_lf: int =1
    eps_lf: float=0.001
    temperature: float = 0.8
    regularization: float = 0.001
    device: str = 'cuda'
    cuda: bool= True
    beta_zero: float = 0.3
    metric_fc: int = 400
    dynamic_binarization: bool= False
    dataset: str=None


@dataclass
class VAMP_config:
    model_name="VAMP"
    architecture: str= 'convnet'
    input_size: int = None
    z1_size: int = None
    n_channels: int = None
    prior: str = 'vampprior'
    input_type: str = 'continuous'
    use_training_data_init: bool = False
    pseudoinputs_mean: float = 0.05
    pseudoinputs_std: float = 0.01
    number_components: int = 10
    dataset: str=None
    dynamic_binarization = False
    warmup: int = 0
    beta: int = 1
    cuda = True
