# Deploying Transformer and Bert Using Apache TVM

## About

CPU: AMD Ryzen 5600x

GPU: NVIDIA RTX 3070Ti

Python Version: 3.7

Pytorch Version: 1.8.0

TVM version: 0.8.dev


## Transformer
transformer (https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)

模型相关参数设置：
- d_model = 512 (feature size)
- nheader = 16
- num_encoder_layers = 12
- num_decoder_layers = 6
- 其他参数默认

输入设置参考：
- src = random tensor with shape (10,32,512) # (S,N,E)
- tgt = random tensor with shape (20,32,512) # (T,N,E)
- 其他参数默认

### Unoptimized Performance (in ms):

CPU版本 {'mean': 261.97880415005784, 'median': 261.69940710024093, 'std': 0.7671719541232292}

CUDA版本 {'mean': 75.53232621000012, 'median': 76.68678340000028, 'std': 3.5133938666919913}

优化使用TVM提供的AutoTVM
- tuner = XGBTuner # CPU verison use RandomTuner
- n_trails = 2000 # 搜索的尝试次数，越大越好。CPU推荐1500，GPU推荐3000-4000。
- early_stopping = 600 # 如果累计600次没超过历史性能最高值，就结束当前搜索
- 以上是影响最大的因素

### Optimized Performance (in ms):

CPU版本 {'mean': 95.10979770999256, 'median': 95.10866304990486, 'std': 0.06900490877264685}

CUDA版本 {'mean': 36.3591017899995, 'median': 36.77307985000198, 'std': 1.2366104747615936}

---

## Bert
Bert-base-uncased (https://huggingface.co/bert-base-uncased)

模型相关参数设置：
- hidden_size = 768
- num_hidden_layers = 12
- num_attention_heads = 12
- (其实全部默认参数)

输入设置参考：
- batchsize = 1
- seq_len = 512 

### Unoptimized Performance (in ms):
CPU版本 {'mean': 1323.46220884996, 'median': 1320.1354465498298, 'std': 11.709776383482959}

CUDA版本 {'mean': 242.44719299003918, 'median': 249.41605134999918, 'std': 20.962010872337736}

优化使用TVM提供的AutoTVM
- tuner = XGBTuner # CPU version use RandomTuner
- n_trails = 2000 # 搜索的尝试次数，越大越好。CPU推荐1500，GPU推荐3000-4000。
- early_stopping = 600 # 如果累计600次没超过历史性能最高值，就结束当前搜索
- 以上是影响最大的因素

### Optimized Performance (in ms):

CPU版本 {'mean': 339.4948378801928, 'median': 339.75638930023706, 'std': 1.0983344402061206}

CUDA版本 {'mean': 169.0238639299787, 'median': 169.95936239982257, 'std': 2.988801769889063}



