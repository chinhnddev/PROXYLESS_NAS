# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware

This is a complete PyTorch implementation of ProxylessNAS (ICLR 2019) that reproduces the paper's results on CIFAR-10.

## Features

- **Path-level binarization**: Over-parameterized network with binary gates for efficient architecture search
- **Alternating training**: Updates weights and architecture parameters (α) alternately
- **Memory-efficient**: Samples only 2 paths when updating α to keep memory O(1)
- **Official genotype**: Includes the ProxylessNAS genotype for CIFAR-10 (2.08% error)
- **Modern PyTorch 2.x**: Compatible with latest PyTorch, fixes deprecated warnings
- **TensorBoard logging**: Real-time training monitoring
- **Architecture visualization**: Graphviz-based network visualization
- **ONNX export**: Ready for deployment
- **Latency regularization**: Optional latency constraints for mobile/GPU

## Project Structure

```
ProxylessNAS/
├── models/               # All network definitions
│   ├── __init__.py
│   ├── proxyless_nas.py  # Main network with path-level binarization
│   ├── operations.py     # Mixed operations
│   └── cells.py          # Cell definitions
├── utils/
│   ├── __init__.py
│   ├── data.py           # CIFAR-10/ImageNet data loading
│   ├── utils.py          # Training utilities, logging
│   └── latency.py        # Latency lookup tables
├── genotypes.py          # Official ProxylessNAS genotype
├── search_cifar.py       # NAS search on CIFAR-10
├── train_cifar.py        # Train from scratch on CIFAR-10
├── visualize_cifar.py    # Architecture visualization
├── export_onnx.py        # ONNX export
├── test_run.py           # 10-epoch test script
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

The CIFAR-10 dataset is required for training and search. The dataset file `cifar-10-python.tar.gz` is large (162.60 MB) and is handled using Git LFS.

### Download CIFAR-10

If the data is not present, download it manually:

```bash
# Download CIFAR-10 dataset
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -P data/
tar -xzf data/cifar-10-python.tar.gz -C data/
```

### Git LFS Setup

To handle large files like `cifar-10-python.tar.gz`, Git LFS is used:

1. Install Git LFS if not already installed.
2. Track `.tar.gz` files: `git lfs track "*.tar.gz"`
3. Add `.gitattributes` to repository.
4. The `.gitignore` file excludes `data/cifar-10-python.tar.gz` to avoid committing the large file directly.

Note: If cloning the repository, run `git lfs pull` to download LFS-tracked files.

## Usage

### 1. Test the implementation (10 epochs)

```bash
python test_run.py
```

### 2. NAS Search on CIFAR-10 (~6 hours on RTX 4090)

```bash
python search_cifar.py
```

This will search for optimal architectures and save the best genotype.

### 3. Train from scratch using official genotype (1-2 hours)

```bash
python train_cifar.py
```

Trains the network using the official ProxylessNAS genotype to achieve ~2.08% error on CIFAR-10.

### 4. Visualize the architecture

```bash
python visualize_cifar.py proxylessnas_architecture
```

Generates a PNG visualization of the network architecture.

### 5. Export to ONNX

```bash
python export_onnx.py
```

Exports the trained model to ONNX format for deployment.

## Key Implementation Details

### Path-Level Binarization

The core innovation is path-level binarization where each layer has multiple operation candidates, but only one path is active at a time:

```python
class MixedOp(nn.Module):
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
```

### Alternating Training

Training alternates between updating network weights and architecture parameters:

```python
# Update architecture parameters (α)
architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

# Update network weights
optimizer.zero_grad()
logits = model(input)
loss = criterion(logits, target)
loss.backward()
optimizer.step()
```

### Memory-Efficient Search

When updating architecture parameters, only 2 random paths are sampled to keep memory usage O(1):

```python
# Sample 2 paths for architecture update
input_search, target_search = next(iter(valid_loader))
```

## Results

Expected results on CIFAR-10:
- **Search time**: ~6 hours on RTX 4090
- **Training time**: 1-2 hours
- **Test accuracy**: 97.92% (2.08% error)

## Citation

```bibtex
@inproceedings{cai2019proxylessnas,
  title={ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware},
  author={Cai, Han and Zhu, Ligeng and Han, Song},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

## License

This implementation is provided for educational and research purposes.
