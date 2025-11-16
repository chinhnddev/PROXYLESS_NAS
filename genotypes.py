# ProxylessNAS genotypes
# Official genotype for CIFAR-10 (2.08% error)

from collections.abc import Iterable  # Fixed for PyTorch 2.x

# ProxylessNAS genotype for CIFAR-10
# From the paper: Han Cai et al., "ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware"
# ICLR 2019

class Genotype:
    def __init__(self, normal, normal_concat, reduce, reduce_concat):
        self.normal = normal
        self.normal_concat = normal_concat
        self.reduce = reduce
        self.reduce_concat = reduce_concat

PROXYLESSNAS_CIFAR10 = Genotype(
    normal=[
        ('skip_connect', 1),
        ('nor_conv_3x3', 0),
        ('nor_conv_3x3', 0),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('nor_conv_3x3', 0),
        ('avg_pool_3x3', 0),
        ('nor_conv_3x3', 1),
        ('nor_conv_3x3', 0),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('nor_conv_3x3', 1),
        ('nor_conv_3x3', 0),
        ('avg_pool_3x3', 0),
        ('nor_conv_3x3', 1),
        ('nor_conv_3x3', 1),
        ('nor_conv_3x3', 0),
        ('skip_connect', 1),
        ('nor_conv_3x3', 0),
    ],
    normal_concat=[2, 3, 4, 5],
    reduce=[
        ('nor_conv_3x3', 0),
        ('nor_conv_3x3', 1),
        ('skip_connect', 0),
        ('skip_connect', 1),
        ('avg_pool_3x3', 0),
        ('nor_conv_3x3', 1),
        ('nor_conv_3x3', 0),
        ('skip_connect', 1),
        ('skip_connect', 0),
        ('avg_pool_3x3', 0),
        ('nor_conv_3x3', 1),
        ('nor_conv_3x3', 0),
        ('avg_pool_3x3', 0),
        ('nor_conv_3x3', 1),
        ('nor_conv_3x3', 1),
        ('nor_conv_3x3', 0),
        ('skip_connect', 1),
        ('nor_conv_3x3', 0),
        ('avg_pool_3x3', 0),
        ('nor_conv_3x3', 1),
    ],
    reduce_concat=[2, 3, 4, 5],
)

# Operation names - will be imported from operations.py
# This is just a placeholder; actual OPS are defined in models/operations.py

# Note: The actual operation classes are defined in models/operations.py
