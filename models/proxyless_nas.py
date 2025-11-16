import torch
import torch.nn as nn
import torch.nn.functional as F
from .cells import ProxylessCell
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genotypes import PROXYLESSNAS_CIFAR10
from .operations import OPS


class ProxylessNAS(nn.Module):
    def __init__(self, C=36, num_classes=10, layers=20, genotype=None, stem_multiplier=3):
        super(ProxylessNAS, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers

        if genotype is None:
            genotype = PROXYLESSNAS_CIFAR10

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = ProxylessCell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # Initialize architecture parameters for search
        self._initialize_alphas()

    def _initialize_alphas(self):
        # For search mode, we need alphas for each cell
        # But for fixed genotype, we don't use them
        pass

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def arch_parameters(self):
        # For search mode
        return []

    def genotype(self):
        return PROXYLESSNAS_CIFAR10


# Searchable version for NAS
class ProxylessNASNetwork(nn.Module):
    def __init__(self, C=36, num_classes=10, layers=20, criterion=None, steps=4, multiplier=4, stem_multiplier=3):
        super(ProxylessNASNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # Initialize architecture parameters
        self._initialize_alphas()

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for j in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.Parameter(1e-3 * torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(1e-3 * torch.randn(k, num_ops))

        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def forward(self, input):
        s0 = s1 = self.stem(input)

        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        weights_reduce = F.softmax(self.alphas_reduce, dim=-1)

        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1, weights_normal, weights_reduce)

        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        def _parse(weights, normal=True):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False)

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype


# Genotype class
class Genotype:
    def __init__(self, normal, normal_concat, reduce, reduce_concat):
        self.normal = normal
        self.normal_concat = normal_concat
        self.reduce = reduce
        self.reduce_concat = reduce_concat


# Import PRIMITIVES
from .operations import PRIMITIVES
from .cells import Cell
