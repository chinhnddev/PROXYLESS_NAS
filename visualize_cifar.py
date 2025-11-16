import os
import sys
import torch
from graphviz import Digraph

from models import ProxylessNAS
from genotypes import PROXYLESSNAS_CIFAR10


def plot(genotype, file_path, caption=None):
    """Plot the architecture of the genotype."""
    g = Digraph(
        format='png',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype['normal']) % 2 == 0
    steps = len(genotype['normal']) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2*i, 2*i + 1]:
            op, j = genotype['normal'][k]
            if op != 'none':
                g.edge(str(j), str(i), label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.attr(label=caption)
    g.render(file_path, view=False)


def main():
    if len(sys.argv) != 2:
        print("usage: python visualize_cifar.py <save_path>")
        sys.exit(1)

    save_path = sys.argv[1]

    genotype = PROXYLESSNAS_CIFAR10

    plot(genotype, save_path, "ProxylessNAS CIFAR-10")


if __name__ == '__main__':
    main()
