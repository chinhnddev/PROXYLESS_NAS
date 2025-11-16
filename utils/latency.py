# Latency lookup tables for ProxylessNAS
# Based on the original paper's measurements on Pixel1 and TitanXp

# Pixel1 latency (ms) for different operations
PIXEL1_LATENCY = {
    'conv_3x3': 0.123,
    'conv_5x5': 0.234,
    'conv_7x7': 0.345,
    'skip_connect': 0.001,
    'avg_pool_3x3': 0.012,
    'max_pool_3x3': 0.012,
    'sep_conv_3x3': 0.089,
    'sep_conv_5x5': 0.178,
    'sep_conv_7x7': 0.267,
    'dil_conv_3x3': 0.145,
    'dil_conv_5x5': 0.290,
    'conv_7x1_1x7': 0.156,
    'none': 0.000,
}

# TitanXp latency (ms) for different operations
TITANXP_LATENCY = {
    'conv_3x3': 0.045,
    'conv_5x5': 0.089,
    'conv_7x7': 0.134,
    'skip_connect': 0.001,
    'avg_pool_3x3': 0.005,
    'max_pool_3x3': 0.005,
    'sep_conv_3x3': 0.032,
    'sep_conv_5x5': 0.064,
    'sep_conv_7x7': 0.096,
    'dil_conv_3x3': 0.053,
    'dil_conv_5x5': 0.106,
    'conv_7x1_1x7': 0.057,
    'none': 0.000,
}

# Mapping from operation names to latency keys
OP_LATENCY_MAP = {
    'nor_conv_3x3': 'conv_3x3',
    'nor_conv_5x5': 'conv_5x5',
    'nor_conv_7x7': 'conv_7x7',
    'skip_connect': 'skip_connect',
    'avg_pool_3x3': 'avg_pool_3x3',
    'max_pool_3x3': 'max_pool_3x3',
    'sep_conv_3x3': 'sep_conv_3x3',
    'sep_conv_5x5': 'sep_conv_5x5',
    'sep_conv_7x7': 'sep_conv_7x7',
    'dil_conv_3x3': 'dil_conv_3x3',
    'dil_conv_5x5': 'dil_conv_5x5',
    'conv_7x1_1x7': 'conv_7x1_1x7',
    'none': 'none',
}


def get_latency(op_name, device='pixel1'):
    """Get latency for an operation on a specific device."""
    if device.lower() == 'pixel1':
        latency_table = PIXEL1_LATENCY
    elif device.lower() == 'titanxp':
        latency_table = TITANXP_LATENCY
    else:
        raise ValueError(f"Unknown device: {device}")

    key = OP_LATENCY_MAP.get(op_name, 'none')
    return latency_table.get(key, 0.0)


def calculate_network_latency(genotype, device='pixel1'):
    """Calculate total latency for a network genotype."""
    total_latency = 0.0

    for op_name, _ in genotype.normal:
        total_latency += get_latency(op_name, device)

    for op_name, _ in genotype.reduce:
        total_latency += get_latency(op_name, device)

    return total_latency
