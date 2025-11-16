import os
import torch
import torch.onnx

from models import ProxylessNAS


def main():
    # Args
    class Args:
        def __init__(self):
            self.init_channels = 36
            self.layers = 20
            self.model_path = 'saved_models/best.pth.tar'
            self.onnx_path = 'proxylessnas.onnx'
            self.batch_size = 1
            self.input_size = (1, 3, 32, 32)  # CIFAR-10 input size

    args = Args()

    # Load model
    model = ProxylessNAS(args.init_channels, 10, args.layers)

    if os.path.isfile(args.model_path):
        print(f"Loading checkpoint from {args.model_path}")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model'])
        print("Model loaded successfully")
    else:
        print(f"No checkpoint found at {args.model_path}, using random weights")

    model.eval()

    # Create dummy input
    dummy_input = torch.randn(args.input_size)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        args.onnx_path,
        verbose=True,
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )

    print(f"Model exported to {args.onnx_path}")


if __name__ == '__main__':
    main()
