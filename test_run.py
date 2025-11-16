import torch
import torch.nn as nn
import torch.optim as optim

from models import ProxylessNAS
from utils import get_cifar10
from utils.utils import accuracy, AverageMeter


def main():
    print("Testing ProxylessNAS implementation...")

    # Model setup
    model = ProxylessNAS(C=36, num_classes=10, layers=20)
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU")
    else:
        print("Using CPU")

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        criterion = criterion.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)

    # Data setup
    train_loader, test_loader = get_cifar10(batch_size=128, num_workers=0)  # num_workers=0 for testing

    # Training loop for 10 epochs
    model.train()
    for epoch in range(10):
        print(f"Epoch {epoch + 1}/10")
        train_loss = AverageMeter()
        train_acc = AverageMeter()

        for i, (inputs, targets) in enumerate(train_loader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            prec1, _ = accuracy(outputs, targets, topk=(1, 5))
            train_loss.update(loss.item(), inputs.size(0))
            train_acc.update(prec1.item(), inputs.size(0))

            if i % 10 == 0:
                print(f"  Step {i}: Loss {train_loss.avg:.4f}, Acc {train_acc.avg:.2f}%")

        print(f"Epoch {epoch + 1} completed: Loss {train_loss.avg:.4f}, Acc {train_acc.avg:.2f}%")

    # Test inference
    print("Testing inference...")
    model.eval()
    test_loss = AverageMeter()
    test_acc = AverageMeter()

    with torch.no_grad():
        for inputs, targets in test_loader:
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1, _ = accuracy(outputs, targets, topk=(1, 5))
            test_loss.update(loss.item(), inputs.size(0))
            test_acc.update(prec1.item(), inputs.size(0))

    print(f"Test completed: Loss {test_loss.avg:.4f}, Acc {test_acc.avg:.2f}%")
    print("Test run successful!")


if __name__ == '__main__':
    main()
