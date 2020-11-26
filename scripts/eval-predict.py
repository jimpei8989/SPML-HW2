import sys

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def main():
    output_txt = sys.argv[1]

    with open(output_txt) as f:
        predictions = f.readlines()[:-1]

    acc = 0
    for i, p in enumerate(predictions):
        acc += 1 if p.strip() == CIFAR10_CLASSES[i // 10] else 0

    print(f"Acc: {acc / 100:.2f}")


main()