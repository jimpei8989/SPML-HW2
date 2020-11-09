from pytorchcv.model_provider import get_model


def build_model(name):
    return get_model(name + "_cifar10", pretrained=True)
