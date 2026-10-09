from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)

weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1

model = mobilenet_v3_small(weights=weights)

print("\nАрхитектура: MobileNetV3 Small")
print("Предобученные веса:", weights.name)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Количество параметров:", parameter_count)
print("Исходное число классов:", model.classifier[-1].out_features)
print("\nКлассификатор:")
print(model.classifier)