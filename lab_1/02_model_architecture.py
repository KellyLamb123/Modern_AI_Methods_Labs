import torch

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("=== Вычислительное устройство ===")
print("Device:", device)

print("\n=== Загрузка нейросети ===")

weights = MobileNet_V3_Small_Weights.DEFAULT

model = mobilenet_v3_small(
    weights=weights
)

model = model.to(device)

model.eval()

print("Архитектура: MobileNetV3 Small")
print("Предобученные веса: ImageNet-1K")
print("Модель успешно загружена!")

print("\n=== Структура MobileNetV3 Small ===")

print("\nБазовая сеть (Feature Extractor):")
print(type(model.features).__name__)

print("\nКоличество блоков базовой сети:")
print(len(model.features))

print("\nКлассификатор:")
print(model.classifier)

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Всего параметров: {total_params:,}")
print(f"Обучаемых параметров: {trainable_params:,}")

print("\n=== Тестирование нейросети ===")

test_image = torch.randn(
    1, 3, 224, 224,
    device=device
)

print("Размер входных данных:", test_image.shape)

with torch.inference_mode():
    output = model(test_image)

print("Размер выходных данных:", output.shape)