import os
import shutil
import torch
import clip
from PIL import Image
import time


source_folder = "body_type/random_imgs"
destination_folder = "body_type/classified"

if os.path.exists(destination_folder) and os.path.isdir(destination_folder):
    os.rmdir(destination_folder)

# Инициализируем устройство
device = "cpu" # cuda
model, preprocess = clip.load("ViT-B/32", device=device)

# Список типов кузовов для классификации
body_types = [
    "Sedan",      # Седан
    "Hatchback",  # Хэтчбек
    "SUV",        # Внедорожник (Sport Utility Vehicle)
    "Wagon",      # Универсал
    "Coupe",      # Купе
    "Minivan",    # Минивэн
    "Pickup",     # Пикап
    "Van",        # Фургон
    "Convertible",# Кабриолет
    "Roadster"    # Родстер
]

# Создаём папки для каждого типа кузова, если они не существуют
for body_type in body_types:
    os.makedirs(os.path.join(destination_folder, body_type), exist_ok=True)

# Функция для классификации изображения
def classify_image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Преобразуем типы кузовов в текстовые эмбеддинги
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {body_type}") for body_type in body_types]).to(device)

    # Получаем эмбеддинги изображения и текста
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_inputs)

        # Нормализация и вычисление сходства
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Получаем наиболее похожий тип кузова
        values, indices = similarity[0].topk(1)
        most_similar_body_type = body_types[indices.item()]

    return most_similar_body_type

# Начало измерения времени
start_time = time.time()

# Обрабатываем все изображения в папке
files = os.listdir(source_folder)
total_files = len(files)
times = []

for filename in files:
    file_path = os.path.join(source_folder, filename)
    if os.path.isfile(file_path):
        # Измеряем время классификации изображения
        image_start_time = time.time()

        # Классифицируем изображение
        body_type = classify_image(file_path)

        # Копируем изображение в соответствующую папку
        dest_path = os.path.join(destination_folder, body_type, filename)
        shutil.copy(file_path, dest_path)

        # Измеряем время завершения обработки изображения
        image_end_time = time.time()
        image_processing_time = image_end_time - image_start_time
        times.append(image_processing_time)

        print(f"Copied {filename} to {body_type} (Processing Time: {image_processing_time:.4f} seconds)")

# Конец измерения времени
end_time = time.time()
total_time = end_time - start_time
average_time_per_image = total_time / total_files

print("\nProcessing complete!")
print(f"Total number of images: {total_files}")
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Average time per image: {average_time_per_image:.4f} seconds")
