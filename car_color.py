import os
import shutil
import torch
import clip
from PIL import Image
import time


device = "cpu" # cuda
model, preprocess = clip.load("ViT-B/32", device=device)

# Define the colors
colors = [
    "Silver",     # Серебристый
    "Black",      # Черный
    "White",      # Белый
    "Gray",       # Серый
    "Beige",      # Бежевый
    "Turquoise",  # Бирюзовый
    "Burgundy",   # Бордовый
    "Bronze",     # Бронза
    "Cherry",     # Вишня
    "Sky Blue",   # Голубой
    "Yellow",     # Жёлтый
    "Green",      # Зеленый
    "Golden",     # Золотистый
    "Brown",      # Коричневый
    "Red",        # Красный
    "Orange",     # Оранжевый
    "Pink",       # Розовый
    "Blue",       # Синий
    "Lilac",      # Сиреневый
    "Purple",     # Фиолетовый
    "Chameleon",  # Хамелеон
    "Eggplant"    # Баклажан
]

# Define paths
image_folder = "color/random_imgs"
output_folder = "color/classified"

if os.path.exists(output_folder) and os.path.isdir(output_folder):
    os.rmdir(output_folder)

# Ensure output folders exist
for color in colors:
    os.makedirs(os.path.join(output_folder, color), exist_ok=True)

# Start measuring time
start_time = time.time()

# Process images
image_files = os.listdir(image_folder)
total_images = len(image_files)
times = []

for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    if os.path.isfile(img_path):
        # Measure time for each image
        image_start_time = time.time()

        image = Image.open(img_path)
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Prepare color names for CLIP
        text_inputs = torch.cat([clip.tokenize(f"car of color {color}") for color in colors]).to(device)

        # Calculate image features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        # Find the best matching color
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        best_color_idx = similarity.argmax().item()
        best_color = colors[best_color_idx]

        # Copy the image to the corresponding color folder
        dest_folder = os.path.join(output_folder, best_color)
        shutil.copy(img_path, dest_folder)

        # Measure time for the image prediction
        image_end_time = time.time()
        image_processing_time = image_end_time - image_start_time
        times.append(image_processing_time)

        print(f"Copied {img_name} to {best_color} (Processing Time: {image_processing_time:.4f} seconds)")

# End measuring time
end_time = time.time()
total_time = end_time - start_time
average_time_per_image = total_time / total_images

print(f"\nClassification and copying complete!")
print(f"Total number of images: {total_images}")
print(f"Total time taken: {total_time:.2f} seconds")
print(f"Average time per image: {average_time_per_image:.4f} seconds")
