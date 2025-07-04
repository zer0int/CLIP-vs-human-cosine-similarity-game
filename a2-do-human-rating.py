import PySimpleGUI as sg
from PIL import Image
import io
import json

def pil_to_data(im, maxsize=(256,256)):
    im = im.convert("RGB")
    im.thumbnail(maxsize)
    with io.BytesIO() as output:
        im.save(output, format="PNG")
        return output.getvalue()

with open('fixed_image_pairs.json', 'r') as f:
    pairs = json.load(f)

ratings = []
for i, (img1_path, img2_path) in enumerate(pairs):
    try:
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        data1 = pil_to_data(img1)
        data2 = pil_to_data(img2)
    except Exception as e:
        print(f"Error loading {img1_path} or {img2_path}: {e}")
        continue

    layout = [
        [sg.Image(data=data1, key='-IMG1-'), sg.Image(data=data2, key='-IMG2-')],
        [sg.Text('Similarity:'), sg.Slider(range=(0, 1), resolution=0.01, orientation='h', key='-SLIDER-', size=(40,15))],
        [sg.Button('OK')]
    ]
    window = sg.Window(f'Pair {i+1}/100', layout)
    event, values = window.read()
    if event == 'OK':
        ratings.append({'img1': img1_path, 'img2': img2_path, 'rating': values['-SLIDER-']})
    window.close()

with open('human_similarity_ratings.json', 'w') as f:
    json.dump(ratings, f)
