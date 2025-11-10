import os
import shutil
import csv
from tqdm import tqdm

INPUT_DIR = 'UTKFace'
OUTPUT_DIR = 'labeled/full/img'

CSV_PATH = 'labeled/full/label.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_data = [['name', 'age', 'gender']]

for idx, file in tqdm(enumerate(os.listdir(INPUT_DIR))):
    if file.endswith('.jpg'):
        try:
            infos = file.split('.')[0]
            infos = infos.split('_')

            age, gender = infos[0], infos[1]

            new_name = f'{age}_{gender}_{idx}.jpg'

            src = os.path.join(INPUT_DIR, file)
            dest = os.path.join(OUTPUT_DIR, new_name)
            shutil.copy2(src, dest)

            csv_data.append([new_name, age, gender])

            idx += 1

        except:
            print(f'error at {file}')

with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
    writter = csv.writer(f)
    writter.writerows(csv_data)