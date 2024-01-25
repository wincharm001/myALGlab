translate = {
    "cane": "dog", 
    "cavallo": "horse", 
    "elefante": "elephant", 
    "farfalla": "butterfly", 
    "gallina": "chicken",
    "gatto": "cat", 
    "mucca": "cow", 
    "pecora": "sheep", 
    "scoiattolo": "squirrel",
    "ragno": "spider"
    }

import os

path = '../../../DATASETS/animal10_classification/raw-img/'
src_names = os.listdir(path)

for src_name in src_names:
    src_path = path + src_name
    os.rename(src_path, path + f'{translate[src_name]}')

