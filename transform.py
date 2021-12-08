import glob
import os
from pathlib import Path

from PIL import Image

files = list(glob.glob('img/*/*.jpg', recursive=True))
for i, filename in enumerate(files):
    if i % 100 == 0:
        print(f'Files {i}/{len(files)}')

    _, folder, fn = filename.split(os.path.sep)
    dest = Path('dataset') / folder / fn
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(filename).convert('L')
    img.resize((32, 32)).save(dest)

print('Finished processing files')
