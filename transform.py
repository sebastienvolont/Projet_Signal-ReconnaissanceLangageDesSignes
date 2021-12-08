import glob
import os
from pathlib import Path

from PIL import Image



files = list(glob.glob('img/*/*.jpg', recursive=True))
print("go")
for i, filename in enumerate(files):
    print(f'File {i}/{len(files)}', end='\r')
    _, folder, fn = filename.split('\\')
    dest = Path('img2') / folder / fn
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(filename).convert('L')
    img.resize((32, 32)).save(dest)

print()
