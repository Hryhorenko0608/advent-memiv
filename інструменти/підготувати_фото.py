# Готує фото від Тетяни для сайту: обрізає прозорі поля, стискає, записує розміри й позиції.
# Запуск (з папки «Адвент мемів»): python інструменти/підготувати_фото.py <папка_з_картинками>
import json, os, sys
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = sys.argv[1]
OUT = os.path.join(ROOT, 'фото')
ORIG = os.path.join(OUT, 'оригінали', 'вирізані')
MEMES = os.path.join(ROOT, 'меми')
os.makedirs(ORIG, exist_ok=True); os.makedirs(MEMES, exist_ok=True)

# номер файлу в чаті → назва (усі вирізані на однаковому полотні, як оригінали)
CUT = {16: 'турка', 17: 'какао_поверхня', 18: 'маршмелоу', 19: 'ложка', 20: 'чашка',
       21: 'мандаринка_почищена', 22: 'мандаринка_ціла', 23: 'пледи'}
SPLIT = {'маршмелоу': 5, 'пледи': 3}   # скільки окремих предметів на кадрі
MEME_FILES = list(range(24, 35))        # меми по порядку днів


def find(n):
    for ext in ('webp', 'png', 'jpg'):
        p = os.path.join(SRC, f'{n}.{ext}')
        if os.path.exists(p):
            return p


def save_crop(im, box, name):
    x0, y0, x1, y1 = box
    part = im.crop((x0, y0, x1, y1))
    part.save(os.path.join(OUT, name + '.webp'), quality=90, method=6)
    return dict(file=name + ".webp", canvas=list(im.size), box=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)])


geo = {}
for n, name in CUT.items():
    p = find(n)
    im = Image.open(p).convert('RGBA')
    im.save(os.path.join(ORIG, name + '.png'))
    a = np.asarray(im)[..., 3] > 12
    if name in SPLIT:
        lab, k = ndi.label(ndi.binary_closing(a, iterations=3))
        sizes = ndi.sum(a, lab, range(1, k + 1))
        keep = sorted(np.argsort(sizes)[::-1][:SPLIT[name]], key=lambda i: ndi.find_objects(lab)[i][1].start)
        geo[name] = []
        for j, idx in enumerate(keep):
            sl = ndi.find_objects(lab)[idx]
            mask = (lab == idx + 1)
            arr = np.asarray(im).copy(); arr[..., 3] = np.where(mask, arr[..., 3], 0)
            box = (sl[1].start - 4, sl[0].start - 4, sl[1].stop + 4, sl[0].stop + 4)
            geo[name].append(save_crop(Image.fromarray(arr), box, f'{name}_{j + 1}'))
    else:
        ys, xs = np.nonzero(a)
        geo[name] = save_crop(im, (xs.min() - 4, ys.min() - 4, xs.max() + 5, ys.max() + 5), name)
    print('Готово:', name)

for day, n in enumerate(MEME_FILES, 1):
    im = Image.open(find(n)).convert('RGB')
    im.thumbnail((1200, 1200))
    im.save(os.path.join(MEMES, f'{day:02d}.jpg'), quality=88, optimize=True)
print('Мемів:', len(MEME_FILES))

with open(os.path.join(OUT, 'геометрія.json'), 'w', encoding='utf-8') as f:
    json.dump(geo, f, ensure_ascii=False, indent=1)
print(json.dumps(geo, ensure_ascii=False))
