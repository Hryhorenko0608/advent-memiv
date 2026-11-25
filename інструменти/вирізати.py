# Вирізає предмети з однотонного сірого фону і зберігає їх як PNG/WebP з прозорістю.
# Запуск: python інструменти/вирізати.py   (з папки «Адвент мемів»)
# Результат: фото/*.webp і фото/геометрія.json (розміри й центри для сторінки).
import json, os
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'фото', 'оригінали')
OUT = os.path.join(ROOT, 'фото')

# назва файлу: поріг відмінності кольору від фону, скільки предметів лишити, пороги яскравості
JOBS = {
    'мандаринка_ціла.jpg':     dict(t=32, keep=1, bright=9),
    'мандаринка_почищена.jpg': dict(t=12, keep=1, bright=1.04, minr=0.7, strong=60, close=10),
    'чашка_порожня.jpg':       dict(t=40, keep=1, bright=9),
    'ложка.jpg':               dict(t=14, keep=1, bright=9, dark=0.62, close=4, refine=True),
}


def background(img):
    """Моделює фон плавною поверхнею за пікселями з країв кадру."""
    h, w, _ = img.shape
    band = 24
    m = np.zeros((h, w), bool)
    m[:band] = m[-band:] = True
    m[:, :band] = m[:, -band:] = True
    yy, xx = np.mgrid[0:h, 0:w]
    x, y = xx / w, yy / h
    A = np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], -1)
    bg = np.zeros_like(img)
    for c in range(3):
        coef, *_ = np.linalg.lstsq(A[m], img[..., c][m], rcond=None)
        bg[..., c] = A @ coef
    return bg


def local_background(im, bg):
    """Точніший фон: розмиваємо кадр, ігноруючи все, що схоже на предмет або тінь."""
    rough = np.linalg.norm(im - bg, axis=-1) > 18
    w = (~ndi.binary_dilation(rough, iterations=25)).astype(np.float32)
    out = np.zeros_like(im)
    den = ndi.gaussian_filter(w, 60) + 1e-4
    for c in range(3):
        out[..., c] = ndi.gaussian_filter(im[..., c] * w, 60) / den
    return np.where(den[..., None] > 0.02, out, bg)


def cut(name, t, keep, bright=1.07, dark=0.0, pink=False, minr=0.0, strong=999, close=4, refine=False):
    im = np.asarray(Image.open(os.path.join(SRC, name)).convert('RGB')).astype(np.float32)
    bg = background(im)
    if refine:
        bg = local_background(im, bg)
    l = im.mean(-1) + 1e-3
    bg_l = bg.mean(-1) + 1e-3
    ratio = l / bg_l                                   # яскравість відносно фону
    cdiff = np.linalg.norm(im / l[..., None] - bg / bg_l[..., None], axis=-1) * 100  # відмінність кольору
    # тінь має той самий відтінок, що й фон, лише темніша, тому її не беремо
    mask = ((cdiff > t) & (ratio > minr)) | (cdiff > strong) | (ratio > bright) | (ratio < dark)
    mask = ndi.binary_opening(mask, iterations=2)
    mask = ndi.binary_closing(mask, iterations=close)
    mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask)
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    if pink:  # лишаємо тільки рожеві шматки (білі зробимо з рожевих фільтром на сторінці)
        redness = ndi.mean(im[..., 0] - im[..., 1], lab, range(1, n + 1))
        sizes = np.where(redness > 12, sizes, 0)
    order = [i for i in np.argsort(sizes)[::-1][:keep] if sizes[i] > 2000]
    results = []
    for k, idx in enumerate(sorted(order, key=lambda i: ndi.find_objects(lab)[i][1].start)):
        part = lab == idx + 1
        part = ndi.binary_erosion(part, iterations=1)
        alpha = ndi.gaussian_filter(part.astype(np.float32), 1.1)
        sl = ndi.find_objects(part.astype(int))[0]
        pad = 6
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, im.shape[0])
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, im.shape[1])
        rgba = np.dstack([im, alpha * 255]).clip(0, 255).astype(np.uint8)[y0:y1, x0:x1]
        base = os.path.splitext(name)[0] + (f'_{k + 1}' if keep > 1 else '')
        Image.fromarray(rgba, 'RGBA').save(os.path.join(OUT, base + '.webp'), quality=90, method=6)
        # «тіло» без зелені (для мандаринки: без листочка)
        sub = im[y0:y1, x0:x1]
        body = part[y0:y1, x0:x1] & ~((sub[..., 1] > sub[..., 0]) & (sub[..., 1] > sub[..., 2]))
        ys, xs = np.nonzero(body)
        results.append(dict(file=base + '.webp', box=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                            body=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]))
    return results


# Маршмелоу надто бліді для автоматики: контури позначені вручну (координати на оригіналі 1376×768)
MARSH = {
    'маршмелоу_1': [(335,148),(400,150),(470,180),(500,205),(507,240),(500,275),(480,310),(445,335),(415,340),(360,325),(312,295),(293,262),(292,225),(305,180)],
    'маршмелоу_2': [(650,282),(710,276),(760,290),(790,320),(800,360),(790,400),(760,430),(700,455),(670,473),(640,472),(600,455),(578,420),(575,375),(590,340),(620,305)],
    'маршмелоу_3': [(930,440),(990,418),(1040,425),(1080,455),(1097,500),(1093,560),(1070,600),(1030,625),(990,642),(950,640),(900,610),(865,575),(858,535),(870,490),(895,460)],
    'маршмелоу_4': [(790,105),(850,83),(905,85),(935,110),(948,150),(945,195),(925,225),(880,245),(825,268),(790,262),(755,240),(738,205),(740,160),(760,125)],
    'маршмелоу_5': [(290,425),(345,407),(395,405),(430,425),(455,460),(462,505),(452,545),(420,575),(370,595),(320,615),(290,612),(255,585),(232,545),(228,500),(245,460)],
}


def cut_polygons():
    """Вирізає маршмелоу за ручними контурами, з м'яким краєм."""
    from PIL import ImageDraw
    src = Image.open(os.path.join(SRC, 'маршмелоу.jpg')).convert('RGB')
    out = []
    for base, pts in MARSH.items():
        m = Image.new('L', src.size, 0)
        ImageDraw.Draw(m).polygon(pts, fill=255)
        a = ndi.gaussian_filter(np.asarray(m, np.float32), 4)
        a = np.clip((a - 110) * 2.2, 0, 255)          # згладжені кути
        a = ndi.gaussian_filter(a, 1.0)
        xs, ys = zip(*pts)
        x0, y0, x1, y1 = min(xs) - 8, min(ys) - 8, max(xs) + 8, max(ys) + 8
        rgba = np.dstack([np.asarray(src), a]).astype(np.uint8)[y0:y1, x0:x1]
        Image.fromarray(rgba, 'RGBA').save(os.path.join(OUT, base + '.webp'), quality=90, method=6)
        out.append(dict(file=base + '.webp', box=[x0, y0, x1 - x0, y1 - y0]))
    return out


geo = {'маршмелоу': cut_polygons()}
for name, cfg in JOBS.items():
    geo[os.path.splitext(name)[0]] = cut(name, **cfg)
    print('Готово:', name, geo[os.path.splitext(name)[0]])
with open(os.path.join(OUT, 'геометрія.json'), 'w', encoding='utf-8') as f:
    json.dump(geo, f, ensure_ascii=False, indent=1)


# Ложка: прибираємо сіру тінь знизу, лишаючи тільки контур ложки (координати на вирізаній ложці)
def clean_spoon():
    from PIL import ImageDraw
    p = os.path.join(OUT, 'ложка.webp')
    im = Image.open(p).convert('RGBA')
    m = Image.new('L', im.size, 0)
    d = ImageDraw.Draw(m)
    d.ellipse((8, 8, 330, 205), fill=255)
    d.polygon([(300, 72), (380, 90), (640, 96), (660, 78), (760, 58), (870, 52), (935, 62), (968, 95), (958, 132),
               (915, 162), (860, 172), (780, 162), (700, 136), (640, 126), (380, 126), (300, 142)], fill=255)
    a = np.minimum(np.asarray(im)[..., 3], ndi.gaussian_filter(np.asarray(m, np.float32), 1.5)).astype(np.uint8)
    arr = np.asarray(im).copy(); arr[..., 3] = a
    Image.fromarray(arr, 'RGBA').save(p, quality=90, method=6)


clean_spoon()
