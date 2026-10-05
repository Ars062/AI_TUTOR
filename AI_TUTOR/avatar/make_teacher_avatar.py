"""Generate the AI-tutor talking-avatar face asset (1024x768 teacher portrait).

Pure-PIL procedural drawing so no external images are needed. The face is
placed on a fixed grid so the PuppetEngine in realtime/avatar.py knows exactly
where the mouth lives (constants MOUTH_BOX / EYE_BOXES below).

Usage:
    python avatar/make_teacher_avatar.py [out.png]
"""
import os
import sys

from PIL import Image, ImageDraw, ImageFilter

W, H = 1024, 768
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "avatar", "teacher.png")

# Mouth region (x, y, w, h) that the avatar engine animates.
MOUTH_BOX = (437, 448, 150, 110)
# Eye regions used for blink animation.
EYE_BOXES = [(418, 328, 68, 40), (538, 328, 68, 40)]

# Palette
BG_TOP = (26, 58, 50)
BG_BOT = (44, 84, 71)
SKIN = (232, 185, 139)
SKIN_SHADE = (214, 160, 116)
HAIR = (58, 40, 34)
EYE_WHITE = (252, 250, 246)
IRIS = (74, 46, 27)
PUPIL = (24, 14, 8)
GLASSES = (85, 58, 46)
BLAZER = (47, 74, 102)
BLAZER_DARK = (36, 58, 82)
SHIRT = (244, 242, 238)
TIE = (176, 58, 46)
LIP = (138, 42, 42)
CHEEK = (232, 167, 139)


def lerp(a, b, t):
    return int(a + (b - a) * t)


def vertical_gradient(draw):
    for y in range(H):
        t = y / H
        col = (lerp(BG_TOP[0], BG_BOT[0], t), lerp(BG_TOP[1], BG_BOT[1], t), lerp(BG_TOP[2], BG_BOT[2], t))
        draw.line([(0, y), (W, y)], fill=col)


def vignette(img):
    # Radial mask: bright (255) at the face center, fading to dark (0) at the
    # edges. Painted largest-first so each smaller disc overwrites the centre
    # with a higher value (PIL fills are clamped to 0..255 in L mode).
    mask = Image.new("L", (W, H), 0)
    dm = ImageDraw.Draw(mask)
    cx, cy = W / 2, 340
    max_r = 620.0
    for r in range(int(max_r), 0, -6):
        if r <= 360:
            fill = 255
        else:
            t = (r - 360) / (max_r - 360)
            fill = int(255 * (1 - t) ** 1.2)
        dm.ellipse((cx - r, cy - r, cx + r, cy + r), fill=max(0, min(255, fill)))
    dark = Image.new("RGB", (W, H), (0, 0, 0))
    img.paste(Image.composite(img, dark, mask), box=(0, 0))
    return img


def draw_teacher():
    img = Image.new("RGB", (W, H), BG_TOP)
    d = ImageDraw.Draw(img)

    vertical_gradient(d)

    # ---- Torso ----
    # Shoulders / blazer
    d.rounded_rectangle((150, 540, 874, 780), radius=220, fill=BLAZER)
    d.rounded_rectangle((205, 600, 819, 780), radius=190, fill=BLAZER_DARK)
    # Shirt V + tie
    d.polygon([(482, 600), (542, 600), (512, 760)], fill=SHIRT)
    d.polygon([(500, 588), (524, 588), (512, 700)], fill=TIE)
    d.rounded_rectangle((508, 584, 516, 612), radius=4, fill=TIE)

    # ---- Neck ----
    d.rectangle((476, 452, 548, 600), fill=SKIN)
    d.rectangle((476, 470, 496, 600), fill=SKIN_SHADE)

    # ---- Head (base skin) ----
    d.ellipse((322, 85, 702, 575), fill=SKIN)

    # ---- Ears ----
    for ex in (318, 706):
        d.ellipse((ex, 336, ex + 52, 416), fill=SKIN)
        d.ellipse((ex + 10, 348, ex + 46, 402), fill=SKIN_SHADE)

    # ---- Hair cap (drawn on top of head) ----
    d.ellipse((297, 55, 727, 340), fill=HAIR)
    # Side burns
    d.rounded_rectangle((312, 300, 340, 430), radius=14, fill=HAIR)
    d.rounded_rectangle((684, 300, 712, 430), radius=14, fill=HAIR)

    # ---- Cheeks ----
    d.ellipse((420, 408, 470, 444), fill=CHEEK)
    d.ellipse((554, 408, 604, 444), fill=CHEEK)

    # ---- Brows ----
    for bx, bw in ((408, 88), (528, 88)):
        d.arc((bx, 300, bx + bw, 350), start=-30, end=210, fill=HAIR, width=9)

    # ---- Eyes (whites + iris + pupil + highlight) ----
    for cx in (452, 572):
        d.ellipse((cx - 34, 348 - 20, cx + 34, 348 + 20), fill=EYE_WHITE)
        d.ellipse((cx - 13, 348 - 13, cx + 13, 348 + 13), fill=IRIS)
        d.ellipse((cx - 6, 348 - 6, cx + 6, 348 + 6), fill=PUPIL)
        d.ellipse((cx + 3, 348 - 5, cx + 8, 348 + 1), fill=EYE_WHITE)
    # Upper lids
    for cx in (452, 572):
        d.arc((cx - 34, 328, cx + 34, 352), start=180, end=360, fill=SKIN_SHADE, width=5)

    # ---- Glasses ----
    for gx in (410, 530):
        d.rounded_rectangle((gx, 324, gx + 84, 376), radius=16, outline=GLASSES, width=7)
    d.line((494, 348, 530, 348), fill=GLASSES, width=6)

    # ---- Nose ----
    d.ellipse((498, 392, 526, 430), fill=SKIN_SHADE)

    # ---- Mouth (closed gentle smile) ----
    d.arc((482, 428, 512, 472), start=20, end=160, fill=LIP, width=6)
    d.arc((512, 428, 542, 472), start=20, end=160, fill=LIP, width=6)
    # Lip gloss line
    d.line((468, 448, 468, 452), fill=(196, 82, 82), width=8)
    d.line((556, 448, 556, 452), fill=(196, 82, 82), width=8)

    img = vignette(img)
    img.save(DEFAULT_OUT)
    print("wrote", DEFAULT_OUT)
    print("MOUTH_BOX", MOUTH_BOX)
    print("EYE_BOXES", EYE_BOXES)


if __name__ == "__main__":
    draw_teacher()