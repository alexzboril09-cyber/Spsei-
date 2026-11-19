#!/usr/bin/env python3
"""
Stick Man "Rules" video generator.

Turns a story file (parts/*.json) into a vertical 9:16 short (MP4) with:
  - an animated stick man in creepy night scenes
  - big captions with highlighted words
  - a clock in the corner, rule cards, phone texts, knocking, etc.
  - generated ambient music + sound effects
  - an AI narrator voice (offline Kokoro voice, downloaded automatically once)
  - an ending where viewers choose A or B in the comments

Usage:
  python make_video.py parts/part01.json
  python make_video.py parts/part01.json --preview        # fast, half size
  python make_video.py parts/part01.json --voice am_michael   # pick another narrator
  python make_video.py parts/part01.json --no-voice           # no narrator
  python make_video.py parts/*.json                        # render every part
"""
import argparse, asyncio, json, math, os, random, re, shutil, subprocess, sys, tempfile, wave
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 30
SR = 44100

# ----------------------------------------------------------------------------- utils

def hexrgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def lerp(a, b, t):
    return a + (b - a) * t

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def ease(t):
    t = clamp(t)
    return t * t * (3 - 2 * t)

def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3

@lru_cache(maxsize=64)
def font(name, size):
    path = os.path.join(HERE, "fonts", name)
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()

BLACK = "Inter-Black.otf"
SEMI = "Inter-SemiBold.otf"
MONO = "JetBrainsMono-Bold.ttf"

WHITE = (240, 240, 248)
RED = (255, 59, 77)


class Canvas:
    """Holds the frame and a scale factor so all drawing uses 1080x1920 design units."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.s = w / 1080.0

    def p(self, v):
        return int(round(v * self.s))

    def pt(self, x, y):
        return (self.p(x), self.p(y))

    def f(self, name, size):
        return font(name, max(6, self.p(size)))


# ----------------------------------------------------------------------------- drawing helpers

def vgrad(cv, top, bottom):
    h = cv.h
    t = np.linspace(0, 1, h)[:, None]
    col = (np.array(top)[None, :] * (1 - t) + np.array(bottom)[None, :] * t).astype(np.uint8)
    arr = np.repeat(col[:, None, :], cv.w, axis=1)
    return Image.fromarray(arr, "RGB")

def overlay(img, draw_fn):
    """Draw semi-transparent shapes: draw_fn(ImageDraw on RGBA layer)."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer))
    img.paste(layer, (0, 0), layer)

def line(d, cv, a, b, col, w):
    d.line([cv.pt(*a), cv.pt(*b)], fill=col, width=cv.p(w))
    r = cv.p(w) / 2
    for (x, y) in (a, b):
        X, Y = cv.pt(x, y)
        d.ellipse([X - r, Y - r, X + r, Y + r], fill=col)

def circle(d, cv, x, y, r, fill=None, outline=None, width=1):
    X, Y, R = cv.p(x), cv.p(y), cv.p(r)
    d.ellipse([X - R, Y - R, X + R, Y + R], fill=fill, outline=outline, width=cv.p(width) if outline else 0)

def rrect(d, cv, x0, y0, x1, y1, r, fill=None, outline=None, width=1):
    d.rounded_rectangle([cv.p(x0), cv.p(y0), cv.p(x1), cv.p(y1)], radius=cv.p(r), fill=fill,
                        outline=outline, width=cv.p(width) if outline else 0)

def text_c(d, cv, x, y, s, fnt, fill, stroke=0, stroke_fill=(0, 0, 0), anchor="mm"):
    d.text(cv.pt(x, y), s, font=fnt, fill=fill, anchor=anchor,
           stroke_width=cv.p(stroke), stroke_fill=stroke_fill)

def wrap(text, fnt, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if fnt.getlength(t) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# ----------------------------------------------------------------------------- stick man

def stickman(d, cv, x, ground, h=420, pose="stand", ph=0.0, facing=1, col=WHITE, lw=None, eyes=True):
    """Draw a stick man with feet on `ground`. ph = animation phase (seconds)."""
    lw = lw or h * 0.045
    r = h * 0.12
    hip_y = ground - h * 0.46
    neck_y = ground - h * 0.76
    head = (x, neck_y - r * 0.95)
    sh = (x, neck_y + h * 0.04)
    hip = (x, hip_y)
    breathe = math.sin(ph * 2.2) * h * 0.006

    def leg(angle, knee_bend=0.0):
        L = h * 0.23
        kx = hip[0] + math.sin(angle) * L
        ky = hip[1] + math.cos(angle) * L
        a2 = angle - knee_bend * facing
        fx = kx + math.sin(a2) * L
        fy = ky + math.cos(a2) * L
        line(d, cv, hip, (kx, ky), col, lw)
        line(d, cv, (kx, ky), (fx, fy), col, lw)

    def arm(angle, bend=0.3, base=sh):
        L = h * 0.19
        ex = base[0] + math.sin(angle) * L
        ey = base[1] + math.cos(angle) * L
        a2 = angle + bend
        hx = ex + math.sin(a2) * L
        hy = ey + math.cos(a2) * L
        line(d, cv, base, (ex, ey), col, lw)
        line(d, cv, (ex, ey), (hx, hy), col, lw)
        return hx, hy

    jitter = 0
    if pose in ("scared", "shiver"):
        jitter = math.sin(ph * 60) * h * 0.006
        head = (head[0] + jitter, head[1])
        sh = (sh[0] + jitter, sh[1])

    if pose == "walk" or pose == "run":
        sp = 7.5 if pose == "walk" else 13
        amp = 0.42 if pose == "walk" else 0.7
        s = math.sin(ph * sp)
        lean = 0.0 if pose == "walk" else h * 0.05 * facing
        sh = (sh[0] + lean, sh[1])
        head = (head[0] + lean, head[1])
        line(d, cv, sh, hip, col, lw)
        leg(s * amp * facing, knee_bend=max(0, -s) * 0.6)
        leg(-s * amp * facing, knee_bend=max(0, s) * 0.6)
        arm(-s * amp * 0.9 * facing, 0.5 * facing)
        arm(s * amp * 0.9 * facing, 0.5 * facing)
    elif pose == "sit":
        hip = (x, ground - h * 0.27)
        neck_y = hip[1] - h * 0.30
        sh = (x, neck_y + h * 0.04)
        head = (x, neck_y - r * 0.95)
        line(d, cv, sh, hip, col, lw)
        knee = (x - facing * h * 0.22, hip[1])
        line(d, cv, hip, knee, col, lw)
        line(d, cv, knee, (knee[0] - facing * h * 0.02, ground), col, lw)
        line(d, cv, sh, (x - facing * h * 0.12, hip[1] - h * 0.04), col, lw)
        line(d, cv, (x - facing * h * 0.12, hip[1] - h * 0.04), (knee[0] + facing * h * 0.03, knee[1] - h * 0.03), col, lw)
    elif pose == "phone":
        line(d, cv, sh, hip, col, lw)
        leg(0.16)
        leg(-0.16)
        arm(-0.25 * facing, 0.2)
        hx, hy = sh[0] + facing * h * 0.13, sh[1] - h * 0.02
        line(d, cv, sh, (sh[0] + facing * h * 0.05, sh[1] + h * 0.15), col, lw)
        line(d, cv, (sh[0] + facing * h * 0.05, sh[1] + h * 0.15), (hx, hy), col, lw)
        overlay_glow = (hx, hy)
        rrect(d, cv, hx - h * 0.03, hy - h * 0.05, hx + h * 0.03, hy + h * 0.03, h * 0.008, fill=(170, 220, 255))
    elif pose == "scared":
        line(d, cv, sh, hip, col, lw)
        leg(0.22)
        leg(-0.22)
        line(d, cv, sh, (sh[0] - h * 0.14, sh[1] - h * 0.05), col, lw)
        line(d, cv, (sh[0] - h * 0.14, sh[1] - h * 0.05), (sh[0] - h * 0.10, sh[1] - h * 0.24), col, lw)
        line(d, cv, sh, (sh[0] + h * 0.14, sh[1] - h * 0.05), col, lw)
        line(d, cv, (sh[0] + h * 0.14, sh[1] - h * 0.05), (sh[0] + h * 0.10, sh[1] - h * 0.24), col, lw)
    elif pose == "lie":
        y = ground - h * 0.06
        line(d, cv, (x - h * 0.35, y), (x + h * 0.35, y), col, lw)
        head = (x - h * 0.35 - r, y - r * 0.3)
    else:  # stand / shiver
        line(d, cv, sh, (hip[0], hip[1] + breathe), col, lw)
        leg(0.14)
        leg(-0.14)
        arm(0.22 + math.sin(ph * 2) * 0.03, -0.1)
        arm(-0.22 - math.sin(ph * 2) * 0.03, 0.1)

    circle(d, cv, head[0], head[1] + breathe, r, fill=col)
    if eyes and pose != "lie":
        ex = head[0] + facing * r * 0.30
        for dx in (-r * 0.22, r * 0.22):
            circle(d, cv, ex + dx, head[1] - r * 0.05 + breathe, r * 0.10, fill=(25, 25, 35))


def shadow_figure(img, cv, x, ground, h, alpha=1.0, eye_col=(255, 255, 255), eye_glow=0.0):
    """Tall black silhouette with glowing eyes."""
    a = int(255 * clamp(alpha))
    if a <= 0:
        return

    def fn(d):
        col = (4, 4, 8, a)
        r = h * 0.11
        hy = ground - h * 0.83
        circle(d, cv, x, hy, r, fill=col)
        d.polygon([cv.pt(x - h * 0.11, ground), cv.pt(x - h * 0.06, hy + r * 0.6),
                   cv.pt(x + h * 0.06, hy + r * 0.6), cv.pt(x + h * 0.11, ground)], fill=col)
        ea = int(a * (0.6 + 0.4 * eye_glow))
        if eye_glow > 0:
            for dx in (-r * 0.35, r * 0.35):
                circle(d, cv, x + dx, hy, r * 0.35, fill=eye_col + (int(ea * 0.25),))
        for dx in (-r * 0.35, r * 0.35):
            circle(d, cv, x + dx, hy, r * 0.11, fill=eye_col + (ea,))
    overlay(img, fn)


def handprint(d, cv, x, y, s, col):
    circle(d, cv, x, y, s * 0.55, fill=col)
    for i, (ang, ln) in enumerate([(-0.9, 0.75), (-0.35, 0.95), (0.0, 1.0), (0.35, 0.92), (0.75, 0.7)]):
        fx = x + math.sin(ang) * s * (0.6 + ln * 0.55)
        fy = y - math.cos(ang) * s * (0.6 + ln * 0.55)
        circle(d, cv, fx, fy, s * 0.17, fill=col)
        mx = x + math.sin(ang) * s * 0.55
        my = y - math.cos(ang) * s * 0.55
        line(d, cv, (mx, my), (fx, fy), col, s * 0.3)


# ----------------------------------------------------------------------------- scenes
# Each scene: fn(img, cv, beat, t, dur, ctx). t = seconds into the beat.

def man_from(beat, t, dur, default_x=540, default_pose="stand"):
    m = beat.get("man")
    if m is False:
        return None
    m = m or {}
    if isinstance(m, str):
        m = {"pose": m}
    x0 = m.get("from", m.get("x", default_x / 1080)) * 1080
    x1 = m.get("to", m.get("x", x0 / 1080)) * 1080
    x = lerp(x0, x1, ease(t / max(dur, 0.01)))
    facing = m.get("facing", 1 if x1 >= x0 else -1)
    pose = m.get("pose", default_pose if x0 == x1 else "walk")
    return x, pose, facing, m.get("size", 1.0)


@lru_cache(maxsize=8)
def bg_street(w, h, accent):
    cv = Canvas(w, h)
    img = vgrad(cv, (10, 14, 34), (24, 30, 58))
    d = ImageDraw.Draw(img)
    circle(d, cv, 820, 330, 60, fill=(225, 228, 240))
    circle(d, cv, 845, 315, 58, fill=(16, 20, 44))
    rnd = random.Random(7)
    x = -20
    while x < 1100:
        bw = rnd.randint(110, 210)
        bh = rnd.randint(260, 620)
        top = 1300 - bh
        d.rectangle([cv.p(x), cv.p(top), cv.p(x + bw), cv.p(1300)], fill=(14, 17, 34))
        for wy in range(int(top) + 30, 1270, 58):
            for wx in range(int(x) + 18, int(x + bw) - 24, 42):
                if rnd.random() < 0.12:
                    d.rectangle([cv.p(wx), cv.p(wy), cv.p(wx + 20), cv.p(wy + 28)], fill=(200, 170, 90))
        x += bw + rnd.randint(4, 20)
    d.rectangle([0, cv.p(1300), w, h], fill=(28, 30, 44))
    d.rectangle([0, cv.p(1300), w, cv.p(1312)], fill=(50, 52, 70))
    d.rectangle([0, cv.p(1560), w, h], fill=(18, 19, 28))
    for xx in range(0, 1080, 180):
        d.rectangle([cv.p(xx + 30), cv.p(1700), cv.p(xx + 120), cv.p(1712)], fill=(70, 70, 60))
    # street lamp + light cone
    d.rectangle([cv.p(880), cv.p(700), cv.p(894), cv.p(1310)], fill=(40, 42, 56))
    d.rectangle([cv.p(800), cv.p(690), cv.p(894), cv.p(704)], fill=(40, 42, 56))
    overlay(img, lambda o: o.polygon([cv.pt(790, 704), cv.pt(830, 704), cv.pt(1000, 1310), cv.pt(620, 1310)],
                                     fill=(255, 220, 140, 34)))
    overlay(img, lambda o: o.ellipse([cv.p(600), cv.p(1270), cv.p(1020), cv.p(1350)], fill=(255, 220, 140, 40)))
    rrect(d, cv, 792, 698, 836, 712, 4, fill=(255, 236, 180))
    # bus stop sign
    d.rectangle([cv.p(300), cv.p(930), cv.p(312), cv.p(1310)], fill=(70, 72, 90))
    circle(d, cv, 306, 900, 62, fill=hexrgb(accent))
    text_c(d, cv, 306, 900, "0", cv.f(BLACK, 72), (15, 15, 25))
    return img


def draw_bus(img, cv, front_x, bottom=1360, accent="#c6ff3d", t=0):
    d = ImageDraw.Draw(img)
    L, H = 1150, 430
    x0, x1, y0 = front_x, front_x + L, bottom - H
    rrect(d, cv, x0, y0, x1, bottom, 46, fill=(38, 52, 64))
    d.rectangle([cv.p(x0 + 10), cv.p(bottom - 120), cv.p(x1), cv.p(bottom - 100)], fill=(hexrgb(accent)))
    # windows
    for i in range(5):
        wx = x0 + 150 + i * 195
        rrect(d, cv, wx, y0 + 60, wx + 165, y0 + 220, 14, fill=(232, 205, 140))
    rrect(d, cv, x0 + 18, y0 + 60, x0 + 120, y0 + 300, 14, fill=(200, 180, 130))  # door
    rrect(d, cv, x0 + 40, y0 + 14, x0 + 330, y0 + 50, 8, fill=(10, 10, 14))
    text_c(d, cv, x0 + 185, y0 + 33, "ROUTE 0", cv.f(MONO, 28), (255, 80, 60))
    for wx in (x0 + 230, x1 - 230):
        circle(d, cv, wx, bottom, 70, fill=(10, 10, 14))
        circle(d, cv, wx, bottom, 30, fill=(70, 70, 80))
    overlay(img, lambda o: o.polygon([cv.pt(x0 + 4, bottom - 80), cv.pt(x0 - 520, bottom - 150),
                                      cv.pt(x0 - 520, bottom + 40)], fill=(255, 240, 190, 45)))
    circle(d, cv, x0 + 8, bottom - 80, 16, fill=(255, 245, 210))


def sc_street(img, cv, b, t, dur, ctx):
    img.paste(bg_street(cv.w, cv.h, ctx["accent"]))
    bus = b.get("bus")
    u = t / dur
    man = man_from(b, t, dur, default_x=430)
    if bus == "arrive":
        fx = lerp(1150, 80, ease_out(u * 1.4))
    elif bus == "leave":
        fx = lerp(80, -1400, ease(u) ** 1.6)
    elif bus == "stay":
        fx = 80
    else:
        fx = None
    if fx is not None:
        draw_bus(img, cv, fx, accent=ctx["accent"], t=t)
    if man:
        x, pose, facing, size = man
        stickman(ImageDraw.Draw(img), cv, x, 1330, 360 * size, pose, t, facing)


@lru_cache(maxsize=8)
def bg_bus(w, h, accent):
    cv = Canvas(w, h)
    img = vgrad(cv, (18, 22, 30), (30, 34, 44))
    d = ImageDraw.Draw(img)
    d.rectangle([0, cv.p(470), w, cv.p(560)], fill=(40, 44, 56))  # luggage rail
    d.rectangle([0, cv.p(552), w, cv.p(560)], fill=(70, 74, 88))
    for i in range(4):
        wx = 40 + i * 265
        rrect(d, cv, wx, 610, wx + 235, 900, 26, fill=(8, 10, 22))
    d.rectangle([0, cv.p(1380), w, h], fill=(22, 24, 30))
    d.rectangle([0, cv.p(1380), w, cv.p(1392)], fill=hexrgb(accent))
    for px in (150, 680):
        d.rectangle([cv.p(px), cv.p(560), cv.p(px + 12), cv.p(1380)], fill=(170, 175, 185))
    # cord along ceiling
    d.line([cv.pt(0, 590), cv.pt(1080, 590)], fill=(200, 60, 60), width=cv.p(5))
    # seats
    for sx in (260, 520):
        rrect(d, cv, sx + 60, 1010, sx + 120, 1270, 20, fill=(60, 40, 90))
        rrect(d, cv, sx - 90, 1230, sx + 120, 1280, 14, fill=(70, 46, 104))
        d.rectangle([cv.p(sx - 10), cv.p(1280), cv.p(sx + 10), cv.p(1380)], fill=(50, 50, 60))
    # back row bench
    rrect(d, cv, 820, 960, 1080, 1280, 20, fill=(78, 30, 40))
    rrect(d, cv, 800, 1230, 1080, 1290, 14, fill=(92, 36, 48))
    text_c(d, cv, 950, 940, "BACK ROW", cv.f(MONO, 24), (150, 70, 80))
    # sign above
    rrect(d, cv, 330, 490, 750, 540, 8, fill=(10, 10, 12))
    return img


def sc_bus(img, cv, b, t, dur, ctx):
    img.paste(bg_bus(cv.w, cv.h, ctx["accent"]))
    d = ImageDraw.Draw(img)
    # passing lights in windows (bus drives left -> lights slide right)
    speed = 900 if not b.get("stopped") else 0
    for i in range(4):
        wx = 40 + i * 265
        for k in range(3):
            lx = wx + ((t * speed + k * 310 + i * 97) % 520) - 140
            if wx < lx < wx + 235:
                circle(d, cv, lx, 700 + k * 60, 12 + k * 4, fill=(110, 100, 70))
    if b.get("window_text"):
        pass
    sign = ctx.get("clock", "")
    text_c(d, cv, 540, 515, b.get("sign", "NEXT STOP: --"), cv.f(MONO, 26), (255, 90, 60))
    if b.get("driver"):
        rim = (120, 130, 165)
        rrect(d, cv, -40, 1060, 170, 1290, 14, fill=(40, 30, 60))           # driver seat
        d.polygon([cv.pt(20, 1290), cv.pt(40, 900), cv.pt(160, 900), cv.pt(180, 1290)], fill=(6, 6, 10), outline=rim)
        circle(d, cv, 100, 830, 70, fill=(6, 6, 10), outline=rim, width=4)
        rrect(d, cv, 30, 750, 175, 795, 12, fill=(16, 28, 60), outline=rim, width=3)  # cap
        d.rectangle([cv.p(150), cv.p(782), cv.p(230), cv.p(796)], fill=(16, 28, 60))
        rrect(d, cv, 220, 1000, 250, 1060, 6, fill=(60, 60, 70))
    man = man_from(b, t, dur, default_x=430, default_pose="sit")
    if man:
        x, pose, facing, size = man
        g = 1290 if pose == "sit" else 1380
        stickman(d, cv, x, g, 400 * size, pose, t, facing if pose != "sit" else -1)
    bf = b.get("back_figure")
    if bf:
        a = 1.0 if bf == "there" else ease(t / max(dur * 0.8, 0.01))
        shadow_figure(img, cv, 940, 1275, 600, alpha=a, eye_col=RED if b.get("red_eyes") else (255, 255, 255),
                      eye_glow=0.5 + 0.5 * math.sin(t * 5))


def sc_bus_window(img, cv, b, t, dur, ctx):
    img.paste(vgrad(cv, (14, 16, 26), (22, 24, 34)))
    d = ImageDraw.Draw(img)
    rrect(d, cv, 90, 420, 990, 1420, 60, fill=(50, 54, 66))
    rrect(d, cv, 130, 460, 950, 1380, 44, fill=(6, 8, 18))
    speed = 0 if b.get("stopped") else 1300
    for k in range(6):
        lx = ((t * speed + k * 330) % 1400) - 200
        if 150 < lx < 930:
            circle(d, cv, lx, 760 + (k % 3) * 160, 20, fill=(70, 64, 46))
    # faint reflection of the stick man
    overlay(img, lambda o: (circle(o, cv, 760, 1180, 90, fill=(255, 255, 255, 22)),
                            line(o, cv, (760, 1270), (760, 1420), (255, 255, 255, 22), 16)))
    n = int(b.get("hands", 0))
    u = t / dur
    for i in range(n):
        appear = (i + 0.3) / (n + 0.3)
        if u >= appear * 0.85:
            rnd = random.Random(i * 13 + 5)
            hx = rnd.uniform(260, 820)
            hy = rnd.uniform(620, 1180)
            a = int(150 * ease((u - appear * 0.85) * 8))
            overlay(img, lambda o, hx=hx, hy=hy, a=a: handprint(o, cv, hx, hy, 75, (210, 220, 235, a)))
    if b.get("face_outside"):
        a = ease(u * 1.6)
        shadow_figure(img, cv, 380, 1500, 900, alpha=a, eye_col=(255, 255, 255), eye_glow=0.8)


@lru_cache(maxsize=4)
def bg_bedroom(w, h):
    cv = Canvas(w, h)
    img = vgrad(cv, (14, 18, 40), (26, 30, 62))
    d = ImageDraw.Draw(img)
    rrect(d, cv, 560, 420, 940, 820, 10, fill=(30, 40, 80))
    rrect(d, cv, 585, 445, 915, 795, 6, fill=(18, 24, 58))
    d.line([cv.pt(750, 445), cv.pt(750, 795)], fill=(30, 40, 80), width=cv.p(12))
    circle(d, cv, 840, 540, 32, fill=(210, 215, 235))
    d.rectangle([0, cv.p(1320), w, h], fill=(16, 18, 36))
    rrect(d, cv, 60, 1150, 520, 1330, 20, fill=(48, 56, 110))
    rrect(d, cv, 80, 1110, 220, 1170, 24, fill=(200, 205, 225))
    rrect(d, cv, 160, 520, 380, 680, 6, outline=(60, 66, 110), width=8)
    return img


def sc_bedroom(img, cv, b, t, dur, ctx):
    img.paste(bg_bedroom(cv.w, cv.h))
    d = ImageDraw.Draw(img)
    if b.get("lamp", False):
        overlay(img, lambda o: o.ellipse([cv.p(700), cv.p(980), cv.p(1100), cv.p(1380)], fill=(255, 220, 140, 40)))
    rrect(d, cv, 860, 1200, 960, 1320, 8, fill=(40, 44, 80))
    d.polygon([cv.pt(870, 1130), cv.pt(950, 1130), cv.pt(975, 1200), cv.pt(845, 1200)],
              fill=(255, 220, 150) if b.get("lamp") else (70, 74, 110))
    man = man_from(b, t, dur, default_x=300, default_pose="sit")
    if man:
        x, pose, facing, size = man
        stickman(d, cv, x, 1210 if pose == "sit" else 1330, 380 * size, pose, t, facing)


def sc_door(img, cv, b, t, dur, ctx):
    img.paste(vgrad(cv, (14, 16, 30), (24, 26, 44)))
    d = ImageDraw.Draw(img)
    knock = b.get("knock", False)
    jig = math.sin(t * 40) * 4 if knock and (t % 0.7) < 0.15 else 0
    rrect(d, cv, 300 + jig, 520, 780 + jig, 1400, 10, fill=(48, 36, 30))
    for (y0, y1) in ((580, 900), (960, 1340)):
        rrect(d, cv, 350 + jig, y0, 730 + jig, y1, 6, outline=(66, 50, 42), width=8)
    circle(d, cv, 720 + jig, 1000, 18, fill=(170, 160, 140))
    circle(d, cv, 540 + jig, 640, 10, fill=(20, 20, 26))  # peephole
    light = b.get("light_under", True)
    if light:
        overlay(img, lambda o: o.polygon([cv.pt(300, 1400), cv.pt(780, 1400), cv.pt(1000, 1700), cv.pt(80, 1700)],
                                         fill=(255, 190, 110, 70)))
        d.rectangle([cv.p(300), cv.p(1398), cv.p(780), cv.p(1408)], fill=(255, 210, 140))
    if b.get("feet", False):
        for fx in (440, 600):
            d.rectangle([cv.p(fx), cv.p(1398), cv.p(fx + 60), cv.p(1408)], fill=(10, 10, 12))
            overlay(img, lambda o, fx=fx: o.polygon([cv.pt(fx, 1408), cv.pt(fx + 60, 1408), cv.pt(fx + 120, 1700),
                                                     cv.pt(fx - 40, 1700)], fill=(0, 0, 0, 120)))


def sc_phone(img, cv, b, t, dur, ctx):
    img.paste(vgrad(cv, (10, 12, 22), (18, 20, 34)))
    d = ImageDraw.Draw(img)
    buzz = math.sin(t * 90) * 5 if t < 0.5 else 0
    ox = buzz
    rrect(d, cv, 170 + ox, 260, 910 + ox, 1560, 80, fill=(30, 32, 40))
    rrect(d, cv, 190 + ox, 280, 890 + ox, 1540, 64, fill=(4, 4, 8))
    rrect(d, cv, 450 + ox, 300, 630 + ox, 340, 20, fill=(0, 0, 0))
    contact = b.get("contact", "Unknown")
    circle(d, cv, 540 + ox, 420, 44, fill=(90, 90, 110))
    text_c(d, cv, 540 + ox, 420, contact[:1].upper(), cv.f(BLACK, 44), WHITE)
    text_c(d, cv, 540 + ox, 498, contact, cv.f(SEMI, 32), WHITE)
    text_c(d, cv, 540 + ox, 545, b.get("time", ctx.get("clock", "")), cv.f(SEMI, 22), (130, 130, 150))
    msgs = b.get("messages", [])
    n = len(msgs)
    fnt = cv.f(SEMI, 40)
    y = 600
    show_upto = (t / dur) * (n + (1 if b.get("typing") else 0.4))
    for i, m in enumerate(msgs):
        if i > show_upto - 0.05:
            break
        mine = m.get("from") == "me"
        lines = wrap(m["text"], fnt, cv.p(520))
        bw = max(fnt.getlength(l) for l in lines) / cv.s + 52
        bh = len(lines) * 52 + 32
        x0 = 860 - bw if mine else 220
        pop = ease_out((show_upto - i) * 3)
        col = (40, 110, 255) if mine else (52, 54, 64)
        if m.get("creepy"):
            col = (140, 20, 30)
        rrect(d, cv, x0 + ox, y + (1 - pop) * 30, x0 + bw + ox, y + bh + (1 - pop) * 30, 28, fill=col)
        for k, l in enumerate(lines):
            d.text(cv.pt(x0 + 26 + ox, y + 16 + k * 52 + (1 - pop) * 30), l, font=fnt, fill=WHITE)
        y += bh + 22
    if b.get("typing") and show_upto > n:
        rrect(d, cv, 220 + ox, y, 340 + ox, y + 70, 30, fill=(52, 54, 64))
        for k in range(3):
            a = 0.5 + 0.5 * math.sin(t * 8 - k)
            circle(d, cv, 250 + k * 30 + ox, y + 35, 9, fill=tuple(int(120 + 120 * a) for _ in range(3)))
    if b.get("glow", True):
        overlay(img, lambda o: o.ellipse([cv.p(0), cv.p(200), cv.p(1080), cv.p(1700)], fill=(120, 160, 255, 10)))


def draw_icon(d, cv, img, icon, x, y, s, t, accent):
    c = (200, 200, 215)
    if icon == "seat":
        rrect(d, cv, x - s * 0.1, y - s * 0.7, x + s * 0.15, y + s * 0.1, 14, fill=(92, 36, 48))
        rrect(d, cv, x - s * 0.6, y, x + s * 0.15, y + s * 0.2, 12, fill=(110, 44, 56))
        line(d, cv, (x - s * 0.7, y - s * 0.8), (x + s * 0.4, y + s * 0.4), RED, 16)
        line(d, cv, (x + s * 0.4, y - s * 0.8), (x - s * 0.7, y + s * 0.4), RED, 16)
    elif icon == "name":
        rrect(d, cv, x - s * 0.6, y - s * 0.35, x + s * 0.6, y + s * 0.35, 16, fill=(230, 230, 240))
        rrect(d, cv, x - s * 0.6, y - s * 0.35, x + s * 0.6, y - s * 0.12, 16, fill=RED)
        text_c(d, cv, x, y - s * 0.23, "HELLO MY NAME IS", cv.f(BLACK, 18), WHITE)
        text_c(d, cv, x, y + s * 0.1, "???", cv.f(BLACK, 54), (20, 20, 30))
    elif icon == "window":
        rrect(d, cv, x - s * 0.55, y - s * 0.45, x + s * 0.55, y + s * 0.45, 20, fill=(50, 54, 66))
        rrect(d, cv, x - s * 0.45, y - s * 0.35, x + s * 0.45, y + s * 0.35, 14, fill=(6, 8, 18))
        circle(d, cv, x - s * 0.12, y, s * 0.05, fill=WHITE)
        circle(d, cv, x + s * 0.12, y, s * 0.05, fill=WHITE)
    elif icon == "eye":
        d.ellipse([cv.p(x - s * 0.6), cv.p(y - s * 0.3), cv.p(x + s * 0.6), cv.p(y + s * 0.3)], fill=c)
        circle(d, cv, x + math.sin(t * 2) * s * 0.15, y, s * 0.2, fill=(20, 20, 30))
    elif icon == "light":
        rrect(d, cv, x - s * 0.22, y - s * 0.4, x + s * 0.22, y + s * 0.4, 14, fill=(220, 220, 228))
        rrect(d, cv, x - s * 0.08, y - s * 0.22, x + s * 0.08, y + s * 0.22, 6, fill=(120, 120, 130))
        rrect(d, cv, x - s * 0.06, y, x + s * 0.06, y + s * 0.18, 4, fill=(60, 60, 70))
    elif icon == "door":
        rrect(d, cv, x - s * 0.3, y - s * 0.6, x + s * 0.3, y + s * 0.6, 8, fill=(70, 50, 40))
        circle(d, cv, x + s * 0.2, y, s * 0.05, fill=(190, 180, 150))
    elif icon == "clock":
        circle(d, cv, x, y, s * 0.5, fill=(230, 230, 238))
        line(d, cv, (x, y), (x, y - s * 0.35), (20, 20, 30), 10)
        line(d, cv, (x, y), (x + s * 0.25, y + s * 0.05), (20, 20, 30), 10)
    elif icon == "phone":
        rrect(d, cv, x - s * 0.28, y - s * 0.55, x + s * 0.28, y + s * 0.55, 30, fill=(30, 32, 40))
        rrect(d, cv, x - s * 0.22, y - s * 0.47, x + s * 0.22, y + s * 0.47, 22, fill=(60, 100, 200))
    elif icon == "cord":
        line(d, cv, (x - s * 0.7, y - s * 0.5), (x + s * 0.7, y - s * 0.5), (200, 60, 60), 12)
        line(d, cv, (x, y - s * 0.5), (x, y + s * 0.2), (200, 60, 60), 8)
        circle(d, cv, x, y + s * 0.28, s * 0.1, fill=(220, 220, 230))
    elif icon == "stop":
        pts = [cv.pt(x + math.cos(math.pi / 8 + k * math.pi / 4) * s * 0.5,
                     y + math.sin(math.pi / 8 + k * math.pi / 4) * s * 0.5) for k in range(8)]
        d.polygon(pts, fill=RED)
        text_c(d, cv, x, y, "STOP", cv.f(BLACK, 46), WHITE)


def sc_rule(img, cv, b, t, dur, ctx):
    img.paste(vgrad(cv, (12, 14, 26), (22, 24, 40)))
    d = ImageDraw.Draw(img)
    rrect(d, cv, 150, 380, 930, 1180, 30, fill=(236, 230, 210))  # laminated card
    rrect(d, cv, 150, 380, 930, 1180, 30, outline=(255, 255, 255), width=4)
    rrect(d, cv, 420, 420, 660, 480, 12, fill=RED)
    text_c(d, cv, 540, 451, f"RULE #{b.get('num', 1)}", cv.f(BLACK, 34), WHITE)
    fnt = cv.f(BLACK, 62)
    lines = wrap(b.get("text", "").upper(), fnt, cv.p(680))
    reveal = ease(t / 0.4)
    for i, l in enumerate(lines):
        text_c(d, cv, 540, 560 + i * 74, l, fnt, (20, 20, 30) if reveal > (i / max(len(lines), 1)) else (236, 230, 210))
    if b.get("icon"):
        draw_icon(d, cv, img, b["icon"], 540, 980, 220, t, ctx["accent"])
    # slight tilt bounce on entry
    return


def sc_face(img, cv, b, t, dur, ctx):
    img.paste(vgrad(cv, (14, 16, 30), (8, 8, 16)))
    d = ImageDraw.Draw(img)
    expr = b.get("expr", "scared")
    shake = math.sin(t * 55) * 4 if expr in ("scared", "panic") else 0
    cx, cy, r = 540 + shake, 760, 260
    # shoulders
    line(d, cv, (cx, cy + r), (cx, 1340), WHITE, 34)
    line(d, cv, (cx, cy + r + 70), (cx - 300, 1340), WHITE, 34)
    line(d, cv, (cx, cy + r + 70), (cx + 300, 1340), WHITE, 34)
    circle(d, cv, cx, cy, r, fill=WHITE)
    look = math.sin(t * 3) * 14 if expr in ("worried", "scared") else 0
    if expr == "calm":
        for ex in (-90, 90):
            circle(d, cv, cx + ex, cy - 20, 22, fill=(20, 20, 30))
        d.arc([cv.p(cx - 70), cv.p(cy + 40), cv.p(cx + 70), cv.p(cy + 120)], 20, 160, fill=(20, 20, 30), width=cv.p(12))
    elif expr == "smile":  # creepy
        for ex in (-90, 90):
            circle(d, cv, cx + ex, cy - 20, 26, fill=(20, 20, 30))
        d.arc([cv.p(cx - 150), cv.p(cy - 40), cv.p(cx + 150), cv.p(cy + 150)], 15, 165, fill=(20, 20, 30), width=cv.p(14))
    else:
        er = 70 if expr in ("scared", "panic", "shocked") else 56
        for ex in (-100, 100):
            circle(d, cv, cx + ex, cy - 30, er, fill=(250, 250, 255), outline=(20, 20, 30), width=8)
            circle(d, cv, cx + ex + look, cy - 30, 18 if expr != "worried" else 24, fill=(20, 20, 30))
        if expr in ("scared", "panic", "shocked"):
            d.ellipse([cv.p(cx - 36), cv.p(cy + 90), cv.p(cx + 36), cv.p(cy + 170)], fill=(20, 20, 30))
        else:
            d.line([cv.pt(cx - 60, cy + 120), cv.pt(cx + 60, cy + 110)], fill=(20, 20, 30), width=cv.p(12))
        if expr in ("worried", "scared", "panic"):
            dy = (t * 120) % 200
            d.polygon([cv.pt(cx + 230, cy - 110 + dy), cv.pt(cx + 210, cy - 60 + dy), cv.pt(cx + 250, cy - 60 + dy)],
                      fill=(120, 200, 255))
            circle(d, cv, cx + 230, cy - 55 + dy, 21, fill=(120, 200, 255))


def sc_text(img, cv, b, t, dur, ctx):
    img.paste(vgrad(cv, (6, 6, 10), (14, 10, 16)))
    d = ImageDraw.Draw(img)
    lines = b.get("lines", [b.get("big", "")])
    col = RED if b.get("color", "red") == "red" else (hexrgb(b["color"]) if b["color"].startswith("#") else WHITE)
    size = b.get("size", 120)
    fnt = cv.f(BLACK, size)
    n = len(lines)
    total_h = n * size * 1.1
    for i, l in enumerate(lines):
        appear = i / max(n, 1) * 0.7
        if t / dur < appear:
            continue
        jx = random.uniform(-6, 6) if b.get("glitch", True) else 0
        x = 540 + (i - (n - 1) / 2) * b.get("stagger", 0) + jx
        y = 860 - total_h / 2 + i * size * 1.1 + size / 2
        if b.get("glitch", True) and random.random() < 0.15:
            text_c(d, cv, x + 10, y, l, fnt, (0, 200, 255))
        text_c(d, cv, x, y, l, fnt, col)


def sc_title(img, cv, b, t, dur, ctx):
    img.paste(vgrad(cv, (8, 8, 14), (20, 14, 22)))
    d = ImageDraw.Draw(img)
    a = ease(t / 0.6)
    acc = hexrgb(ctx["accent"])
    text_c(d, cv, 540, 620, b.get("kicker", "THE RULES"), cv.f(MONO, 36), (150, 150, 170))
    fnt = cv.f(BLACK, 128)
    for i, l in enumerate(wrap(b.get("title", ctx["series"]).upper(), fnt, cv.p(940))):
        text_c(d, cv, 540, 760 + i * 140, l, fnt, tuple(int(c * a) for c in WHITE))
    text_c(d, cv, 540, 1100, b.get("subtitle", f"PART {ctx['part']}"), cv.f(BLACK, 60), acc)
    stickman(d, cv, 540, 1560, 260, "stand", t, 1)


def sc_choice(img, cv, b, t, dur, ctx):
    img.paste(vgrad(cv, (10, 8, 16), (26, 10, 18)))
    d = ImageDraw.Draw(img)
    u = t / dur
    end_at = 0.72
    acc = hexrgb(ctx["accent"])
    fq = cv.f(BLACK, 82)
    q = wrap(b.get("question", "WHAT SHOULD HE DO?").upper(), fq, cv.p(950))
    for i, l in enumerate(q):
        text_c(d, cv, 540, 420 + i * 96, l, fq, WHITE)
    opts = [("A", b.get("a", "OPTION A"), (40, 110, 255)), ("B", b.get("b", "OPTION B"), (255, 110, 40))]
    fo = cv.f(BLACK, 46)
    for i, (k, label, col) in enumerate(opts):
        pop = ease_out((t - 0.3 - i * 0.5) * 3)
        if pop <= 0:
            continue
        y = 720 + i * 210
        x0, x1 = 540 - 440 * pop, 540 + 440 * pop
        rrect(d, cv, x0, y, x1, y + 170, 40, fill=col)
        if pop > 0.9:
            circle(d, cv, 170, y + 85, 54, fill=WHITE)
            text_c(d, cv, 170, y + 85, k, cv.f(BLACK, 62), col)
            for j, l in enumerate(wrap(label.upper(), fo, cv.p(640))[:2]):
                text_c(d, cv, 600, y + 85 + (j - (len(wrap(label.upper(), fo, cv.p(640))[:2]) - 1) / 2) * 52, l, fo, WHITE)
    pulse = 1 + 0.06 * math.sin(t * 6)
    text_c(d, cv, 540, 1240, "COMMENT  A  OR  B", cv.f(BLACK, int(70 * pulse)), acc)
    text_c(d, cv, 540, 1320, b.get("note", "THE MOST LIKED ANSWER HAPPENS NEXT"), cv.f(SEMI, 32), (190, 190, 205))
    # draw a little arrow pointing down-right (to comment button)
    stickman(d, cv, 900, 1800, 190, "phone", t, -1)
    if u > end_at:
        k = ease((u - end_at) / 0.12)
        overlay(img, lambda o: o.rectangle([0, 0, cv.w, cv.h], fill=(0, 0, 0, int(225 * k))))
        d = ImageDraw.Draw(img)
        if k > 0.6:
            text_c(d, cv, 540, 760, b.get("next_kicker", "COMING NEXT"), cv.f(MONO, 40), (170, 170, 185))
            text_c(d, cv, 540, 900, b.get("next", f"PART {ctx['part'] + 1}"), cv.f(BLACK, 170), RED)
            rrect(d, cv, 300, 1030, 780, 1100, 34, outline=WHITE, width=4)
            text_c(d, cv, 540, 1065, "FOLLOW FOR THE NEXT PART", cv.f(BLACK, 28), WHITE)


SCENES = {
    "street": sc_street, "bus": sc_bus, "bus_window": sc_bus_window, "bedroom": sc_bedroom,
    "door": sc_door, "phone": sc_phone, "rule": sc_rule, "face": sc_face, "text": sc_text,
    "title": sc_title, "choice": sc_choice,
}


# ----------------------------------------------------------------------------- captions + HUD

def chunk_words(say):
    words = say.split()
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= 3 or re.search(r"[.,!?…:;]$", w):
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    return chunks

def norm(w):
    return re.sub(r"[^\w:']", "", w.lower())

def draw_caption(img, cv, beat, t, dur, accent):
    say = beat.get("say", "")
    if not say or beat.get("captions") is False:
        return
    if beat.get("_speech"):
        dur = beat["_speech"] + 0.1  # follow the narrator's timing
    chunks = chunk_words(say)
    weights = [sum(len(w) for w in c) + 3 for c in chunks]
    tot = sum(weights)
    acc_t = 0
    idx = len(chunks) - 1
    start = 0
    for i, wgt in enumerate(weights):
        seg = dur * 0.95 * wgt / tot
        if t < acc_t + seg:
            idx, start = i, acc_t
            break
        acc_t += seg
    else:
        start = acc_t
    words = chunks[idx]
    hl = set()
    for h in beat.get("hl", []):
        hl.update(norm(x) for x in h.split())
    fnt = cv.f(BLACK, 76)
    space = fnt.getlength(" ")
    widths = [fnt.getlength(w.upper()) for w in words]
    total = sum(widths) + space * (len(words) - 1)
    lines = [list(range(len(words)))]
    if total > cv.p(960):
        lines = [list(range(len(words) - 1)), [len(words) - 1]]
    pop = ease_out((t - start) / 0.12)
    y = beat.get("caption_y", 1460) + (1 - pop) * 22
    d = ImageDraw.Draw(img)
    for li, ids in enumerate(lines):
        lw = sum(widths[i] for i in ids) + space * (len(ids) - 1)
        x = cv.w / 2 - lw / 2
        for i in ids:
            w = words[i].upper()
            col = hexrgb(accent) if norm(words[i]) in hl else WHITE
            d.text((x, cv.p(y + li * 88)), w, font=fnt, fill=col, anchor="lm", stroke_width=cv.p(9),
                   stroke_fill=(0, 0, 0))
            x += widths[i] + space


def draw_hud(img, cv, ctx, beat, t):
    if beat.get("hud") is False:
        return
    d = ImageDraw.Draw(img)
    circle(d, cv, 62, 92, 8, fill=RED if (t % 1.2) < 0.8 else (90, 20, 30))
    d.text(cv.pt(80, 92), f"PART {ctx['part']}", font=cv.f(MONO, 24), fill=(200, 200, 210), anchor="lm")
    clk = ctx.get("clock", "")
    if clk:
        tm, _, ap = clk.partition(" ")
        fnt = cv.f(MONO, 84)
        if beat.get("clock_alert"):
            wdt = fnt.getlength(tm) / cv.s
            rrect(d, cv, 44, 120, 70 + wdt + 70, 230, 10, fill=(140, 16, 28))
        d.text(cv.pt(54, 176), tm, font=fnt, fill=WHITE, anchor="lm")
        d.text((cv.p(60) + fnt.getlength(tm), cv.p(196)), ap, font=cv.f(MONO, 30), fill=WHITE, anchor="ls")
    d.text(cv.pt(56, 250), ctx["series"].upper(), font=cv.f(MONO, 22), fill=(130, 130, 150), anchor="lm")
    if ctx.get("handle"):
        d.text(cv.pt(1030, 1830), ctx["handle"], font=cv.f(SEMI, 26), fill=(160, 160, 175), anchor="rm")


# ----------------------------------------------------------------------------- post effects

class Post:
    def __init__(self, w, h):
        y, x = np.mgrid[0:h, 0:w]
        dx = (x - w / 2) / (w / 2)
        dy = (y - h / 2) / (h / 2)
        v = 1 - 0.55 * np.clip((dx ** 2 * 0.9 + dy ** 2 * 0.7) - 0.15, 0, 1)
        self.vig = v[..., None].astype(np.float32)
        rng = np.random.default_rng(1)
        self.grain = [rng.normal(0, 3.5, (h, w, 1)).astype(np.float32) for _ in range(6)]
        self.w, self.h = w, h

    def apply(self, img, fi, brightness=1.0, shake=0.0, zoom=1.0):
        if zoom != 1.0 or shake:
            w, h = self.w, self.h
            cw, ch = w / zoom, h / zoom
            ox = (w - cw) / 2 + random.uniform(-shake, shake)
            oy = (h - ch) / 2 + random.uniform(-shake, shake)
            if shake and zoom == 1.0:
                cw, ch = w / 1.03, h / 1.03
                ox = (w - cw) / 2 + random.uniform(-shake, shake)
                oy = (h - ch) / 2 + random.uniform(-shake, shake)
            img = img.resize((w, h), Image.BILINEAR, box=(ox, oy, ox + cw, oy + ch))
        a = np.asarray(img, dtype=np.float32)
        a = a * self.vig * brightness + self.grain[(fi // 3) % len(self.grain)]
        return np.clip(a, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- audio

def tone(f, dur, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * f * t)

def env(n, a=0.005, decay=0.3, sr=SR):
    t = np.arange(n) / sr
    e = np.minimum(1, t / max(a, 1e-4)) * np.exp(-t / max(decay, 1e-4))
    return e

def noise(dur, sr=SR, seed=0):
    return np.random.default_rng(seed).normal(0, 1, int(dur * sr))

def lowpass(x, k):
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")

def sfx(name):
    n = lambda d: int(d * SR)
    if name == "knock":
        out = np.zeros(n(1.3))
        for i in range(3):
            s = i * n(0.38)
            hit = tone(85, 0.25) * env(n(0.25), 0.002, 0.06) + lowpass(noise(0.25, seed=i), 8) * env(n(0.25), 0.001, 0.02) * 0.6
            out[s:s + len(hit)] += hit
        return out * 0.9
    if name == "tap":
        out = np.zeros(n(1.2))
        for i in range(3):
            s = i * n(0.3)
            hit = lowpass(noise(0.08, seed=i + 9), 2) * env(n(0.08), 0.001, 0.012) + tone(900, 0.08) * env(n(0.08), 0.001, 0.02) * 0.4
            out[s:s + len(hit)] += hit
        return out * 0.7
    if name == "buzz":
        out = np.zeros(n(1.0))
        for i in range(2):
            s = i * n(0.5)
            b = np.sign(tone(160, 0.32)) * 0.3 * tone(4, 0.32) ** 2
            out[s:s + len(b)] += b
        return lowpass(out, 6) * 0.8
    if name == "sting":
        d = 2.5
        x = sum(tone(f, d) for f in (110, 116.5, 155.6, 233, 311)) / 5
        x = x * env(n(d), 0.01, 0.9) + lowpass(noise(d, seed=3), 3) * env(n(d), 0.001, 0.15) * 0.5
        return x * 1.1
    if name == "whoosh":
        d = 1.2
        x = lowpass(noise(d, seed=4), 30) * np.sin(np.linspace(0, np.pi, n(d))) * 2.5
        return x
    if name == "brakes":
        d = 1.8
        x = lowpass(noise(d, seed=5), 2) * np.linspace(0.4, 0, n(d)) * 0.5
        x += tone(70, d) * np.linspace(0.6, 0, n(d)) * 0.5
        return x
    if name == "heartbeat":
        out = np.zeros(n(2.4))
        for i in range(3):
            for off in (0, 0.22):
                s = n(i * 0.8 + off)
                h = tone(55, 0.2) * env(n(0.2), 0.005, 0.07)
                out[s:s + len(h)] += h * (1 if off == 0 else 0.7)
        return out * 1.4
    if name == "click":
        return lowpass(noise(0.05, seed=6), 2) * env(n(0.05), 0.0005, 0.006) * 0.9
    if name == "static":
        d = 1.0
        return noise(d, seed=7) * 0.12 * np.linspace(1, 0, n(d))
    if name == "riser":
        d = 2.5
        t = np.arange(n(d)) / SR
        f = 120 + 500 * (t / d) ** 2
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * (t / d) ** 2 * 0.5 + lowpass(noise(d, seed=8), 10) * (t / d) ** 2 * 0.8
    if name == "pop":
        return tone(600, 0.12) * env(n(0.12), 0.001, 0.03) * 0.4
    return np.zeros(1)

def drone(dur, seed=0):
    t = np.arange(int(dur * SR)) / SR
    lfo = 0.6 + 0.4 * np.sin(2 * np.pi * 0.07 * t)
    x = (np.sin(2 * np.pi * 55 * t) + 0.8 * np.sin(2 * np.pi * 55.35 * t) + 0.5 * np.sin(2 * np.pi * 82.4 * t)
         + 0.25 * np.sin(2 * np.pi * 164.8 * t + np.sin(2 * np.pi * 0.2 * t)))
    x = x * lfo * 0.09
    x += lowpass(noise(dur, seed=seed + 11), 60) * 0.35
    # sparse music-box notes for creepiness
    rng = np.random.default_rng(seed)
    notes = [220, 261.6, 311.1, 329.6, 207.7]
    tt = 2.0
    while tt < dur - 1:
        f = rng.choice(notes) * 2
        s = int(tt * SR)
        l = int(1.6 * SR)
        seg = tone(f, 1.6) * env(l, 0.003, 0.5) * 0.05
        x[s:s + l] += seg[: len(x[s:s + l])]
        tt += rng.choice([1.5, 2.0, 3.0])
    return x

def write_wav(path, x):
    x = np.clip(x, -1, 1)
    pcm = (x * 32767).astype(np.int16)
    st = np.stack([pcm, pcm], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(st.tobytes())

def read_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768


async def _tts(text, voice, path, rate):
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate, pitch="-4Hz").save(path)

KOKORO_DIR = os.path.join(HERE, "voice_models")
KOKORO_FILES = {
    "kokoro-v1.0.int8.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

def kokoro_engine():
    """Offline AI narrator (Kokoro). Downloads the voice model (~120 MB) the first time."""
    try:
        from kokoro_onnx import Kokoro
    except ImportError:
        print("  ! narrator needs: pip install kokoro-onnx soundfile")
        return None
    os.makedirs(KOKORO_DIR, exist_ok=True)
    for name, url in KOKORO_FILES.items():
        path = os.path.join(KOKORO_DIR, name)
        if not os.path.exists(path):
            print(f"  downloading voice model {name} (first time only)...")
            import urllib.request
            urllib.request.urlretrieve(url, path + ".part")
            os.replace(path + ".part", path)
    return Kokoro(os.path.join(KOKORO_DIR, "kokoro-v1.0.int8.onnx"), os.path.join(KOKORO_DIR, "voices-v1.0.bin"))

def speakable(text):
    """Make clock times read naturally: '12:13' -> '12 13', '3:07' -> '3 oh 7', 'AM/PM' -> 'A M'."""
    text = re.sub(r"\b(\d{1,2}):00\b", r"\1 o'clock", text)
    text = re.sub(r"\b(\d{1,2}):0(\d)\b", r"\1 oh \2", text)
    text = re.sub(r"\b(\d{1,2}):(\d\d)\b", r"\1 \2", text)
    text = re.sub(r"\bAM\b", "A.M.", text)
    text = re.sub(r"\bPM\b", "P.M.", text)
    return text.replace("...", ",")

def make_voice(beats, voice, tmp, rate):
    lines = [b.get("voice", b.get("say", "")) for b in beats]
    clips = []
    if "Neural" in voice:  # Microsoft edge-tts voice (online)
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            print("  ! edge-tts not installed (pip install edge-tts) - rendering without narrator")
            return None
        for i, say in enumerate(lines):
            if not say:
                clips.append(None)
                continue
            p = os.path.join(tmp, f"v{i}.mp3")
            try:
                asyncio.run(_tts(say, voice, p, rate))
                clips.append(read_audio(p))
            except Exception as e:
                print(f"  ! voice failed ({e.__class__.__name__}: {e}) - rendering without narrator")
                return None
        return clips
    try:
        k = kokoro_engine()
    except Exception as e:
        print(f"  ! could not load narrator ({e}) - rendering without narrator")
        return None
    if k is None:
        return None
    speed = 1.0 + float(rate.strip("%").replace("+", "") or 0) / 100
    for say in lines:
        if not say:
            clips.append(None)
            continue
        a, sr = k.create(speakable(say), voice=voice, speed=speed, lang="en-us")
        a = np.asarray(a, dtype=np.float32)
        a = np.interp(np.arange(int(len(a) * SR / sr)) * sr / SR, np.arange(len(a)), a)
        clips.append(a / max(1e-6, np.max(np.abs(a))) * 0.95)
    return clips


# ----------------------------------------------------------------------------- render

def beat_duration(b, clip):
    if "dur" in b:
        return float(b["dur"])
    words = len(b.get("say", "").split())
    base = words / 2.6 + 0.5
    if clip is not None:
        base = len(clip) / SR + 0.35
    if b.get("scene") == "phone":
        base = max(base, 1.4 * len(b.get("messages", [])) + 0.6)
    if b.get("scene") == "choice":
        base = max(base, 7.0)
    return max(1.6, base)


def render(story_path, out_dir, preview=False, voice=None, rate="+8%"):
    with open(story_path, encoding="utf-8") as f:
        story = json.load(f)
    beats = story["beats"]
    W, H = (540, 960) if preview else (1080, 1920)
    cv = Canvas(W, H)
    base = os.path.splitext(os.path.basename(story_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, base + (".preview" if preview else "") + ".mp4")
    print(f"> {story_path} -> {out}")

    tmp = tempfile.mkdtemp()
    clips = make_voice(beats, voice, tmp, rate) if voice else None
    if clips is None:
        clips = [None] * len(beats)
    for b, c in zip(beats, clips):
        if c is not None:
            b["_speech"] = len(c) / SR
    durs = [beat_duration(b, c) for b, c in zip(beats, clips)]
    total = sum(durs)
    starts = np.cumsum([0] + durs[:-1])

    # audio mix
    mix = drone(total + 1, seed=story.get("part", 1)) * (0.55 if any(c is not None for c in clips) else 1.0)
    def add(x, at, g=1.0):
        s = int(at * SR)
        e = min(len(mix), s + len(x))
        if e > s:
            mix[s:e] += x[: e - s] * g
    for b, st, c, du in zip(beats, starts, clips, durs):
        fx = b.get("sfx", [])
        fx = [fx] if isinstance(fx, str) else fx
        for name in fx:
            add(sfx(name), st + (0.05 if name != "riser" else max(0, du - 2.5)))
        if b.get("scene") == "phone":
            add(sfx("pop"), st + 0.1, 0.6)
        if c is not None:
            add(c, st + 0.1, 1.0)
    mix = mix[: int(total * SR)]
    mix = mix / max(1e-6, np.max(np.abs(mix))) * 0.9
    wav = os.path.join(tmp, "a.wav")
    write_wav(wav, mix)

    ctx = {"series": story.get("series", "The Rules"), "part": story.get("part", 1),
           "accent": story.get("accent", "#c6ff3d"), "handle": story.get("handle", ""), "clock": ""}
    post = Post(W, H)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "veryfast" if preview else "medium",
           "-crf", "22", "-maxrate", "8M", "-bufsize", "16M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest",
           "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    fi = 0
    nframes = int(total * FPS)
    random.seed(story.get("part", 1))
    for bi, (b, st, du) in enumerate(zip(beats, starts, durs)):
        if "clock" in b:
            ctx["clock"] = b["clock"]
        scene = SCENES.get(b.get("scene", "text"))
        if scene is None:
            raise SystemExit(f"Unknown scene '{b.get('scene')}' in beat {bi + 1}. Options: {', '.join(SCENES)}")
        while fi < nframes and fi / FPS < st + du:
            t = fi / FPS - st
            img = Image.new("RGB", (W, H))
            scene(img, cv, b, t, du, ctx)
            draw_hud(img, cv, ctx, b, t)
            draw_caption(img, cv, b, t, du, ctx["accent"])
            br = 1.0
            if b.get("flicker") and random.random() < 0.18:
                br = random.uniform(0.25, 0.7)
            if t < 0.15 and (bi == 0 or b.get("fade")):
                br *= ease(t / 0.15)
            shake = cv.p(b.get("shake", 0) * 10) if b.get("shake") else 0
            zoom = 1.0 + 0.06 * (t / du) if b.get("zoom") else 1.0
            frame = post.apply(img, fi, br, shake, zoom)
            proc.stdin.write(frame.tobytes())
            fi += 1
        print(f"  beat {bi + 1:>2}/{len(beats)}  {b.get('scene','text'):<10} {du:4.1f}s  {b.get('say','')[:50]}")
    proc.stdin.close()
    proc.wait()
    shutil.rmtree(tmp, ignore_errors=True)

    # companion post text with the choice for the comments
    ch = next((b for b in beats if b.get("scene") == "choice"), None)
    txt = os.path.join(out_dir, base + ".post.txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(story.get("post_caption", f"{ctx['series']} - Part {ctx['part']}") + "\n\n")
        if ch:
            f.write(f"A = {ch.get('a')}\nB = {ch.get('b')}\n")
            f.write("Comment A or B - the most liked answer decides the next part.\n\n")
            f.write("Pinned comment idea:\n")
            f.write(f"Vote here! Like this for A ({ch.get('a')}) - reply B if you want: {ch.get('b')}\n\n")
        f.write(" ".join(story.get("hashtags", ["#stickman", "#rules", "#horror", "#storytime", "#choices", "#fyp"])) + "\n")
        f.write("\nNarration script:\n")
        for b in beats:
            if b.get("say"):
                f.write(f"- {b.get('voice', b['say'])}\n")
    print(f"  done: {out}  ({total:.1f}s)\n  post text: {txt}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Render stick man rules videos from story files.")
    ap.add_argument("stories", nargs="+", help="story .json files")
    ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--preview", action="store_true", help="half resolution, faster")
    ap.add_argument("--voice", default="am_onyx",
                    help="narrator voice. Offline Kokoro voices: am_onyx (deep, default), am_michael, am_adam, "
                         "bm_george (British), af_bella, af_nicole (female). Or an edge-tts name like en-US-GuyNeural.")
    ap.add_argument("--no-voice", action="store_true", help="captions + music only, no narrator")
    ap.add_argument("--rate", default="+5%", help="narrator speed, e.g. +0%% or +15%%")
    a = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required: https://ffmpeg.org/download.html")
    for s in a.stories:
        render(s, a.out, a.preview, None if a.no_voice else a.voice, a.rate)


if __name__ == "__main__":
    main()
