"""Generate deterministic PNGs for the synthetic sample fixture.

Run once to populate data/samples/images/.  All images are built with
PIL only (no external assets) so they are fully reproducible offline.

Coverage (50 tasks):
  - UI / web interactions  (click, type, select, scroll)  — 25 tasks
  - Visual QA              (answer)                        — 25 tasks
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
IMG_DIR = HERE / "images"
IMG_DIR.mkdir(exist_ok=True)

W, H = 320, 200   # standard canvas


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _canvas(bg: tuple = (245, 245, 245)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), bg)
    return img, ImageDraw.Draw(img)


def _button(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
            label: str, bg=(70, 130, 180), fg=(255, 255, 255), radius: int = 6) -> None:
    d.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=bg, outline=(40, 80, 120), width=2)
    font = _font(14)
    bbox = d.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((x + (w - tw) // 2, y + (h - th) // 2), label, fill=fg, font=font)


def _input_field(d: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
                 placeholder: str = "", value: str = "") -> None:
    d.rectangle([x, y, x + w, y + h], fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    text = value or placeholder
    color = (120, 120, 120) if not value else (30, 30, 30)
    d.text((x + 8, y + (h - 14) // 2), text, fill=color, font=_font(13))


def _label(d: ImageDraw.ImageDraw, x: int, y: int, text: str,
           color=(50, 50, 50), size: int = 13) -> None:
    d.text((x, y), text, fill=color, font=_font(size))


def _save(img: Image.Image, name: str) -> Path:
    p = IMG_DIR / name
    img.save(p, format="PNG")
    return p


# ---------------------------------------------------------------------------
# UI / web images (25)
# ---------------------------------------------------------------------------

def ui_login():
    img, d = _canvas((230, 240, 255))
    _label(d, 10, 8, "Sign in to your account", size=15, color=(30, 30, 80))
    _label(d, 10, 42, "Email")
    _input_field(d, 10, 58, 200, 28, "you@example.com")
    _label(d, 10, 96, "Password")
    _input_field(d, 10, 112, 200, 28, "••••••••")
    _button(d, 10, 152, 100, 32, "Log in", bg=(60, 100, 200))
    _button(d, 120, 152, 100, 32, "Cancel", bg=(160, 160, 160))
    return _save(img, "login.png")


def ui_submit():
    img, d = _canvas((230, 255, 230))
    _label(d, 10, 8, "Submit your response", size=15, color=(30, 80, 30))
    _label(d, 10, 42, "Your answer:")
    _input_field(d, 10, 58, 290, 28, "Type here…")
    _label(d, 10, 100, "Agree to terms?")
    d.rectangle([10, 116, 26, 132], fill=(255, 255, 255), outline=(100, 100, 100), width=2)
    _label(d, 34, 115, "I agree to the Terms of Service", size=12)
    _button(d, 10, 152, 120, 32, "Submit", bg=(40, 160, 40))
    _button(d, 140, 152, 80, 32, "Reset", bg=(180, 80, 40))
    return _save(img, "submit.png")


def ui_search():
    img, d = _canvas()
    _label(d, 10, 8, "Search", size=16, color=(30, 30, 30))
    _input_field(d, 10, 40, 240, 32, "Search products…")
    _button(d, 258, 40, 50, 32, "🔍 Go")
    _label(d, 10, 90, "Recent searches:", size=12, color=(100, 100, 100))
    for i, term in enumerate(["laptop", "keyboard", "monitor"]):
        _label(d, 20, 110 + i * 20, f"• {term}", size=12, color=(70, 70, 200))
    return _save(img, "search.png")


def ui_dropdown():
    img, d = _canvas()
    _label(d, 10, 10, "Sort results by:", size=14)
    d.rectangle([10, 36, 200, 64], fill=(255, 255, 255), outline=(150, 150, 150), width=2)
    _label(d, 18, 44, "Price: Low to High", size=13)
    d.polygon([(186, 46), (196, 46), (191, 56)], fill=(80, 80, 80))
    for i, opt in enumerate(["Relevance", "Price: Low to High",
                              "Price: High to Low", "Customer Rating"]):
        y = 68 + i * 26
        bg = (220, 230, 255) if i == 1 else (255, 255, 255)
        d.rectangle([10, y, 200, y + 24], fill=bg, outline=(200, 200, 200))
        _label(d, 18, y + 5, opt, size=12)
    return _save(img, "dropdown.png")


def ui_navbar():
    img, d = _canvas((40, 60, 100))
    for i, item in enumerate(["Home", "Products", "About", "Contact"]):
        x = 10 + i * 76
        bg = (255, 255, 255) if item == "Home" else (40, 60, 100)
        fg = (30, 30, 30) if item == "Home" else (220, 220, 220)
        d.rectangle([x, 10, x + 68, 40], fill=bg, outline=(100, 120, 160))
        _label(d, x + 8, 18, item, color=fg, size=13)
    d.rectangle([0, 50, W, 200], fill=(245, 245, 245))
    _label(d, 10, 70, "Welcome to our store!", size=15, color=(30, 30, 80))
    _label(d, 10, 100, "Browse our latest collection below.", size=12)
    return _save(img, "navbar.png")


def ui_signup():
    img, d = _canvas((250, 230, 255))
    _label(d, 10, 8, "Create Account", size=16, color=(80, 30, 120))
    _label(d, 10, 42, "Full Name")
    _input_field(d, 10, 58, 290, 26)
    _label(d, 10, 92, "Email")
    _input_field(d, 10, 108, 290, 26)
    _label(d, 10, 140, "Password")
    _input_field(d, 10, 156, 290, 26, "Min. 8 characters")
    _button(d, 90, W // 2 - 20, 140, 32, "Sign Up", bg=(120, 40, 180))
    return _save(img, "signup.png")


def ui_cart():
    img, d = _canvas()
    _label(d, 10, 8, "Shopping Cart  (2 items)", size=15, color=(30, 30, 30))
    d.line([0, 32, W, 32], fill=(200, 200, 200), width=1)
    items = [("Mechanical Keyboard", "$89.99"), ("USB-C Hub", "$34.99")]
    for i, (name, price) in enumerate(items):
        y = 44 + i * 44
        d.rectangle([6, y, W - 6, y + 36], fill=(255, 255, 255), outline=(210, 210, 210))
        _label(d, 14, y + 10, name, size=13)
        _label(d, W - 70, y + 10, price, size=13, color=(180, 40, 40))
    _button(d, 10, 148, 140, 36, "Add to Cart", bg=(255, 153, 0), fg=(30, 30, 30))
    _button(d, 162, 148, 148, 36, "Proceed to Checkout", bg=(40, 160, 40))
    return _save(img, "cart.png")


def ui_email_form():
    img, d = _canvas()
    _label(d, 10, 8, "Contact Us", size=16, color=(30, 30, 80))
    _label(d, 10, 40, "Your Email Address")
    _input_field(d, 10, 58, 290, 28, "email@domain.com")
    _label(d, 10, 96, "Subject")
    _input_field(d, 10, 112, 290, 28, "Enter subject")
    _label(d, 10, 148, "Message")
    d.rectangle([10, 164, 300, 190], fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    _label(d, 18, 170, "Type your message here…", size=12, color=(160, 160, 160))
    return _save(img, "email_form.png")


def ui_password():
    img, d = _canvas()
    _label(d, 10, 8, "Change Password", size=15, color=(30, 30, 80))
    for i, lbl in enumerate(["Current Password", "New Password", "Confirm Password"]):
        y = 42 + i * 46
        _label(d, 10, y, lbl)
        _input_field(d, 10, y + 16, 260, 26, "••••••••")
    _button(d, 10, 166, 140, 28, "Update Password", bg=(60, 100, 200))
    return _save(img, "password.png")


def ui_checkbox():
    img, d = _canvas()
    _label(d, 10, 10, "Account Preferences", size=15, color=(30, 30, 80))
    opts = [
        (True,  "Receive newsletter"),
        (False, "Enable two-factor authentication"),
        (True,  "Show profile publicly"),
        (False, "Dark mode"),
    ]
    for i, (checked, text) in enumerate(opts):
        y = 44 + i * 34
        d.rectangle([10, y, 26, y + 16], fill=(255, 255, 255), outline=(120, 120, 120), width=2)
        if checked:
            d.line([12, y + 8, 17, y + 13], fill=(40, 160, 40), width=3)
            d.line([17, y + 13, 24, y + 4], fill=(40, 160, 40), width=3)
        _label(d, 34, y + 1, text, size=13)
    return _save(img, "checkbox.png")


def ui_close():
    img, d = _canvas()
    _label(d, 10, 10, "Confirmation Dialog", size=15, color=(30, 30, 80))
    d.rectangle([10, 38, W - 10, H - 10], fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    d.text((W - 30, 14), "✕", fill=(160, 60, 60), font=_font(18))
    _label(d, 20, 52, "Are you sure you want to delete", size=13)
    _label(d, 20, 70, "this item? This cannot be undone.", size=13)
    _button(d, 20, 110, 100, 30, "Cancel", bg=(160, 160, 160))
    _button(d, 134, 110, 120, 30, "Delete", bg=(200, 50, 50))
    return _save(img, "close.png")


def ui_pagination():
    img, d = _canvas()
    _label(d, 10, 10, "Search Results  (Page 3 of 8)", size=14)
    for i in range(1, 6):
        x = 10 + (i - 1) * 52
        bg = (60, 100, 200) if i == 3 else (240, 240, 240)
        fg = (255, 255, 255) if i == 3 else (60, 60, 60)
        _button(d, x, 50, 44, 32, str(i), bg=bg, fg=fg)
    _button(d, 10, 104, 80, 30, "◀ Prev", bg=(240, 240, 240), fg=(60, 60, 60))
    _button(d, 100, 104, 80, 30, "Next ▶", bg=(60, 100, 200))
    return _save(img, "pagination.png")


def ui_download():
    img, d = _canvas((230, 255, 245))
    _label(d, 10, 10, "File Downloads", size=15, color=(20, 80, 60))
    files = [("report_2024.pdf", "1.2 MB"), ("data.csv", "340 KB"), ("slides.pptx", "5.7 MB")]
    for i, (name, size) in enumerate(files):
        y = 44 + i * 40
        _label(d, 10, y + 4, f"📄 {name}", size=13)
        _label(d, 220, y + 4, size, size=11, color=(120, 120, 120))
        _button(d, 248, y, 60, 26, "⬇ Save", bg=(40, 160, 120), fg=(255, 255, 255))
    return _save(img, "download.png")


def ui_delete():
    img, d = _canvas()
    _label(d, 10, 10, "Manage Files", size=15, color=(30, 30, 80))
    files = ["project_draft.docx", "old_backup.zip", "temp_notes.txt"]
    for i, name in enumerate(files):
        y = 44 + i * 42
        d.rectangle([6, y, W - 6, y + 34], fill=(255, 255, 255), outline=(210, 210, 210))
        _label(d, 14, y + 9, name, size=13)
        _button(d, W - 76, y + 5, 64, 24, "🗑 Delete", bg=(200, 50, 50), fg=(255, 255, 255))
    return _save(img, "delete.png")


def ui_save():
    img, d = _canvas()
    _label(d, 10, 10, "Edit Profile", size=15, color=(30, 30, 80))
    _label(d, 10, 44, "Display Name")
    _input_field(d, 10, 60, 290, 26, value="Alex Johnson")
    _label(d, 10, 96, "Bio")
    d.rectangle([10, 112, 300, 148], fill=(255, 255, 255), outline=(180, 180, 180), width=2)
    _label(d, 18, 118, "Software engineer & coffee enthusiast.", size=12)
    _button(d, 10, 160, 80, 28, "💾 Save", bg=(60, 100, 200))
    _button(d, 100, 160, 80, 28, "Discard", bg=(160, 160, 160))
    return _save(img, "save.png")


def ui_upload():
    img, d = _canvas((240, 248, 255))
    _label(d, 10, 10, "Upload Document", size=15, color=(30, 30, 80))
    d.rectangle([20, 44, W - 20, 130], fill=(255, 255, 255),
                outline=(100, 140, 220), width=2)
    _label(d, 80, 60, "⬆", size=28, color=(100, 140, 220))
    _label(d, 60, 96, "Drag & drop or click to browse", size=12, color=(120, 120, 160))
    _button(d, 80, 144, 160, 32, "📂 Choose File", bg=(80, 120, 200))
    return _save(img, "upload.png")


def ui_filter():
    img, d = _canvas()
    _label(d, 10, 8, "Filter Products", size=15, color=(30, 30, 80))
    _label(d, 10, 36, "Category")
    d.rectangle([10, 52, 200, 76], fill=(255, 255, 255), outline=(150, 150, 150), width=2)
    _label(d, 18, 58, "All Categories  ▾", size=13)
    _label(d, 10, 86, "Price Range")
    d.rectangle([10, 102, 200, 126], fill=(255, 255, 255), outline=(150, 150, 150), width=2)
    _label(d, 18, 108, "$0 – $500  ▾", size=13)
    _label(d, 10, 136, "In Stock Only")
    d.rectangle([10, 152, 26, 168], fill=(255, 255, 255), outline=(120, 120, 120), width=2)
    _button(d, 60, 150, 120, 30, "Apply Filters", bg=(60, 100, 200))
    return _save(img, "filter.png")


def ui_sort():
    img, d = _canvas()
    _label(d, 10, 8, "Sort by:", size=14, color=(50, 50, 50))
    options = ["Name (A–Z)", "Date Modified", "File Size", "Type"]
    for i, opt in enumerate(options):
        y = 36 + i * 36
        selected = (i == 1)
        bg = (220, 230, 255) if selected else (255, 255, 255)
        d.rectangle([10, y, 220, y + 28], fill=bg, outline=(180, 180, 200), width=2)
        dot = "●" if selected else "○"
        _label(d, 18, y + 6, f"{dot}  {opt}", size=13,
               color=(60, 80, 180) if selected else (60, 60, 60))
    return _save(img, "sort.png")


def ui_settings():
    img, d = _canvas()
    _label(d, 10, 8, "⚙  Settings", size=16, color=(30, 30, 30))
    items = ["Account", "Notifications", "Privacy", "Help & Support"]
    for i, item in enumerate(items):
        y = 40 + i * 36
        d.rectangle([6, y, W - 6, y + 28], fill=(255, 255, 255), outline=(210, 210, 210))
        _label(d, 16, y + 7, item, size=13)
        _label(d, W - 26, y + 7, "›", size=16, color=(160, 160, 160))
    return _save(img, "settings.png")


def ui_profile():
    img, d = _canvas((240, 245, 255))
    d.ellipse([10, 16, 70, 76], fill=(180, 200, 240), outline=(100, 130, 200), width=3)
    _label(d, 22, 36, "AJ", size=20, color=(60, 80, 160))
    _label(d, 80, 20, "Alex Johnson", size=15, color=(30, 30, 80))
    _label(d, 80, 42, "alex.j@example.com", size=12, color=(100, 100, 100))
    _label(d, 80, 60, "Member since 2021", size=11, color=(140, 140, 140))
    d.line([0, 90, W, 90], fill=(200, 200, 210), width=1)
    _button(d, 10, 104, 140, 30, "Edit Profile", bg=(60, 100, 200))
    _button(d, 160, 104, 140, 30, "Log Out", bg=(200, 60, 60))
    return _save(img, "profile.png")


def ui_scroll():
    img, d = _canvas()
    _label(d, 10, 8, "News Feed", size=15, color=(30, 30, 80))
    headlines = [
        "Scientists discover new exoplanet",
        "Stock markets reach record high",
        "New AI model breaks benchmarks",
        "Local team wins championship",
    ]
    for i, h in enumerate(headlines):
        y = 36 + i * 36
        d.line([0, y - 4, W, y - 4], fill=(220, 220, 220), width=1)
        _label(d, 10, y + 2, h, size=12)
        _label(d, 10, y + 18, "2 hours ago", size=10, color=(150, 150, 150))
    d.rectangle([W - 14, 0, W, H], fill=(230, 230, 230))
    d.rectangle([W - 13, 20, W - 1, 60], fill=(150, 150, 150), outline=(120, 120, 120))
    return _save(img, "scroll.png")


def ui_terms():
    img, d = _canvas()
    _label(d, 10, 8, "Terms and Conditions", size=14, color=(30, 30, 80))
    lines = [
        "1. You agree to use this service lawfully.",
        "2. We respect your privacy.",
        "3. Content may not be reproduced.",
        "4. Service may change without notice.",
    ]
    for i, line in enumerate(lines):
        _label(d, 10, 36 + i * 22, line, size=11, color=(60, 60, 60))
    d.rectangle([10, 130, 26, 146], fill=(255, 255, 255), outline=(100, 100, 100), width=2)
    _label(d, 34, 130, "I have read and accept the Terms", size=12)
    _button(d, 10, 158, 140, 30, "Continue", bg=(60, 100, 200))
    return _save(img, "terms.png")


def ui_notification():
    img, d = _canvas()
    _label(d, 10, 8, "🔔  Notifications  (3 unread)", size=14, color=(30, 30, 80))
    items = [
        ("Alice liked your post", "2 min ago", True),
        ("Bob started following you", "1 hr ago", True),
        ("Your report is ready", "3 hrs ago", True),
        ("System update completed", "Yesterday", False),
    ]
    for i, (msg, time, unread) in enumerate(items):
        y = 38 + i * 38
        bg = (230, 238, 255) if unread else (255, 255, 255)
        d.rectangle([6, y, W - 6, y + 30], fill=bg, outline=(200, 208, 230))
        if unread:
            d.ellipse([W - 20, y + 11, W - 10, y + 21], fill=(60, 100, 200))
        _label(d, 14, y + 6, msg, size=12)
        _label(d, 14, y + 20, time, size=10, color=(140, 140, 140))
    return _save(img, "notification.png")


def ui_rating():
    img, d = _canvas()
    _label(d, 10, 8, "Rate this product", size=15, color=(30, 30, 80))
    _label(d, 10, 36, "Wireless Headphones XZ-400", size=13)
    for i in range(5):
        x = 10 + i * 40
        color = (255, 200, 0) if i < 4 else (210, 210, 210)
        d.polygon([
            (x + 20, 70), (x + 26, 88), (x + 40, 88),
            (x + 28, 98), (x + 33, 116), (x + 20, 106),
            (x + 7, 116), (x + 12, 98), (x, 88), (x + 14, 88)
        ], fill=color, outline=(180, 150, 0))
    _label(d, 10, 130, "4 out of 5 stars", size=12, color=(100, 100, 100))
    _button(d, 10, 156, 120, 30, "Submit Rating", bg=(60, 100, 200))
    return _save(img, "rating.png")


# ---------------------------------------------------------------------------
# Visual QA images (25)
# ---------------------------------------------------------------------------

def qa_diagram():
    """Existing H2O molecule."""
    img, d = _canvas((255, 255, 220))
    _label(d, 10, 8, "Identify the molecule:", size=14, color=(30, 30, 80))
    cx, cy = 160, 110
    d.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], fill=(200, 200, 255), outline=(60, 60, 180), width=3)
    _label(d, cx - 8, cy - 10, "O", size=18, color=(60, 60, 180))
    for dx in [-60, 60]:
        d.ellipse([cx + dx - 14, cy + 20, cx + dx + 14, cy + 48],
                  fill=(255, 200, 200), outline=(180, 60, 60), width=2)
        _label(d, cx + dx - 6, cy + 26, "H", size=14, color=(180, 60, 60))
        d.line([cx + dx // 3, cy + 12, cx + dx - 2, cy + 24], fill=(80, 80, 80), width=2)
    _label(d, 10, 170, 'Formula: H₂O  (water)', size=12, color=(80, 80, 80))
    return _save(img, "diagram.png")


def qa_barchart():
    img, d = _canvas((250, 250, 255))
    _label(d, 10, 6, "Monthly Sales ($k)  — Q1 2024", size=13, color=(30, 30, 80))
    bars = [("Jan", 42, (100, 149, 237)), ("Feb", 68, (100, 200, 100)), ("Mar", 55, (237, 149, 100))]
    max_val = 80
    for i, (month, val, color) in enumerate(bars):
        x = 30 + i * 90
        bar_h = int((val / max_val) * 110)
        y0 = 160 - bar_h
        d.rectangle([x, y0, x + 60, 160], fill=color, outline=(80, 80, 80))
        _label(d, x + 18, y0 - 18, f"${val}k", size=11)
        _label(d, x + 20, 164, month, size=12)
    d.line([20, 30, 20, 165], fill=(80, 80, 80), width=2)
    d.line([20, 165, 295, 165], fill=(80, 80, 80), width=2)
    return _save(img, "barchart.png")


def qa_piechart():
    img, d = _canvas((255, 255, 240))
    _label(d, 10, 6, "Budget Allocation 2024", size=14, color=(30, 30, 80))
    cx, cy, r = 120, 115, 80
    slices = [
        ("Engineering", 40, (100, 149, 237), 0),
        ("Marketing",   25, (100, 200, 100), 144),
        ("Operations",  20, (237, 200, 100), 234),
        ("R&D",         15, (200, 100, 200), 306),
    ]
    for name, pct, color, start in slices:
        extent = pct * 3.6
        d.pieslice([cx - r, cy - r, cx + r, cy + r],
                   start=start, end=start + extent, fill=color, outline=(255, 255, 255), width=2)
    for i, (name, pct, color, _) in enumerate(slices):
        y = 28 + i * 36
        d.rectangle([220, y, 236, y + 14], fill=color, outline=(80, 80, 80))
        _label(d, 242, y, f"{name} {pct}%", size=11)
    return _save(img, "piechart.png")


def qa_thermometer():
    img, d = _canvas((240, 248, 255))
    _label(d, 10, 8, "What temperature is shown?", size=13, color=(30, 30, 80))
    tx, ty = 160, 100
    d.ellipse([tx - 18, ty + 50, tx + 18, ty + 86], fill=(220, 60, 60), outline=(160, 40, 40), width=3)
    d.rectangle([tx - 8, ty - 60, tx + 8, ty + 68], fill=(255, 255, 255), outline=(160, 160, 180), width=2)
    d.rectangle([tx - 5, ty + 10, tx + 5, ty + 66], fill=(220, 60, 60))
    for i, temp in enumerate([0, 20, 40, 60, 80, 100]):
        y = ty + 8 - i * 12
        d.line([tx + 8, y, tx + 16, y], fill=(80, 80, 80), width=1)
        _label(d, tx + 18, y - 6, f"{temp}°", size=9, color=(60, 60, 60))
    _label(d, 10, 170, "Scale: Celsius", size=11, color=(120, 120, 120))
    return _save(img, "thermometer.png")


def qa_clock():
    img, d = _canvas()
    _label(d, 10, 6, "What time does the clock show?", size=13, color=(30, 30, 80))
    cx, cy, r = 160, 110, 72
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255), outline=(60, 60, 60), width=4)
    for h in range(12):
        angle = math.radians(h * 30 - 90)
        x1, y1 = cx + (r - 10) * math.cos(angle), cy + (r - 10) * math.sin(angle)
        x2, y2 = cx + (r - 4) * math.cos(angle), cy + (r - 4) * math.sin(angle)
        d.line([x1, y1, x2, y2], fill=(60, 60, 60), width=2)
        _label(d, cx + (r - 22) * math.cos(angle) - 6,
               cy + (r - 22) * math.sin(angle) - 8, str(h or 12), size=11)
    hour_a = math.radians(3 * 30 - 90)  # 3 o'clock
    min_a = math.radians(0 * 6 - 90)    # :00
    d.line([cx, cy, cx + 44 * math.cos(hour_a), cy + 44 * math.sin(hour_a)],
           fill=(30, 30, 30), width=5)
    d.line([cx, cy, cx + 60 * math.cos(min_a), cy + 60 * math.sin(min_a)],
           fill=(60, 60, 60), width=3)
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(30, 30, 30))
    return _save(img, "clock.png")


def qa_trafficlight():
    img, d = _canvas((200, 200, 200))
    _label(d, 10, 8, "What does the traffic light show?", size=13)
    d.rectangle([130, 30, 190, 170], fill=(40, 40, 40), outline=(20, 20, 20), width=3)
    lights = [(60, 60, 60), (60, 60, 60), (40, 200, 40)]
    centers = [60, 100, 140]
    for i, (color, cy) in enumerate(zip(lights, centers)):
        d.ellipse([140, cy, 180, cy + 40], fill=color, outline=(20, 20, 20), width=2)
    _label(d, 10, 80, "Red", size=12, color=(180, 60, 60))
    _label(d, 10, 110, "Yellow", size=12, color=(180, 160, 0))
    _label(d, 10, 140, "Green ←", size=12, color=(40, 160, 40))
    return _save(img, "trafficlight.png")


def qa_flag():
    img, d = _canvas()
    _label(d, 10, 8, "What country does this flag represent?", size=12, color=(30, 30, 80))
    # French tricolour
    fw, fh = 180, 110
    fx, fy = 70, 40
    d.rectangle([fx, fy, fx + fw // 3, fy + fh], fill=(0, 35, 149))
    d.rectangle([fx + fw // 3, fy, fx + 2 * fw // 3, fy + fh], fill=(255, 255, 255))
    d.rectangle([fx + 2 * fw // 3, fy, fx + fw, fy + fh], fill=(237, 41, 57))
    d.rectangle([fx, fy, fx + fw, fy + fh], outline=(80, 80, 80), width=2)
    _label(d, 10, 165, "Hint: blue, white, red vertical stripes", size=11, color=(100, 100, 100))
    return _save(img, "flag.png")


def qa_shapes():
    img, d = _canvas((255, 255, 240))
    _label(d, 10, 8, "How many triangles are in the image?", size=13, color=(30, 30, 80))
    positions = [(60, 80), (150, 80), (240, 80), (105, 130), (195, 130)]
    for px, py in positions:
        d.polygon([(px, py + 40), (px - 25, py + 80), (px + 25, py + 80)],
                  fill=(100, 160, 240), outline=(40, 80, 180), width=2)
    return _save(img, "shapes.png")


def qa_colors():
    img, d = _canvas()
    _label(d, 10, 8, "What color is the largest circle?", size=13, color=(30, 30, 80))
    d.ellipse([60, 40, 200, 180], fill=(220, 60, 60), outline=(160, 30, 30), width=3)
    d.ellipse([10, 130, 70, 190], fill=(60, 120, 220), outline=(30, 60, 160), width=2)
    d.ellipse([230, 40, 310, 120], fill=(60, 180, 60), outline=(30, 120, 30), width=2)
    return _save(img, "colors.png")


def qa_count():
    img, d = _canvas((255, 250, 240))
    _label(d, 10, 8, "How many stars are visible?", size=13, color=(30, 30, 80))
    positions = [(60, 60), (130, 45), (200, 70), (80, 120), (160, 110),
                 (240, 90), (110, 160), (185, 155)]
    for px, py in positions:
        pts = []
        for k in range(10):
            r = 16 if k % 2 == 0 else 7
            a = math.radians(k * 36 - 90)
            pts.extend([px + r * math.cos(a), py + r * math.sin(a)])
        d.polygon(pts, fill=(255, 210, 0), outline=(200, 160, 0), width=1)
    return _save(img, "count.png")


def qa_equation():
    img, d = _canvas((245, 255, 245))
    _label(d, 10, 8, "What is the result of this equation?", size=13, color=(30, 30, 80))
    _label(d, 40, 60, "3 × 7 + 4 = ?", size=32, color=(30, 30, 30))
    _label(d, 40, 120, "Show your reasoning:", size=12, color=(100, 100, 100))
    _label(d, 40, 140, "3 × 7 = 21   →   21 + 4 = 25", size=13, color=(60, 100, 60))
    return _save(img, "equation.png")


def qa_direction():
    img, d = _canvas()
    _label(d, 10, 8, "Which direction is the arrow pointing?", size=13, color=(30, 30, 80))
    cx, cy = 160, 110
    d.polygon([(cx, 50), (cx + 30, 90), (cx + 12, 90), (cx + 12, 170),
               (cx - 12, 170), (cx - 12, 90), (cx - 30, 90)],
              fill=(60, 100, 200), outline=(30, 60, 140), width=2)
    return _save(img, "direction.png")


def qa_co2():
    img, d = _canvas((255, 255, 220))
    _label(d, 10, 8, "Name this molecule:", size=14, color=(30, 30, 80))
    cx, cy = 160, 100
    for dx in [-80, 80]:
        d.ellipse([cx + dx - 18, cy - 18, cx + dx + 18, cy + 18],
                  fill=(255, 160, 160), outline=(180, 60, 60), width=3)
        _label(d, cx + dx - 8, cy - 9, "O", size=16, color=(160, 40, 40))
        d.line([cx + dx // 4, cy, cx + dx - 20, cy], fill=(80, 80, 80), width=2)
    d.ellipse([cx - 18, cy - 18, cx + 18, cy + 18],
              fill=(200, 200, 200), outline=(80, 80, 80), width=3)
    _label(d, cx - 8, cy - 9, "C", size=16, color=(60, 60, 60))
    _label(d, 10, 155, "Linear molecule  |  3 atoms", size=11, color=(100, 100, 100))
    return _save(img, "co2.png")


def qa_element():
    img, d = _canvas((240, 248, 240))
    _label(d, 10, 8, "What chemical element is shown?", size=13, color=(30, 30, 80))
    d.rectangle([70, 40, 250, 160], fill=(255, 215, 0), outline=(180, 150, 0), width=4)
    _label(d, 130, 50, "Au", size=40, color=(120, 90, 0))
    _label(d, 108, 110, "Gold", size=16, color=(100, 70, 0))
    _label(d, 80, 130, "Atomic Number: 79", size=12, color=(120, 100, 0))
    return _save(img, "element.png")


def qa_map():
    img, d = _canvas((180, 210, 255))
    _label(d, 10, 8, "Which continent is highlighted?", size=13, color=(30, 30, 80))
    # Africa silhouette (rough)
    pts = [
        150, 30, 190, 32, 210, 50, 215, 80, 200, 110, 210, 140,
        195, 165, 170, 175, 145, 165, 125, 150, 110, 120, 115, 90,
        130, 60, 140, 42
    ]
    d.polygon(pts, fill=(100, 200, 100), outline=(40, 140, 40), width=3)
    _label(d, 148, 98, "?", size=22, color=(30, 80, 30))
    _label(d, 10, 170, "Largest continent by area on this view", size=11, color=(60, 80, 120))
    return _save(img, "map.png")


def qa_chart_highest():
    img, d = _canvas((250, 255, 250))
    _label(d, 10, 6, "Which city has the highest temperature?", size=12, color=(30, 30, 80))
    data = [("NYC", 22, (100, 149, 237)), ("LA", 31, (237, 149, 100)),
            ("CHI", 18, (149, 200, 149)), ("HOU", 28, (200, 149, 237))]
    max_val = 40
    for i, (city, val, color) in enumerate(data):
        x = 20 + i * 72
        bar_h = int((val / max_val) * 110)
        y0 = 155 - bar_h
        d.rectangle([x, y0, x + 52, 155], fill=color, outline=(80, 80, 80))
        _label(d, x + 14, y0 - 16, f"{val}°C", size=10)
        _label(d, x + 14, 158, city, size=11)
    d.line([12, 28, 12, 158], fill=(80, 80, 80), width=2)
    d.line([12, 158, 302, 158], fill=(80, 80, 80), width=2)
    return _save(img, "chart_highest.png")


def qa_largest_slice():
    img, d = _canvas((255, 248, 240))
    _label(d, 10, 6, "Which department has the largest budget?", size=11, color=(30, 30, 80))
    cx, cy, r = 115, 115, 80
    slices = [
        ("Engineering", 45, (100, 149, 237), 0),
        ("Marketing",   20, (237, 149, 100), 162),
        ("HR",          15, (149, 200, 100), 234),
        ("Legal",       20, (200, 149, 237), 288),
    ]
    for name, pct, color, start in slices:
        extent = pct * 3.6
        d.pieslice([cx - r, cy - r, cx + r, cy + r],
                   start=start, end=start + extent, fill=color, outline=(255, 255, 255), width=2)
    for i, (name, pct, color, _) in enumerate(slices):
        y = 28 + i * 36
        d.rectangle([216, y, 232, y + 14], fill=color, outline=(80, 80, 80))
        _label(d, 238, y, f"{name} {pct}%", size=11)
    return _save(img, "largest_slice.png")


def qa_logo_color():
    img, d = _canvas()
    _label(d, 10, 8, "What color is the company logo?", size=13, color=(30, 30, 80))
    d.ellipse([95, 50, 225, 150], fill=(220, 30, 30), outline=(160, 20, 20), width=4)
    _label(d, 120, 78, "ACME", size=28, color=(255, 255, 255))
    _label(d, 75, 162, "Official brand color shown above", size=12, color=(100, 100, 100))
    return _save(img, "logo_color.png")


def qa_next_number():
    img, d = _canvas((245, 245, 255))
    _label(d, 10, 8, "What is the next number in the sequence?", size=12, color=(30, 30, 80))
    nums = [2, 4, 8, 16, 32, "?"]
    for i, n in enumerate(nums):
        x = 20 + i * 48
        color = (60, 100, 200) if str(n) != "?" else (200, 60, 60)
        d.rounded_rectangle([x, 60, x + 38, 100], radius=6,
                             fill=(220, 230, 255) if str(n) != "?" else (255, 220, 220),
                             outline=color, width=2)
        _label(d, x + 8, 70, str(n), size=16, color=color)
    _label(d, 10, 120, "Pattern: each number is doubled", size=12, color=(100, 100, 100))
    return _save(img, "sequence.png")


def qa_planet():
    img, d = _canvas((20, 20, 60))
    _label(d, 10, 8, "Which planet is shown?", size=13, color=(200, 200, 255))
    cx, cy = 160, 115
    d.ellipse([cx - 60, cy - 60, cx + 60, cy + 60],
              fill=(210, 180, 120), outline=(180, 150, 90), width=3)
    d.ellipse([cx - 100, cy - 22, cx + 100, cy + 22],
              outline=(200, 170, 110), width=6, fill=None)
    for star in [(40, 40), (280, 60), (30, 160), (290, 150), (150, 170), (60, 100)]:
        d.ellipse([star[0] - 2, star[1] - 2, star[0] + 2, star[1] + 2], fill=(255, 255, 255))
    _label(d, 10, 170, "Famous for its ring system", size=11, color=(160, 160, 200))
    return _save(img, "planet.png")


def qa_weather():
    img, d = _canvas((135, 185, 235))
    _label(d, 10, 8, "What weather condition is shown?", size=13, color=(20, 20, 80))
    d.ellipse([80, 50, 180, 130], fill=(255, 255, 200), outline=(240, 220, 0), width=4)
    for angle in range(0, 360, 45):
        a = math.radians(angle)
        x1 = 130 + 56 * math.cos(a)
        y1 = 90 + 56 * math.sin(a)
        x2 = 130 + 72 * math.cos(a)
        y2 = 90 + 72 * math.sin(a)
        d.line([x1, y1, x2, y2], fill=(240, 220, 0), width=4)
    _label(d, 40, 155, "Sky condition: clear and bright", size=12, color=(20, 40, 80))
    return _save(img, "weather.png")


def qa_fruit():
    img, d = _canvas((240, 255, 240))
    _label(d, 10, 8, "How many apples are shown?", size=13, color=(30, 60, 30))
    positions = [(70, 80), (160, 70), (250, 85), (115, 130), (205, 125)]
    for px, py in positions:
        d.ellipse([px - 28, py - 28, px + 28, py + 28],
                  fill=(220, 40, 40), outline=(160, 20, 20), width=2)
        d.line([px, py - 28, px, py - 42], fill=(60, 120, 30), width=3)
        d.arc([px - 4, py - 44, px + 10, py - 32], start=200, end=340,
              fill=(60, 120, 30), width=2)
    return _save(img, "fruit.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

GENERATORS = [
    # UI / web
    ui_login, ui_submit, ui_search, ui_dropdown, ui_navbar,
    ui_signup, ui_cart, ui_email_form, ui_password, ui_checkbox,
    ui_close, ui_pagination, ui_download, ui_delete, ui_save,
    ui_upload, ui_filter, ui_sort, ui_settings, ui_profile,
    ui_scroll, ui_terms, ui_notification, ui_rating,
    # QA / visual
    qa_diagram, qa_barchart, qa_piechart, qa_thermometer, qa_clock,
    qa_trafficlight, qa_flag, qa_shapes, qa_colors, qa_count,
    qa_equation, qa_direction, qa_co2, qa_element, qa_map,
    qa_chart_highest, qa_largest_slice, qa_logo_color,
    qa_next_number, qa_planet, qa_weather, qa_fruit,
]


def main() -> None:
    for fn in GENERATORS:
        path = fn()
        print(f"  wrote {path.name}")
    print(f"\n{len(GENERATORS)} images written to {IMG_DIR}")


if __name__ == "__main__":
    main()
