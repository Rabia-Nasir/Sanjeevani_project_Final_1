import os
import math
from PIL import Image, ImageDraw

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python code", "static")
os.makedirs(static_dir, exist_ok=True)

def create_sanjeevani_icon(size):
    # Create high-res canvas (4x supersampling for ultra-crisp antialiasing)
    supersample = 4
    s = size * supersample
    img = Image.new("RGBA", (s, s), (251, 251, 253, 255))
    draw = ImageDraw.Draw(img)

    # Subtle rounded card background with border
    corner_radius = int(s * 0.22)
    # Background fill
    draw.rounded_rectangle([0, 0, s - 1, s - 1], radius=corner_radius, fill=(255, 255, 255, 255), outline=(229, 229, 234, 255), width=int(3 * supersample))

    center = s / 2
    
    # 1. Outer circle
    r_outer = s * 0.38
    draw.ellipse([center - r_outer, center - r_outer, center + r_outer, center + r_outer],
                 outline=(29, 29, 31, 60), width=int(2.5 * supersample))

    # 2. Reticle lines
    line_w = int(2.5 * supersample)
    reticle_inner = s * 0.38
    reticle_outer = s * 0.44
    # Top
    draw.line([(center, center - reticle_outer), (center, center - reticle_inner)], fill=(29, 29, 31, 80), width=line_w)
    # Bottom
    draw.line([(center, center + reticle_inner), (center, center + reticle_outer)], fill=(29, 29, 31, 80), width=line_w)
    # Left
    draw.line([(center - reticle_outer, center), (center - reticle_inner, center)], fill=(29, 29, 31, 80), width=line_w)
    # Right
    draw.line([(center + reticle_inner, center), (center + reticle_outer, center)], fill=(29, 29, 31, 80), width=line_w)

    # 3. Middle dark arc (180 deg)
    r_mid = s * 0.28
    draw.arc([center - r_mid, center - r_mid, center + r_mid, center + r_mid],
             start=90, end=270, fill=(29, 29, 31, 255), width=int(3.5 * supersample))

    # 4. Inner Emerald arc (180 deg)
    r_in = s * 0.17
    draw.arc([center - r_in, center - r_in, center + r_in, center + r_in],
             start=270, end=450, fill=(5, 150, 105, 255), width=int(4 * supersample))

    # 5. Center emerald core dot
    r_dot = s * 0.075
    draw.ellipse([center - r_dot, center - r_dot, center + r_dot, center + r_dot],
                 fill=(5, 150, 105, 255))

    # Downsample to target size with high-quality Lanczos resampling
    final_img = img.resize((size, size), Image.Resampling.LANCZOS)
    return final_img

# Generate 192x192, 512x512, and 180x180 (Apple touch icon)
img_192 = create_sanjeevani_icon(192)
img_192.save(os.path.join(static_dir, "icon-192.png"), "PNG")

img_512 = create_sanjeevani_icon(512)
img_512.save(os.path.join(static_dir, "icon-512.png"), "PNG")

img_apple = create_sanjeevani_icon(180)
img_apple.save(os.path.join(static_dir, "apple-touch-icon.png"), "PNG")

# Also write crisp SVG icon
svg_content = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" fill="none">
  <rect width="512" height="512" rx="112" fill="#FFFFFF"/>
  <rect x="2" y="2" width="508" height="508" rx="110" stroke="#E5E5EA" stroke-width="4"/>
  <circle cx="256" cy="256" r="190" stroke="#1D1D1F" stroke-opacity="0.25" stroke-width="12"/>
  <path d="M256 96C167.6 96 96 167.6 96 256C96 344.4 167.6 416 256 416" stroke="#1D1D1F" stroke-width="18" stroke-linecap="round"/>
  <path d="M256 160C202.9 160 160 202.9 160 256C160 309.1 202.9 352 256 352C309.1 352 352 309.1 352 256" stroke="#059669" stroke-width="18" stroke-linecap="round"/>
  <circle cx="256" cy="256" r="42" fill="#059669"/>
  <path d="M256 64V128M256 384V448M64 256H128M384 256H448" stroke="#1D1D1F" stroke-width="14" stroke-linecap="round" stroke-opacity="0.3"/>
</svg>'''

with open(os.path.join(static_dir, "icon.svg"), "w", encoding="utf-8") as f:
    f.write(svg_content)

print(f"PWA Icons created successfully in {static_dir}")
