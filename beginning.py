from PIL import Image, ImageColor, ImageDraw

# --- parameters ---
WIDTH = 500
HEIGHT = 500
TRI_SIDE_LENGTH = 125
RADIUS_OUTER = 100
RADIUS_INNER = 50
CENTRE = WIDTH // 2

# --- initialisation ---
image = Image.new("RGB", (WIDTH, HEIGHT))
draw = ImageDraw.Draw(image)
pixels = image.load()

# --- Paint the entire image red ---
draw.rectangle([(0, 0), (WIDTH - 1, HEIGHT - 1)], fill="purple")

# --- painting corners blue ---
blue = ImageColor.getrgb("blue")
corner_count = TRI_SIDE_LENGTH

for x in range(TRI_SIDE_LENGTH):
    for y in range(corner_count):
        pixels[x, y] = blue # top left
        pixels[WIDTH - x - 1, y] = blue # top right
        pixels[x, WIDTH - y - 1] = blue # bottom left
        pixels[WIDTH - x - 1, WIDTH - y - 1] = blue # bottom right

    corner_count -= 1

# --- painting inner and outer circles ---
def paint_circle(radius, colour):
    left = CENTRE - radius
    top = CENTRE - radius
    right = CENTRE + radius
    bottom = CENTRE + radius

    # Paint the circle
    draw.ellipse([(left, top), (right, bottom)], fill=colour)

paint_circle(RADIUS_OUTER, "red")
paint_circle(RADIUS_INNER, "orange")

image.save("beginning.png", "PNG")
