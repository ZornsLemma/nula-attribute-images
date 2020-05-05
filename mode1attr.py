# TODO: Possible idea to try...
#
# We start of by performing clustering on the colours, which will hopefully reveal a small number
# of clusters. We then do an initial histogram building pass on pixel pairs but replacing the
# colour numbers with their cluster numbers. For all (or maybe just "a few" - perhaps until we get down to some percentage of pixels in image counted or some number of colours allocated) of these, in descending order of frequency, we say "we are going to put one colour from each of these cluster pairs into the same palette group". Suppose the highest frequency cluster pair is (0, 2). We need to pick a colour from cluster pair 0 and one from cluster pair 2 and insert them into the palette. Which colours will we pick? We will use a histogram exactly like the one we currently have on colour pairs, and work down it and take the first entry which has colours from those clusters. We then move on to the next highest frequency cluster pair and so on. When we stop (which may be on some threshold, as noted earlier, or just - though unlikely? - because we've done them all and there's still space in the palette), we follow the existing algorithm on the colour histogram to fill in any space in the palette. We still have the constraint that each colour can only be mentioned once and therefore every colour is mentioned at least once.
#
# My thinking here is that at the cost of perhaps distributing colours from the same cluster around a bit more and thus reducing fine detail on areas of similar colour, we are less likely to have to make a "bad" colour choice in the final image because the group we're using has no good approximation for one of our colours.
#
# TODO: And when we calculate distance between colours for clustering, we should probably just use the high four bits like we do when outputting the final image, so as to get "correct" distances - probably doesn't make a huge amount of difference, but it might make some. (And so we don't behave inconsistently when picking the "closest" colour in the final stage, colour_error() should do the same thing too - if indeed colour_error() isn't what we use during clustering anyway.)

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import math
import subprocess
import sys
from collections import defaultdict

# TODO: Use of assert for error checking is naughty



def colour_error(a, b):
    p = image.getpalette()
    return (math.pow(p[a*3+0] - p[b*3+0], 2) + 
            math.pow(p[a*3+1] - p[b*3+1], 2) +
            math.pow(p[a*3+2] - p[b*3+2], 2))

def best_effort_palette_group_lookup(desired_colour, palette_group):
    best_colour = None
    for colour in palette_group:
        error = colour_error(desired_colour, colour)
        if best_colour is None or error < best_error:
            best_colour = colour
            best_error = error
    return best_colour, best_error

def palette_group_average_error(colour, palette_group):
    if len(palette_group) == 0:
        # This is very unlikely, but possible
        return 0
    return (sum(colour_error(colour, palette_colour) for palette_colour in palette_group) /
            len(palette_group))

def best_effort_pixel_representation(pixels, palette):
    best_palette_group = None
    for i, palette_group in enumerate(palette):
        adjusted_pixels = []
        total_error = 0
        for pixel in pixels:
            adjusted_pixel, error = best_effort_palette_group_lookup(pixel, palette_group)
            adjusted_pixels.append(adjusted_pixel)
            total_error += error
        if best_palette_group is None or total_error < best_total_error:
            best_palette_group = i
            best_total_error = total_error
            best_adjusted_pixels = adjusted_pixels
    return best_palette_group, best_adjusted_pixels

def distance(a, b):
    # TODO: Do we need to bother taking square root here?
    return math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2) + math.pow(a[2] - b[2], 2)

def visualise_palette(palette, filename):
    cell_size = 64
    output = PIL.Image.new("RGB", (4*cell_size, 4*cell_size))
    image_palette = image.getpalette()
    d = PIL.ImageDraw.ImageDraw(output)
    font = PIL.ImageFont.truetype("Arial.ttf", 18)
    colour_black = (0, 0, 0)
    colour_white = (255, 255, 255)
    for y, palette_group in enumerate(palette):
        for x, colour in enumerate(palette_group):
            colour_rgb = (image_palette[colour*3+0], image_palette[colour*3+1], image_palette[colour*3+2])
            d.rectangle((x*cell_size, y*cell_size, (x+1)*cell_size, (y+1)*cell_size), fill=colour_rgb, outline=colour_rgb)
            if distance(colour_rgb, colour_white) < distance(colour_rgb, colour_black):
                font_colour = colour_black
            else:
                font_colour = colour_white
            font_size = font.getsize(str(colour))
            d.text((x*cell_size + (cell_size-font_size[0])/2, y*cell_size + (cell_size-font_size[1])/2), str(colour), font=font)
    output.show() # TODO: we don't currently use the provided filename...



if len(sys.argv) != 3:
    sys.stderr.write('Usage: %s INFILE OUTFILE\n' % sys.argv[0])
    sys.exit(1)

image = PIL.Image.open(sys.argv[1])
xsize, ysize = image.size
assert xsize == 240
assert ysize == 256
# TODO: verify it's an indexed colour image with 16 or fewer colours

# We need to build up a ULA palette; this splits the 16 colours into four
# groups of four colours each, and any pixel triple on the BBC screen will only
# be able to use the colours from one of the four groups.
#
# The current strategy is:
#
# - We insist that every colour appears exactly once in the ULA palette. This
#   means that every colour can appear on the BBC screen. A consequence of this
#   is that a triple with all three pixels the same colour can always be
#   displayed correctly and we therefore can ignore such triples.
#
# - We consider all pixel triples ABC in the image and treat them as three
#   pixel pairs AB/AC/BC. Each pixel pair where the two colours are different
#   feeds into a histogram of colour pair frequency; we use this to try to ensure
#   colours which are used together the most in the original image end up in the
#   same ULA palette group and can therefore be used together in the output
#   image.
#
# - We then start with an empty ULA palette and work through the histogram,
#   starting with the most frequent colour pair. For each colour pair:
#
#   - If both of the colours are already in the palette, we don't do anything
#     else, because each colour can only appear once and earlier colour pairs
#     forced us to put these colours in already.
#
#   - If only one of the colours is already in the palette:
#
#     - If there's space in the colour group containing the colour already in
#       the palette, add the other one there as well.
#
#     - Otherwise ignore this colour pair - we can't put them together because
#       of decisions already made, and the fact that every colour appears
#       exactly once in the final palette means the colour not yet in the
#       palette will be added eventually. There's no point forcing it in at an
#       arbitrary spot here since it won't help this colour pair and we want
#       to add it in the best spot for some later colour pair.
#
#   - If neither of the colours is already in the palette:
#
#     - If there's a colour group with space for at least two colours, add both
#       of these colours to it. We prefer the colour group with the most free
#       space if there's more than one, in an attempt to keep options open for
#       later colour pairs.
#
#     - Otherwise ignore this colour pair - as in other cases, we can't put them
#       in the same group and they will both be individually present in the final
#       palette.
#
# TODO: There's lot of scope for experimentation here, e.g.:
#
# - We could allow the user to specify a partial ULA palette up front, or at
#   least some kind of hints, to tweak the output and compensate for lack of
#   intelligence in this code.
#
# - We could not insist on including all 16 colours in the palette, and instead
#   allow some colours to appear in more than one group. This would obviously
#   reduce the total number of on-screen colours but might be worth it sometimes
#   to reduce blocking.
#
# - Following on from the previous idea, if we did our own
#   quantisation/dithering we could re-do that using the reduced number of
#   colours if we decide not to use all 16. (I'm not sure if this is a good idea,
#   but so far everything I've tried which can do dithering programatically gives
#   much worse results than manually dithering with gimp, so it's really not
#   viable.)
#
# - We could perhaps allow pixels to "swap" between adjacent triples if this
#   would allow them to appear in the correct colour but a slightly incorrect
#   position.
#
# - We could attempt to do some kind of "colour space clustering" on the original
#   palette and use that information to guide placing the colours in the ULA palette
#   groups. For example, when neither of the colours in a colour pair is already
#   present in the palette and there are multiple palette groups which they could be
#   added to, we could prefer a palette group which already has other colours from
#   the same cluster. Or we could try to disperse large-ish clusters of colours across
#   several palette groups so that we can still get a good approximation to those
#   colours (even if not perfect ones) when they appear together and can also get a
#   good approximation to those colours when they appear together with distinct
#   colours.

# Examine the pixel triples in the image to build the histogram of colour pairs.
data = list(image.getdata())
hist = defaultdict(int)
for i in range(0, len(data), 3):
    pixel_triple = data[i:i+3]
    def do_pair(i, j):
        # We use a set because the order of the two colours is irrelevant.
        if pixel_triple[i] != pixel_triple[j]:
            hist[frozenset([pixel_triple[i], pixel_triple[j]])] += 1
    do_pair(0, 1)
    do_pair(0, 2)
    do_pair(1, 2)
hist = sorted(hist.items(), key=lambda x: x[1], reverse=True)

for hist_entry in hist:
    print "%s\t%s" % (hist_entry[0], hist_entry[1])
#assert False

palette = [set() for i in range(0, 4)]

# Work through the colour pairs in order from most common to least common.
for hist_entry in hist:
    colour_set = hist_entry[0]
    assert len(colour_set) == 2
    palette_union = set.union(*palette)
    if len(palette_union) >= 15:
        # Just a minor optimisation; if we've already got 15 colours in the
        # palette there's no choice to be made any more, because we insist
        # all 16 colours are present.
        break
    if colour_set.issubset(palette_union):
        # Both of these colours are already in the palette, so we can't add
        # them again (whether or not this allows this pair to be represented
        # or not).
        pass
    else:
        intersection = colour_set.intersection(palette_union)
        if len(intersection) == 1:
            # One of these colours is already in the palette. If there's space
            # in its palette group for the other, add it. If not, we can't
            # represent this pair properly so do nothing.
            existing_colour = tuple(intersection)[0]
            new_colour = tuple(colour_set - intersection)[0]
            for palette_group in palette:
                if existing_colour in palette_group:
                    if len(palette_group) < 4:
                        palette_group.add(new_colour)
                    break
        else:
            # Neither of these colours is already in the palette. Pick one of
            # the palette groups with most free space and add the pair there.
            # If no group has space for a pair, just ignore this pair.
            emptiest_palette_group = None
            for palette_group in palette:
                if len(palette_group) <= 2 and (
                        emptiest_palette_group is None or 
                        len(palette_group) < emptiest_palette_group_len):
                    emptiest_palette_group = palette_group
                    emptiest_palette_group_len = len(palette_group)
            if emptiest_palette_group is not None:
                emptiest_palette_group.update(colour_set)

# Dump the palette out at this stage; it's interesting (though unlikely?) if there are
# any gaps in it.
#print "Partial palette:", palette

# If some colours haven't yet been added to the palette, add them. There probably won't
# be much wiggle room left, but we try to put these isolated colours with similar ones.
for i in range(0, 16):
    if i not in palette_union:
        best_palette_group = None
        for palette_group in palette:
            if len(palette_group) < 4:
                error = palette_group_average_error(i, palette_group)
                if best_palette_group is None or error < best_error:
                    best_palette_group = palette_group
                    best_error = error
        assert best_palette_group is not None
        best_palette_group.add(i)
print "Final palette:", palette
visualise_palette(palette, "zpal.png")

# We need to renumber the palette because the 0th palette group has to contain colours
# 0-3, the 1st 4-7 and so on.
bbc_colour_map = [None]*16
bbc_colour = 0
for palette_group in palette:
    for original_colour in palette_group:
        bbc_colour_map[original_colour] = bbc_colour
        bbc_colour += 1

bbc_image = open(sys.argv[2], "wb")

# Write the palette out at the start of the image; slideshow.bas will use this to
# program the palette.
for original_colour in range(0, 16):
    p = image.getpalette()
    r = p[original_colour*3+0] >> 4
    g = p[original_colour*3+1] >> 4
    b = p[original_colour*3+2] >> 4
    bbc_colour = bbc_colour_map[original_colour]
    #print bbc_colour, r, g, b
    bbc_image.write(bytearray([(bbc_colour<<4) | r, (g<<4) | b]))

# Write the image data out with appropriate bit-swizzling. We also make the
# same attribute-constrained modifications to our in-memory image so we can
# dump it out for viewing on the host to get an idea of how well we did without
# needing to fire up an emulator or real machine. (The resulting image is not
# identical to that on the emulator or real machine, because we don't restrict
# the palette to 12-bit colour. We could, but it seems better for flipping back
# and forth between the input and output to compare them to avoid this
# additional difference.)
pixel_map = image.load()
for y_block in range(0, ysize, 8):
    print "Y:", y_block
    for x in range(0, xsize, 3):
        for y in range(y_block, y_block+8):
            pixels = (pixel_map[x,y], pixel_map[x+1,y], pixel_map[x+2,y])
            palette_index, adjusted_pixels = best_effort_pixel_representation(pixels, palette)
            pixel_map[x,y] = adjusted_pixels[0]
            pixel_map[x+1,y] = adjusted_pixels[1]
            pixel_map[x+2,y] = adjusted_pixels[2]
            assert bbc_colour_map[adjusted_pixels[0]]/4 == bbc_colour_map[adjusted_pixels[1]]/4
            assert bbc_colour_map[adjusted_pixels[1]]/4 == bbc_colour_map[adjusted_pixels[2]]/4
            attribute_value = bbc_colour_map[adjusted_pixels[0]] / 4
            pixel2 = bbc_colour_map[adjusted_pixels[0]] % 4
            pixel1 = bbc_colour_map[adjusted_pixels[1]] % 4
            pixel0 = bbc_colour_map[adjusted_pixels[2]] % 4
            def adjust_bbc_pixel(n):
                assert 0 <= n <= 3
                return ((n & 2) << 3) | (n & 1)
            bbc_byte = ((adjust_bbc_pixel(pixel2) << 3) |
                        (adjust_bbc_pixel(pixel1) << 2) |
                        (adjust_bbc_pixel(pixel0) << 1) |
                        adjust_bbc_pixel(attribute_value))
            bbc_image.write(chr(bbc_byte))

# Save the attribute-constrained version of the image.
image.save("z.png")
