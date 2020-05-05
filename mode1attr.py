import PIL.Image
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

def best_effort_palette_entry_lookup(pixel, palette_entry):
    best_colour = None
    for colour in palette_entry:
        error = colour_error(pixel, colour)
        if best_colour is None or error < best_error:
            best_colour = colour
            best_error = error
    return best_colour, best_error

def palette_entry_average_error(colour, palette_entry):
    if len(palette_entry) == 0:
        # This is very unlikely, but possible
        return 0
    return sum(colour_error(colour, palette_colour) for palette_colour in palette_entry)/len(palette_entry)

def best_effort_pixel_representation(pixels, palette):
    best_palette_entry = None
    for i, palette_entry in enumerate(palette):
        adjusted_pixels = []
        total_error = 0
        for pixel in pixels:
            adjusted_pixel, error = best_effort_palette_entry_lookup(pixel, palette_entry)
            adjusted_pixels.append(adjusted_pixel)
            total_error += error
        if best_palette_entry is None or total_error < best_total_error:
            best_palette_entry = i
            best_total_error = total_error
            best_adjusted_pixels = adjusted_pixels
    return best_palette_entry, best_adjusted_pixels



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
    # We consider all three "sub-pairs" of pixels; this is an attempt to identify
    # colour pairs which often occur together. If a pair of pixels are the same
    # colour, this doesn't count; in the extreme case where all three pixels are the
    # same we don't have to worry (because we will arrange for every colour to appear
    # *somewhere* in the palette) and in the case of two pixels of one colour and
    # one pixel of another we will attach 2/3 of the maximum weight to it for the
    # two non-similar pairs.
    def do_pair(i, j):
        if pixel_triple[i] != pixel_triple[j]:
            hist[tuple(set([pixel_triple[i], pixel_triple[j]]))] += 1
    do_pair(0, 1)
    do_pair(0, 2)
    do_pair(1, 2)
hist2 = sorted(hist.items(), key=lambda x: x[1], reverse=True)

#for hist_entry in hist2:
#    print "%s\t%s" % (hist_entry[0], hist_entry[1])
#assert False

palette = [set() for i in range(0, 4)]

# Work through the colour pairs in order from most common to least common.
for i, hist_entry in enumerate(hist2):
    colour_set = set(hist_entry[0])
    assert len(colour_set) == 2
    palette_union = set.union(*palette)
    if len(palette_union) >= 15:
        # Just a minor optimisation; if we've already got 15 colours in the
        # palette there's no choice to be made any more.
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
            # in its palette entry for the other, add it. If not, we can't
            # represent this pair properly so do nothing.
            existing_colour = tuple(intersection)[0]
            new_colour = tuple(colour_set - intersection)[0]
            for palette_entry in palette:
                if existing_colour in palette_entry:
                    if len(palette_entry) < 4:
                        palette_entry.add(new_colour)
                    break
        else:
            # Neither of these colours is already in the palette. Pick one of
            # the palette entries with most free space and add the pair there.
            # If no entry has space for a pair, just ignore this pair.
            emptiest_palette_entry = None
            for palette_entry in palette:
                if len(palette_entry) <= 2 and (emptiest_palette_entry is None or len(palette_entry) < emptiest_palette_entry_len):
                    emptiest_palette_entry = palette_entry
                    emptiest_palette_entry_len = len(palette_entry)
            if emptiest_palette_entry is not None:
                emptiest_palette_entry.update(colour_set)
    print colour_set, palette

print "A", palette

# If some colours haven't yet been added to the palette, add them. There probably won't
# be much wiggle room left, but we try to put these isolated colours with similar ones.
for i in range(0, 16):
    if i not in palette_union:
        best_palette_entry = None
        for palette_entry in palette:
            if len(palette_entry) < 4:
                error = palette_entry_average_error(i, palette_entry)
                if best_palette_entry is None or error < best_error:
                    best_palette_entry = palette_entry
                    best_error = error
        assert best_palette_entry is not None
        best_palette_entry.add(i)

print "Q", palette


bbc_colour_map = [None]*16
bbc_colour = 0
for palette_entry in palette:
    palette_entry_tuple = tuple(palette_entry)
    for original_colour in palette_entry_tuple:
        bbc_colour_map[original_colour] = bbc_colour
        bbc_colour += 1

bbc_image = open(sys.argv[2], "wb")

# Write the palette out
for original_colour in range(0, 16):
    bbc_colour = bbc_colour_map[original_colour]
    p = image.getpalette()
    r = p[original_colour*3+0] >> 4
    g = p[original_colour*3+1] >> 4
    b = p[original_colour*3+2] >> 4
    print bbc_colour, r, g, b
    bbc_image.write(bytearray([(bbc_colour<<4) | r, (g<<4) | b]))

pixel_map = image.load()
for y_block in range(0, ysize, 8):
    print y_block
    for x in range(0, xsize, 3):
        for y in range(y_block, y_block+8):
            pixels = (pixel_map[x,y], pixel_map[x+1,y], pixel_map[x+2,y])
            palette_index, adjusted_pixels = best_effort_pixel_representation(pixels, palette)
            pixel_map[x,y] = adjusted_pixels[0]
            pixel_map[x+1,y] = adjusted_pixels[1]
            pixel_map[x+2,y] = adjusted_pixels[2]
            assert bbc_colour_map[adjusted_pixels[0]]/4 == bbc_colour_map[adjusted_pixels[1]]/4
            assert bbc_colour_map[adjusted_pixels[1]]/4 == bbc_colour_map[adjusted_pixels[2]]/4
            attribute_value = bbc_colour_map[adjusted_pixels[0]]/4
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
            if False:
                if bbc_byte == 0x1e:
                    print
                    print adjusted_pixels[0], adjusted_pixels[1], adjusted_pixels[2]
                    print attribute_value, adjust_bbc_pixel(attribute_value)
                    print pixel2, pixel1, pixel0
                    assert False
            bbc_image.write(chr(bbc_byte))

image.save("z.png")
