import PIL.Image
import math
import subprocess
import sys
from collections import defaultdict



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



if len(sys.argv) != 2:
    sys.stderr.write('Usage: %s FILE ...\n' % sys.argv[0])
    sys.exit(1)

original_image = PIL.Image.open(sys.argv[1])
xsize, ysize = original_image.size
assert xsize == 240
assert ysize == 256

our_colours = 16
while True:
    #image = original_image.convert(mode="P", dither=PIL.Image.FLOYDSTEINBERG, palette=PIL.Image.ADAPTIVE, colors=our_colours)
    #
    #image_palette = hitherdither.palette.Palette.create_by_median_cut(original_image, n=our_colours)
    #image = hitherdither.diffusion.error_diffusion_dithering(original_image, image_palette)
    #
    #image.save('zo-%d.png' % (our_colours,))
    #result = subprocess.call(["convert", sys.argv[1], "-colors", str(our_colours), "zo-%d.png" % (our_colours,)])
    #assert result == 0
    #image = PIL.Image.open("zo-%d.png" % (our_colours,))
    image = original_image

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

    break
    print our_colours, palette
    our_colours_used = len(set.union(*palette))
    if our_colours_used == our_colours:
        break
    our_colours = our_colours_used


pixel_map = image.load()
for y in range(0, ysize):
    print y
    for x in range(0, xsize, 3):
        pixels = (pixel_map[x,y], pixel_map[x+1,y], pixel_map[x+2,y])
        palette_index, adjusted_pixels = best_effort_pixel_representation(pixels, palette)
        pixel_map[x,y] = adjusted_pixels[0]
        pixel_map[x+1,y] = adjusted_pixels[1]
        pixel_map[x+2,y] = adjusted_pixels[2]

image.save("z.png")
