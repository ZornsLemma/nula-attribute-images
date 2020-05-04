import PIL.Image
import sys
from collections import defaultdict



def find_colour_set_in_palette(colour_set, palette):
    for i, palette_entry in enumerate(palette):
        if all(colour in palette_entry for colour in colour_set):
            return i
    return None

def expand_palette_for_colour_set(colour_set, palette):
    best_index = None
    best_delta = None
    # We can't have more than four colours in a palette entry. We prefer to add
    # as few colours as possible to a palette entry to increase flexibility for
    # later additions.
    for i, palette_entry in enumerate(palette):
        delta = len(colour_set - palette_entry)
        if len(palette_entry) + delta <= 4:
            if best_index is None or delta < best_delta:
                best_index = i
                best_delta = delta
    if best_index is not None:
        palette[best_index] = palette[best_index].union(colour_set)
    #print best_index, delta
    return best_index



if len(sys.argv) != 2:
    sys.stderr.write('Usage: %s FILE ...\n' % sys.argv[0])
    sys.exit(1)

original_image = PIL.Image.open(sys.argv[1])
xsize, ysize = original_image.size
assert xsize == 240
assert ysize == 256

our_colours = 16
while True:
    image = original_image.convert(mode="P", dither=PIL.Image.FLOYDSTEINBERG, palette=PIL.Image.ADAPTIVE, colors=our_colours)

    data = list(image.getdata())
    hist = defaultdict(int)
    for i in range(0, len(data), 3):
        hist[tuple(set(data[i:i+3]))] += 1
    hist2 = sorted(hist.items(), key=lambda x: x[1], reverse=True)

    #for hist_entry in hist2:
    #    print "%s\t%s" % (hist_entry[0], hist_entry[1])

    palette = [set()]*4

    # Work through the colour sets in order from most common to least common.
    palette_index_for_colour_set = defaultdict(int)
    for i, hist_entry in enumerate(hist2):
        colour_set = set(hist_entry[0])
        #print "%s\t%s" % (colour_set, hist_entry[1])
        if tuple(colour_set) in palette_index_for_colour_set:
            # We can perfectly display this set of colours, and we already have it in
            # our dictionary.
            pass
        else:
            palette_index = find_colour_set_in_palette(colour_set, palette)
            if palette_index is not None:
                # We can perfectly display this set of colours, but it's not in our
                # dictionary yet.
                palette_index_for_colour_set[tuple(colour_set)] = palette_index
            else:
                # We can't display this set of colours using the palette; can we expand
                # the palette to include it?
                palette_index = expand_palette_for_colour_set(colour_set, palette)
                if palette_index is None:
                    # We can't expand the palette to include it perfectly.
                    #print colour_set
                    #print palette
                    #print palette_index
                    #assert False
                    pass

    print our_colours, palette
    our_colours_used = len(set.union(*palette))
    if our_colours_used == our_colours:
        break
    our_colours = our_colours_used
