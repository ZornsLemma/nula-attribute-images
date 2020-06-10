# TODO: Perhaps try measuring distance in HSV space not RGB space? And perhaps do a weighted average of those three components with a big weight attached to hue?

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
import colorsys
import copy
import math
import os.path
import pickle
import subprocess
import sys
from collections import defaultdict

# TODO: Use of assert for error checking is naughty


def image_palette_rgb(colour):
    p = image.getpalette()
    # We discard the low nybble of the colour here because that's what we'll do in the
    # final VideoNuLA image, and we want to make our judgements of colour distance using
    # the actual colours, not the higher bit depth ones in the original palette.
    return (p[colour*3+0] >> 4, p[colour*3+1] >> 4, p[colour*3+2] >> 4)

def distance(a, b):
    # TODO: Do we need to bother taking square root here? We *might* for clustering purposes
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2) + math.pow(a[2] - b[2], 2))
    a_hsv = hsv_from_rgb(a)
    b_hsv = hsv_from_rgb(b)
    h_dist = abs(a_hsv[0] - b_hsv[0])
    if h_dist > 0.5:
        h_dist = 1 - h_dist
    s_dist = abs(a_hsv[1] - b_hsv[1])
    v_dist = abs(a_hsv[2] - b_hsv[2])
    # TODO: Magic numbers!
    # At this point, 0 <= h_dist <= 0.5 and 0 <= [sv]_dist <= 1. We want to emphasise h_dist
    # so we scale it up and calculate a Euclidean distance. This probably makes some kind of
    # sense...
    #h_dist *= 2
    # TODO: Do we need to bother taking square root here? We *might* for clustering purposes
    return math.sqrt(h_dist*h_dist + s_dist*s_dist + v_dist*v_dist)

def hsv_from_rgb(rgb):
    assert all(0 <= rgb[i] <= 15 for i in range(0, 2))
    return colorsys.rgb_to_hsv(*[c/15.0 for c in rgb])

def colour_error(a, b):
    a_rgb = image_palette_rgb(a)
    b_rgb = image_palette_rgb(b)
    return distance(a_rgb, b_rgb)

# TODO: We should probably get this to return the rank of the hist entry we picked, then we could use that to "be smarter" when picking between different palette groups as noted in some other TODO comments.
# TODO: It kind of feels like we could be using the histogram "before" we call this function in order to decide what's best, but maybe I'm confused and let's just try this way first.
# TODO: Do we need to be given palette_group? This is the palette_group we might add the returned colour to, but maybe that's not useful information, especially if I go with one of the above ideas.
# TODO: With the current tweak to cut off x% through the colour class histogram so as to leave some of the palette free for the colour histogram pass, I find myself wondering if this should (and this is a bit fuzzily self-contradictory, but it's late and I want to go to bed ;-) ) avoid the "top" part of the colour histogram so as to leave those colours available for exact pairing later on and instead prefer "mid-popularity" colours from the colour histogram. Maybe as a completely different approach the colour class phase should avoid packing out any palette groups completely (maybe even only half filling them), doing the best it can to support the most popular colour class pairs in that way. Not at all sure.
def pick_colour_from_colour_class(palette, palette_group, colour_class):
    palette_union = set.union(*palette)
    possible_colours = colour_class_to_colour_map[colour_class] - palette_union
    if len(possible_colours) == 0:
        return None
    for colour_set, _ in hist:
        # If both colours are already in the palette this histogram entry isn't helpful; we can't
        # add those colours.
        if not colour_set.issubset(palette_union):
            difference = colour_set - palette_union
            possible_colours = colour_class_to_colour_map[colour_class].intersection(difference)
            if len(possible_colours) > 0:
                return min(possible_colours) # pick one arbitrarily-but-consistently if there are two
    # TODO: At this point it might be *possible* to pick a colour, but I'm not sure it's helpful
    # to do so.
    assert False # TODO: Let's see if this can happen
    return None

def best_effort_palette_group_lookup(local_map, desired_colour, palette_group, aux_palette):
    if desired_colour in local_map:
        return local_map[desired_colour]
    already_used_colours = set(c[0] for c in local_map.values())
    best_colour = None
    for colour in palette_group:
        error = colour_error(aux_palette[desired_colour], aux_palette[colour])
        if colour in already_used_colours: # try to avoid removing dither and ending up three pixels same when we can't get a perfect match for all colours
            error *= 1.5 # TODO: magic
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

def nearest_colour(aux_palette, rgb, x, y, errors):
    best_aux_index = None
    for aux_index, aux_colour in enumerate(aux_palette):
        aux_colour_rgb = image_palette_rgb(aux_colour)
        error = distance(rgb, aux_colour_rgb)
        if best_aux_index is None or error < best_error:
            best_aux_index = aux_index
            best_error = error
    best_rgb = image_palette_rgb(aux_palette[best_aux_index])
    error_rgb = [a-b for a, b in zip(best_rgb, rgb)]
    error_dist = [
        ( 1, 0, 7.0/16),
        ( 1, 1, 1.0/16),
        ( 0, 1, 5.0/16),
        (-1, 1, 3.0/16)]
    for xoff, yoff, fraction in error_dist:
        if x+xoff < xsize and y+yoff < ysize:
            errors[y+yoff][x+xoff] = [a+(b*fraction) for a, b in zip(errors[y+yoff][x+xoff], error_rgb)]
    return best_aux_index

# TODO: I wonder if when we can't get the exact colour, we should avoid the closest match *if* it would result in making two pixels in this triplet identical when they previously weren't; in that case take the second closest match (perhaps not if it's "very different"). My thinking here is that while we might change the colour of things, we'd hopefully do so with some consistency and this would avoid losing detail in dithering and replacing it with flat colours giving that ugly-ish horizontal mini-stripe attribute appearance.
def best_effort_pixel_representation(pixels, palette, aux_palette, x, y, errors):
    pixels = list(pixels)
    if False:
        for i, pixel in enumerate(pixels):
            pixel_rgb = image_palette_rgb(aux_palette[pixel])
            error_rgb = (0, 0, 0) if x+i >= xsize else errors[y][x]
            pixel_rgb = [a-b for a, b in zip(pixel_rgb, error_rgb)]
            new_pixels.append(nearest_colour(aux_palette, pixel_rgb, x, y, errors))
        pixels = new_pixels

    # This error distribution is a bit sketchy because we might not actually *get* the
    # first pixel colour we choose (and fix, and distribute error from) depending on
    # subsequent attribute choices. But it does seem to fractionally improve the quality, so
    # let's leave it in for now.
    best_adjusted_pixels = pixels
    for pixel_index in range(len(pixels)):
        pixel_rgb = image_palette_rgb(aux_palette[best_adjusted_pixels[pixel_index]])
        error_rgb = (0, 0, 0) if x+pixel_index >= xsize else errors[y][x]
        pixel_rgb = [a-b for a, b in zip(pixel_rgb, error_rgb)]
        pixels[pixel_index] = nearest_colour(aux_palette, pixel_rgb, x+pixel_index, y, errors)

        best_palette_group = None
        #print "AAA", pixels
        for i, palette_group in enumerate(palette):
            local_map = {}
            adjusted_pixels = []
            total_error = 0
            for pixel in pixels:
                if pixel in palette_group:
                    local_map[pixel] = (pixel, 0.0)
            for pixel in pixels:
                adjusted_pixel, error = best_effort_palette_group_lookup(local_map, pixel, palette_group, aux_palette)
                local_map[pixel] = (adjusted_pixel, error)
                adjusted_pixels.append(adjusted_pixel)
                total_error += error
            #print "QQQ", local_map, total_error
            if best_palette_group is None or total_error < best_total_error:
                best_palette_group = i
                best_total_error = total_error
                best_adjusted_pixels = adjusted_pixels
    return best_palette_group, best_adjusted_pixels

def canonicalise_palette(palette):
    return sorted(palette, key=lambda palette_group: min(palette_group))

def palette_from_hist(hist):
    palette = [set() for i in range(0, 4)]
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
    palette_union = set.union(*palette)
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
    #print "Final palette:", palette
    #visualise_palette(palette, "zpal.png")
    
    return canonicalise_palette(palette)

def visualise_palette(palette, filename):
    cell_size = 64
    output = PIL.Image.new("RGB", (4*cell_size, 4*cell_size))
    image_palette = image.getpalette()
    d = PIL.ImageDraw.ImageDraw(output)
    font = PIL.ImageFont.truetype("Arial.ttf", 18)
    colour_black = colour_black_4bit = (0, 0, 0)
    colour_white = (255, 255, 255)
    colour_white_4bit = (15, 15, 15)
    for y, palette_group in enumerate(palette):
        for x, colour in enumerate(palette_group):
            #colour_rgb = (image_palette[colour*3+0], image_palette[colour*3+1], image_palette[colour*3+2])
            colour_rgb_4bit = image_palette_rgb(colour)
            colour_rgb = tuple(x<<4 for x in colour_rgb_4bit)
            d.rectangle((x*cell_size, y*cell_size, (x+1)*cell_size, (y+1)*cell_size), fill=colour_rgb, outline=colour_rgb)
            if distance(colour_rgb_4bit, colour_white_4bit) < distance(colour_rgb_4bit, colour_black_4bit):
                font_colour = colour_black
            else:
                font_colour = colour_white
            font_size = font.getsize(str(colour))
            d.text((x*cell_size + (cell_size-font_size[0])/2, y*cell_size + (cell_size-font_size[1])/2), str(colour), font=font)
    output.show() # TODO: temporary?
    output.save(filename)

def diff_palettes(old, new):
    diffs = []
    for old_palette_group, new_palette_group in zip(old, new):
        diffs.append((old_palette_group - new_palette_group, new_palette_group - old_palette_group))
    return diffs

def palette_error(pixel_hist, aux_palette):
    error_by_pixel_colour = []
    best_aux_palette_colour = {}
    for pixel_colour, freq in pixel_hist:
        #print "Q", pixel_colour, freq
        best_error = None
        #if pixel_colour == 11:
        #    print
        for palette_colour in aux_palette:
            error = colour_error(palette_colour, pixel_colour)
            #assert error > 0.01 or palette_colour == pixel_colour
            #print "Q2", pixel_colour, palette_colour, error
            #if pixel_colour == 11:
            #    print "Q3", pixel_colour, palette_colour, error
            if best_error is None or error < best_error:
                #print "PX", error, best_error
                best_error = error
                best_error_palette_colour = palette_colour
            #print "QQ", pixel_colour, best_error, freq, best_error_palette_colour
        # TODO: Maybe we should do e.g. best_error * freq^2 instead of just best_error * freq???
        #if pixel_colour in (30, 31):
        #    #print "XXX"
        #    best_error = 10000000 # TODO HACK!
        error_by_pixel_colour.append((pixel_colour, best_error * freq, best_error_palette_colour))
        best_aux_palette_colour[pixel_colour] = best_error_palette_colour
    return error_by_pixel_colour, best_aux_palette_colour

# TODO: Ultimately this might take a y range and get the pixels directly from image, but for
# now let's just assume we have a simple list of input pixels and ignore efficiency
def aux_palette(current_aux_palette, pixels, max_aux_changes):
    current_aux_palette = current_aux_palette[:] # TODO: jic not sure if necessary
    assert len(current_aux_palette) == 16

    pixel_hist = defaultdict(int)
    for pixel_colour in pixels:
        pixel_hist[pixel_colour] += 1

    # Find the max_aux_changes colours in pixels which have the worst overall error with
    # current_aux_palette.
    error_by_pixel_colour, _ = palette_error(pixel_hist.items(), current_aux_palette)
    error_by_pixel_colour = sorted(error_by_pixel_colour, key=lambda x: x[1], reverse=True)
    #print "EEE", error_by_pixel_colour
    worst_pixel_colours = set(x[0] for x in error_by_pixel_colour[0:max_aux_changes])
    print "QXX", worst_pixel_colours

    # Add those colours into current_aux_palette, giving us an oversized palette.
    oversized_aux_palette = list(set(current_aux_palette).union(worst_pixel_colours))

    # Now repeatedly strip off the colour which has the least effect on the error of this pixel data
    # until the palette is back down to size. (I am assuming we can't just strip off the n with least
    # effect in a single pass, because removing one colour might mean another colour which previously
    # had little effect is now important as the best remaining substitute for it.)
    while len(oversized_aux_palette) > 16:
        error_if_not_in_aux = []
        best_error = None
        for removed_element in range(0, len(oversized_aux_palette)):
            candidate_aux_palette = oversized_aux_palette[:]
            candidate_aux_palette.pop(removed_element)
            error, _ = palette_error(pixel_hist.items(), candidate_aux_palette)
            #print "XD0", error
            error = sum(x[1] for x in error)
            if best_error is None or error < best_error:
                #print "XD1", error, best_error
                best_error = error
                best_removed_element = removed_element
        #if oversized_aux_palette[best_removed_element] in (30, 31):
        #    print "YYY", best_error
        oversized_aux_palette.pop(best_removed_element)

    return oversized_aux_palette




if len(sys.argv) < 3 or len(sys.argv) > 4:
    sys.stderr.write('Usage: %s INFILE OUTFILE [OUTFILESIM]\n' % sys.argv[0])
    sys.exit(1)

image = PIL.Image.open(sys.argv[1])
xsize, ysize = image.size
assert xsize == 240
assert ysize == 256
# TODO: verify it's an indexed colour image
pixel_map = image.load()

max_changes = 8
max_aux_changes = 2 # TODO WAS 3, EXPERIMENTAL


print "Source palette:"
for i in range(0, 32):
    print i, image_palette_rgb(i)


# TODO: Can we get number of colours in input palette from PIL instead of assuming 32?
# If not, can we use pypng either entirely (we don't do much sophisticated with PIL) or
# just use it to query the palette size?

aux_palette_by_y = [None]*ysize
dummy_aux_palette = range(16, 32) # TODO: "could" use 0-16, it's arbitrary, this is a sort of test to make sure the initial value isn't actually important - probably fine now so switch to 0-16 later for neatness
aux_palette_window = 2 # TODO: highly arbitrary
if False:
    max_actual_aux_change = 0
    for y in range(0, ysize):
        print "Y", y
        pixels = []
        for y2 in range(y, min(y + aux_palette_window, ysize)):
            for x in range(0, xsize):
                pixels.append(pixel_map[x, y2])
        aux_palette_by_y[y] = aux_palette(dummy_aux_palette if y == 0 else aux_palette_by_y[y-1], pixels, 16 if y == 0 else max_aux_changes)
        if y == 0:
            aux_change = 0
        else:
            aux_change = len(set(aux_palette_by_y[y]) - set(aux_palette_by_y[y-1]))
        max_actual_aux_change = max(max_actual_aux_change, aux_change)
        print y, aux_palette_by_y[y], aux_change, max_actual_aux_change
    with open("auxpal.dat", "wb") as f:
        pickle.dump(aux_palette_by_y, f)
else:
    print "LOADING AUX PALETTE FROM FILE"
    with open("auxpal.dat", "rb") as f:
        aux_palette_by_y = pickle.load(f)

lines_with_colour = defaultdict(int)
for y in range(0, ysize):
    for colour in set(aux_palette_by_y[y]):
        lines_with_colour[colour] += 1
# TODO: If it turns out we are only using e.g. 24 of the 32 colours in 95% of lines, it might
# be better to go and redo the input so gimp is dithering it down to 24 colours, so as to make
# best use of what we have.
for colour in range(0, 32):
    print "A", colour, lines_with_colour[colour]

# The entries in each element aux_palette_by_y are in arbitrary order at the moment. We need to rearrange them so if colour c is present in aux_palette_by_y[n] and aux_palette_by_y[n+1] it has the same index each time. (TODO: We could possibly have generated them to have that property already, but it's not a huge deal to swizzle here for now - this is all highly experimental inefficiency code.)
aux_palette_changes_by_y = [[]] # empty entry for y=0
stat_min_aux_changes = 1000
stat_max_aux_changes = 0
for y in range(1, ysize):
    aux_palette = set(aux_palette_by_y[y])
    aux_palette_by_y[y] = []
    for previous_entry in aux_palette_by_y[y-1]:
        if previous_entry in aux_palette:
            c = previous_entry
        else:
            c = min(aux_palette - set(aux_palette_by_y[y-1]))
        assert c in aux_palette
        aux_palette.remove(c)
        aux_palette_by_y[y].append(c)
    aux_palette_changes = []
    for i in range(0, 16):
        if aux_palette_by_y[y-1][i] != aux_palette_by_y[y][i]:
            aux_palette_changes.append((i, aux_palette_by_y[y][i]))
    assert len(aux_palette_changes) <= max_aux_changes
    stat_min_aux_changes = min(stat_min_aux_changes, len(aux_palette_changes))
    stat_max_aux_changes = max(stat_max_aux_changes, len(aux_palette_changes))
    aux_palette_changes_by_y.append(aux_palette_changes)

aux_palette_index_by_y = []
for y in range(0, ysize):
    # Because we pre-processed pixel_map earlier to restrict to 16 colours (using the
    # "original" full palette colour numbers) we should never have more that one value
    # for a key here.
    d = {}
    for aux_colour, source_colour in enumerate(aux_palette_by_y[y]):
        assert source_colour not in d
        d[source_colour] = aux_colour
    aux_palette_index_by_y.append(d)


for y in range(0, ysize):
    print "AP", y, aux_palette_by_y[y], aux_palette_changes_by_y[y]


if stat_max_aux_changes < max_aux_changes:
    print "Warning: max_aux_changes %d but no more than %d ever used; reducing max_aux_changes to take advantage, but input and/or some parameters may benefit from tweaking" % (max_aux_changes, stat_max_aux_changes_by_line)
    max_aux_changes = stat_max_aux_changes_by_line
max_ula_changes_by_line = [max_changes - (max_aux_changes * 2)] * ysize
assert max_aux_changes >= 0
assert max_ula_changes_by_line[0] >= 0
for y in range(1, ysize):
    if len(aux_palette_changes_by_y[y]) < max_aux_changes:
        #print "BONUS", y
        max_ula_changes_by_line[y] += 2

for y in range(0, ysize):
    pixel_hist = defaultdict(int)
    for x in range(0, xsize):
        pixel_hist[pixel_map[x,y]] += 1
    _, best_aux_palette_colour = palette_error(pixel_hist.items(), aux_palette_by_y[y])

    colours = set()
    for x in range(0, xsize):
        # TODO: *Maybe* we could attempt to do some kind of error-distribution as we
        # approximate here?
        if y == 91 and (102<=x<=104):
            print "BEFORE", y, x, pixel_map[x,y]
        pixel_map[x,y] = best_aux_palette_colour[pixel_map[x,y]]
        colours.add(pixel_map[x,y])
        if y == 91 and (102<=x<=104):
            print "AFTER", y, x, pixel_map[x,y]
    print "colours-in-Y", y, len(colours), colours


if False:
    # Save the colour-constrained (but not attribute-constrained) image - temp hack only
    simulated_image = image.resize((1280, 1024), resample=PIL.Image.NEAREST)
    if len(sys.argv) == 4:
        simulated_image.save(sys.argv[3])
    else:
        simulated_image.save("z.png")


if False:
    raw_hist_by_y = [None]*ysize
    for y in range(0, ysize):
        hist = defaultdict(int)
        #for x in range(0, xsize, 3):
        #    pixel_triple = (pixel_map[x,y], pixel_map[x+1,y], pixel_map[x+2,y])
        #    hist[frozenset(pixel_triple)] += 1
        for x in range(0, xsize):
            hist[pixel_map[x, y]] += 1
        hist = sorted(hist.items(), key=lambda x: x[1], reverse=True)
        raw_hist_by_y[y] = hist
    prev_l = None
    for y in range(0, ysize):
        l = [x[0] for x in raw_hist_by_y[y]]
        if prev_l is None:
            l2 = None
        else:
            l2 = set(l) - set(prev_l)
        print y, len(raw_hist_by_y[y]), 0 if l2 is None else len(l2), l2, [x[0] for x in raw_hist_by_y[y]]
        prev_l = l



#for y in range(0, ysize):
#
#
#    while something:
#        actual_ula_palette, actual_aux_palette = magic(previous_ula_palette, previous_aux_palette)
#        assert actual_aux_palette can be reached from previous_aux_palette (<=m changes)
#        assert actual_ula_palette can be reached from previous_ula_palette (<=n changes, m+n=8)
#        score = foo(image, actual_ula_palette, actual_aux_palette)



# TODO: I suspect with a really smart algorithm we should start from the bottom row and work
# up, but while I'm starting to explore this let's not complicate things.


def elem(s):
    [e] = s
    return e



class Palette:
    def __init__(self, hist):
        self.hist = hist[:]
        self.crystallised = False

    @staticmethod
    def default_palette():
        p = Palette([])
        p.crystallised_palette = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        p.crystallised = True
        return p

    # TODO: I think a crystallised palette may want to use lists not sets inside 'palette'
    # because the actual precise index is important when deciding how much a change is going
    # to cost

    @staticmethod
    def entries_used(palette, pending_colours):
        return (sum(len(palette_group) for palette_group in palette) +
                len(pending_colours - set.union(*(set(palette_group) for palette_group in palette))))


    @staticmethod
    def valid_palette(palette, pending_colours):
        assert len(palette) == 4
        assert max(len(palette_group) for palette_group in palette) <= 4
        palette_union = set.union(*(set(palette_group) for palette_group in palette))
        assert all(isinstance(x, int) for x in palette_union)
        assert all(0 <= colour <= 15 for colour in palette_union)
        assert Palette.entries_used(palette, pending_colours) <= 16
        #assert all(len(set(palette_group)) == len(palette_group) for palette_group in palette)
        return True

    @staticmethod
    def diff(old_palette, new_palette, pending_colours):
        #print "D", old_palette, new_palette, pending_colours
        assert Palette.valid_palette(old_palette, set())
        assert Palette.valid_palette(new_palette, pending_colours)
        assert all(isinstance(palette_group, list) for palette_group in old_palette)
        assert all(isinstance(palette_group, set) for palette_group in new_palette)
        # We must not mutate our arguments, as we're frequently called "speculatively"
        # TODO: Play it safe for now while experimenting, it may be we don't actually
        # try to mutate some of these and we can therefore avoid the copy.
        old_palette = copy.deepcopy(old_palette)
        new_palette = copy.deepcopy(new_palette)
        pending_colours = pending_colours - set.union(*new_palette)

        # The order of the rows in the palettes is arbitrary (we can just adjust the
        # attribute values on the pixel data as we encode it), so reorder new_palette
        # to minimise the pairwise differences.
        #print "FA0", old_palette
        #print "FA1", new_palette
        new_palette_reordered = [None] * 4
        old_palette_copy = old_palette[:]
        for new_palette_group in sorted(new_palette, key=lambda x: len(x), reverse=True):
            best_old_palette_group = None
            for old_palette_group_index, old_palette_group in enumerate(old_palette_copy):
                if old_palette_group is not None and (best_old_palette_group is None or old_palette_group is not None and len(set(old_palette_group).intersection(new_palette_group)) > len(set(best_old_palette_group).intersection(new_palette_group))):
                    best_old_palette_group = old_palette_group
                    best_old_palette_group_index = old_palette_group_index
            new_palette_reordered[best_old_palette_group_index] = new_palette_group
            old_palette_copy[best_old_palette_group_index] = None
        new_palette = new_palette_reordered
        #print "FA2", new_palette

        # Assign an index to the elements of the palette groups in the new palette,
        # re-using the index from the old palette where possible.
        #print "AOLD", old_palette
        #print "ANEW", new_palette, pending_colours
        new_palette_list = []
        for old_palette_group, new_palette_group_set in zip(old_palette, new_palette):
            new_palette_group_list = [None]*4
            #print "AB", old_palette_group, new_palette_group_set
            for i, old_colour in enumerate(old_palette_group):
                #print "WW", i, old_colour, new_palette_group_set
                if old_colour in new_palette_group_set or (
                        (len(new_palette_group_set) + len(new_palette_group_list)) < 4 and old_colour in pending_colours):
                    #print "XX", i
                    new_palette_group_list[i] = old_colour
                    new_palette_group_set.discard(old_colour)
                    pending_colours.discard(old_colour)
                else:
                    #print "YY", i
                    if len(new_palette_group_set - set(old_palette_group)) > 0:
                        #print "Q1", new_palette_group_list, new_palette_group_set
                        candidates = new_palette_group_set - set(old_palette_group)
                        new_palette_group_list[i] = min(candidates)
                        new_palette_group_set.remove(min(candidates))
                        #print "Q2", new_palette_group_list, new_palette_group_set
            new_palette_list.append(new_palette_group_list)
        new_palette = new_palette_list
        new_palette_list = None
        #print "BNEW", new_palette

        # Any remaining pending_colours need to be put into new_palette We
        # prefer putting them in emptier palette groups; this is perhaps a bit
        # arbitrary.
        while len(pending_colours) > 0:
            best_palette_group = None
            for palette_group in new_palette:
                if None in palette_group and (best_palette_group is None or palette_group.count(None) > best_palette_group.count(None)):
                    best_palette_group = palette_group
            best_palette_group[best_palette_group.index(None)] = min(pending_colours)
            pending_colours.remove(min(pending_colours))

        # If there are any leftover entries in new_palette, copy the corresponding
        # colours over from old_palette.
        for old_palette_group, new_palette_group in zip(old_palette, new_palette):
            for i in range(0, len(new_palette_group)):
                if new_palette_group[i] is None:
                    new_palette_group[i] = old_palette_group[i]

        changes = []
        for i, (old_palette_group, new_palette_group) in enumerate(zip(old_palette, new_palette)):
            for j, (old_colour, new_colour) in enumerate(zip(old_palette_group, new_palette_group)):
                if old_colour != new_colour:
                    changes.append((i*4+j, new_colour))

        return new_palette, changes

    def crystallise(self, current_palette, max_ula_changes):
        assert current_palette is None or current_palette.crystallised
        assert not self.crystallised # make this case a no-op instead?

        if current_palette is None:
            # We need a "dummy" current_palette to start from; we set max_changes to 16
            # so it can be completely replaced.
            current_palette = Palette.default_palette()
            max_changes = 16
            changes_weight = 0
        else:
            max_changes = max_ula_changes
            changes_weight = 1.1

        crystallise_pass = 0
        old_palette = current_palette.crystallised_palette
        while True:
            #print "G0", crystallise_pass
            old_hist_total_freq = sum(x[1] for x in self.hist)
            #print "G1", self.hist
            new_palette, pending_colours, self.hist, decomposed = self.crystallise_for_hist(old_palette, self.hist, max_changes, changes_weight)
            new_hist_total_freq = sum(x[1] for x in self.hist)
            #print "G2", self.hist
            #print "G3", old_hist_total_freq, new_hist_total_freq
            #assert abs(old_hist_total_freq - new_hist_total_freq) < 0.1
            if not decomposed:
                break
            crystallise_pass += 1

        #print "FINALDIFF", old_palette, new_palette, pending_colours
        self.crystallised_palette, changes = Palette.diff(old_palette, new_palette, pending_colours)
        self.crystallised = True
        return changes

    def crystallise_for_hist(self, old_palette, hist, max_changes, changes_weight):
        new_palette = [set() for i in range(0, 4)]
        pending_colours = set()

        # We iterate over the histogram using while/pop because we want to modify it as we
        # go through.
        modified_hist = []
        decomposed = False
        while len(hist) > 0:
            colour_set, freq = hist.pop(0)
            modified_hist.append((colour_set, freq))
            pending_colours -= set.union(*new_palette)
            #print
            #print "P-pre", new_palette, pending_colours
            #print "H", colour_set, freq, len(hist)

            _, changes = Palette.diff(old_palette, new_palette, pending_colours)
            assert len(changes) <= max_changes

            #print "AAA"
            if (any(colour_set.issubset(palette_group) for palette_group in new_palette) or
                    (len(colour_set) == 1 and colour_set in pending_colours)):
                # Nothing to do, we can represent this perfectly
                continue

            #print "BBB"
            if len(colour_set) == 1:
                if Palette.entries_used(new_palette, pending_colours) < 16:
                    # Single colours must go in somewhere, but we don't care where, so
                    # don't rush to place them.
                    pending_colours.add(elem(colour_set))
                    _, changes = Palette.diff(old_palette, new_palette, pending_colours)
                    if len(changes) > max_changes:
                        pending_colours.remove(elem(colour_set))
                continue

            #print "CCC"
            # Try to add a colour set to a single palette group within a palette.
            best_palette_group = None
            for palette_group_index, palette_group in enumerate(new_palette):
                #print "D", palette_group
                if len(colour_set.union(palette_group)) > 4:
                    continue

                new_palette_copy = copy.deepcopy(new_palette)
                new_palette_copy[palette_group_index].update(colour_set)
                if Palette.entries_used(new_palette_copy, pending_colours) > 16:
                    continue
                #print "EEE", new_palette_copy

                assert len(new_palette_copy[palette_group_index]) <= 4
                #print "E1", old_palette
                #print "E2", new_palette_copy, pending_colours
                _, changes = Palette.diff(old_palette, new_palette_copy, pending_colours)
                #print "E3", len(changes), changes
                #print "FFF", changes
                changes = len(changes)

                if changes > max_changes:
                    continue

                new_colours_in_group = len(colour_set - palette_group)
                new_group_size = len(palette_group.union(colour_set))

                # We give changes a slightly higher weighting so if two alternatives
                # both add the same number of new colours, we prefer the one which can
                # be done with fewest changes.
                # TODO: could probably tweak weightings here
                # TODO: We are *probably* a little too keen to fill up a nearly-full palette with a colouir triple; it would be one thing if *none* of those colours were already in the palette, but if we have two of them in separate sets this feels a little bit out of order. Then again, if the frequency count says the colour triple is next in priority perhaps this is fine.
                palette_group_score = -(changes*changes_weight + new_colours_in_group + 0.25*new_group_size)
                #print "PQ", old_palette, new_palette_copy, pending_colours
                #print "Q", palette_group, palette_group_score, changes, new_colours_in_group, new_group_size

                if best_palette_group is None or palette_group_score > best_palette_group_score:
                    best_palette_group = palette_group
                    best_palette_group_score = palette_group_score

            if best_palette_group is not None:
                saved_new_palette = copy.deepcopy(new_palette)
                best_palette_group.update(colour_set)
                if len(Palette.diff(old_palette, new_palette, pending_colours)) <= max_changes:
                    continue
                new_palette = saved_new_palette
                continue

            # The colour set can't be added as a unit, so divide its frequency count among
            # its components (colour triples decay to colour pairs, colour pairs decay to
            # single colours) and carry on.
            decomposed = True
            modified_hist.pop()
            new_hist = defaultdict(float)
            #print "H-1", colour_set, freq
            for hist_colour_set, copy_freq in hist:
                new_hist[hist_colour_set] += copy_freq
            SFTODOA = sum(x[1] for x in new_hist.items())
            SFTODOB = sum(x[1] for x in hist)
            #print "H0", SFTODOA, SFTODOB
            t = tuple(colour_set)
            f = freq / float(len(colour_set))
            #print "Z1", freq, float(len(colour_set)), f
            if len(colour_set) == 3:
                new_hist[frozenset([t[0], t[1]])] += f
                new_hist[frozenset([t[0], t[2]])] += f
                new_hist[frozenset([t[1], t[2]])] += f
            else:
                assert len(colour_set) == 2
                f *= single_boost
                new_hist[frozenset([t[0]])] += f
                new_hist[frozenset([t[1]])] += f
            SFTODOA = sum(x[1] for x in new_hist.items())
            #print "H1", SFTODOA
            hist = sorted(new_hist.items(), key=lambda x: x[1], reverse=True)
            SFTODOC = sum(x[1] for x in hist)
            #print "H2", SFTODOC
            assert abs(SFTODOA - SFTODOC) < 0.1

        pending_colours -= set.union(*new_palette)
        modified_hist.extend(hist)
        return new_palette, pending_colours, modified_hist, decomposed


            







single_boost = 2 # TODO: magic

raw_hist_by_y = [None]*ysize
for y in range(0, ysize):
    hist = defaultdict(int)
    for x in range(0, xsize, 3):
        pixel_triple_source = (pixel_map[x,y], pixel_map[x+1,y], pixel_map[x+2,y])
        pixel_triple = frozenset(aux_palette_index_by_y[y][c] for c in pixel_triple_source)
        #print "XX", frozenset(pixel_triple)
        # TODO: Even *without* trying to artifically boost singles, should be perhaps be tripling
        # singles and 1.5x-ing doubles? Gut feeling is no - a triple represents three pixels just like a
        # colour single, so why boost one more than the other? Especially since things which don't
        # get satisfied get decomposed so they can pool their votes. Yes we are boosting singles because
        # if we don't do this we might end up (we still might, but we make it less likely) unable to
        # use one of our colours algogether and that will potentially force a jarring visual glitch.
        hist[pixel_triple] += 1 * (1 if len(pixel_triple) > 1 else single_boost)
        #print "AK", pixel_triple, hist[pixel_triple]
    hist = sorted(hist.items(), key=lambda x: x[1], reverse=True)
    raw_hist_by_y[y] = hist
    #print y, hist
    #assert False

def merge_hist(hist_list):
    merged_hist = defaultdict(int)
    for hist in hist_list:
        for colour_set, freq in hist:
            merged_hist[colour_set] += freq
    return sorted(merged_hist.items(), key=lambda x: x[1], reverse=True)

#foo = merge_hist(raw_hist_by_y[0:5])
#for a, b in foo:
#    print a, b
#assert False


#a = Palette([])
#a.crystallised = True
#a.crystallised_palette = [[2, 5, 1, 7], [6, 8, 10, 13], [1, 6, 14, 15], [1, 5, 8, 10]]
#b = Palette(merge_hist(raw_hist_by_y[7:7+5]))
#print b.crystallise(a)
#assert False


ula_palette_window_size = 3
palette_by_y = [None]*ysize
for y in range(0, ysize):
    # When we get to the last few lines we will be considering fewer than ula_palette_window_size
    # lines. This is OK from a palette changing perspective as we won't back ourselves
    # into a corner as there *are* no more lines. From an "avoiding visual glitches"
    # point of view it may not be so good, as the palette may adapt dramatically and
    # suddenly new colours will appear and cause horizontal striping; if this seems to
    # happen we could always constrain the window to be the bottom ula_palette_window_size lines
    # if it would otherwise be smaller.
    window_hist = merge_hist(raw_hist_by_y[y:y+ula_palette_window_size])
    palette_by_y[y] = Palette(window_hist)
    if y == 91:
        print "AAA", y
        for a, b in window_hist:
            print a, b


palette_actions_by_y = [None]*ysize
for y in range(0, ysize):
    #print "Y", y
    palette_actions_by_y[y] = palette_by_y[y].crystallise(None if y == 0 else palette_by_y[y-1], max_ula_changes_by_line[y])
    print "Y", y, palette_by_y[y].crystallised_palette, len(palette_actions_by_y[y])






# TODO: Rename this init_nula_palette or init_aux_palette
nula_palette = bytearray()
for aux_colour, source_colour in enumerate(aux_palette_by_y[0]):
    p = image.getpalette()
    r = p[source_colour*3+0] >> 4
    g = p[source_colour*3+1] >> 4
    b = p[source_colour*3+2] >> 4
    nula_palette.extend(bytearray([(g<<4) | b, (aux_colour<<4) | r]))

def SFTODORENAME(palette):
    assert len(palette.crystallised_palette) == 4
    original_to_bbc_colour_map = defaultdict(set)
    #bbc_colour = 0
    for i, palette_group in enumerate(palette.crystallised_palette):
        for j, original_colour in enumerate(palette_group):
            original_to_bbc_colour_map[original_colour].add(i*4+j)
    return original_to_bbc_colour_map

# TODO: Rename this init_ula_palette
ula_palette = bytearray()
#print "QQQX", palette_by_y[0].crystallised_palette
bbc_colour = 0
for palette_group in palette_by_y[0].crystallised_palette:
    for original_colour in palette_group:
        ula_palette += chr((bbc_colour<<4) | (original_colour ^ 7))
        bbc_colour += 1

# We won't use this data for line 0 in the final result, but during the process of building up the
# changes for subsequent lines we may copy from it, so we need something valid. We reverse the order
# because the data is backwards in nula_palette.
aux_changes_by_line = [bytearray(nula_palette[max_aux_changes*2-1::-1])]
for y in range(1, ysize):
    line_changes = bytearray()
    max_aux_changes_for_line = max_aux_changes - int(0.5*(max_ula_changes_by_line[y] - max_ula_changes_by_line[0]))
    for aux_colour, source_colour in aux_palette_changes_by_y[y]:
        assert 0 <= aux_colour <= 15
        p = image.getpalette()
        r = p[source_colour*3+0] >> 4
        g = p[source_colour*3+1] >> 4
        b = p[source_colour*3+2] >> 4
        line_changes += chr((aux_colour<<4) | r) + chr((g<<4) | b)
    #print "QXX", y, max_aux_changes, max_aux_changes_for_line, aux_palette_changes_by_y[y], max_ula_changes_by_line[y], max_ula_changes_by_line[0]
    assert len(line_changes) <= max_aux_changes_for_line*2

    # We always make all the changes, so if we aren't using the maximum number of changes
    # we must provide some safe no-op data. If we have no changes at all we copy the
    # previous line's changes, otherwise we just redo the last change several times.
    if len(line_changes) == 0:
        line_changes = aux_changes_by_line[y-1][0:max_aux_changes_for_line*2]
    while len(line_changes) < max_aux_changes_for_line*2:
        line_changes += line_changes[-2:]
    assert len(line_changes) == max_aux_changes_for_line*2
        
    aux_changes_by_line.append(line_changes)

# Again, we won't use this line 0 data in the final result but we need valid data to copy during the
# process of building up the subsequent lines.
ula_changes_by_line = [bytearray(ula_palette[0:max_ula_changes_by_line[0]])]
stat_min_ula_changes = 1000
stat_max_ula_changes = 0
for y in range(1, ysize):
    line_changes = bytearray()
    for change in palette_actions_by_y[y]:
        line_changes += chr((change[0]<<4) | (change[1]^7))
    assert len(line_changes) <= max_ula_changes_by_line[y]
    stat_min_ula_changes = min(stat_min_ula_changes, len(line_changes))
    stat_max_ula_changes = max(stat_max_ula_changes, len(line_changes))

    # We always make all the changes, so if we aren't using the maximum number of changes
    # we must provide some safe no-op data. If we have no changes at all we copy the
    # previous line's changes, otherwise we just redo the last change several times.
    if len(line_changes) == 0:
        line_changes = ula_changes_by_line[y-1][0:max_ula_changes_by_line[y]]
    while len(line_changes) < max_ula_changes_by_line[y]:
        line_changes += chr(line_changes[-1])
    assert len(line_changes) == max_ula_changes_by_line[y]

    ula_changes_by_line.append(line_changes)

changes_per_line = 9 # in file, not "can be done without flicker"
assert changes_per_line == max_changes + 1
palette_changes = bytearray()
nulapal = 0xfe23
ulapal = 0xfe21
for y in range(0, ysize):
    palette_changes += ula_changes_by_line[y]
    palette_changes += aux_changes_by_line[y]
    palette_changes += chr(ulapal & 0xff) if len(ula_changes_by_line[y]) > len(ula_changes_by_line[0]) else chr(nulapal & 0xff)

# The line 0 "palette" data is unused as palette data, so it provides a convenient place
# for other miscellaneous info. We could potentially store the screen mode in here and
# perhaps a screen width to allow narrower images as tricky's code does.
palette_changes[0:changes_per_line] = bytearray([0]*changes_per_line)
palette_changes[0] = max_ula_changes_by_line[0]

assert len(palette_changes) == ysize * changes_per_line
with open('zrawchange', 'wb') as f:
    f.write(palette_changes)

def interleave_changes(raw_changes):
    interleaved_changes = bytearray([0])*changes_per_line*ysize
    for y in range(0, ysize):
        for i in range(0, changes_per_line):
            interleaved_changes[y+i*256] = raw_changes[y*changes_per_line+i]
    return interleaved_changes
palette_changes = interleave_changes(palette_changes)


# Write the image data out with appropriate bit-swizzling. We also make the
# same attribute-constrained modifications to our in-memory image so we can
# dump it out for viewing on the host to get an idea of how well we did without
# needing to fire up an emulator or real machine. (The resulting image is not
# identical to that on the emulator or real machine, because we don't restrict
# the palette to 12-bit colour. We could, but it seems better for flipping back
# and forth between the input and output to compare them to avoid this
# additional difference.)
image_data = bytearray()
for y_block in range(0, ysize, 8):
    print "Y:", y_block
    errors = [[[0, 0, 0] for x in range(0, xsize)] for y in range(0, ysize)]
    for x in range(0, xsize, 3):
        for y in range(y_block, y_block+8):
            #print "Y2:", y
            #if y == 2:
            #    assert False
            if False and y == 91 and x == 138:
                pixels = (14, 14, 14)
            else:
                pixels = (pixel_map[x,y], pixel_map[x+1,y], pixel_map[x+2,y])
                SFTODOorigpixels = pixels[:]
                pixels = tuple(aux_palette_index_by_y[y][c] for c in pixels)
            palette_index, adjusted_pixels = best_effort_pixel_representation(pixels, palette_by_y[y].crystallised_palette, aux_palette_by_y[y], x, y, errors)
            pixel_map[x,y] = aux_palette_by_y[y][adjusted_pixels[0]]
            pixel_map[x+1,y] = aux_palette_by_y[y][adjusted_pixels[1]]
            pixel_map[x+2,y] = aux_palette_by_y[y][adjusted_pixels[2]]
            original_to_bbc_colour_map = SFTODORENAME(palette_by_y[y])
            bbc_pixels = []
            bbc_colour_range = set(range(palette_index*4, (palette_index+1)*4))
            for original_colour in adjusted_pixels:
                bbc_pixels.append(tuple(bbc_colour_range.intersection(original_to_bbc_colour_map[original_colour]))[0] % 4)
            if y == 91 and x == 102:
                print "pm", pixel_map[x,y], pixel_map[x+1,y], pixel_map[x+2,y]
                print "auxpal", aux_palette_by_y[y]
                print "auxpali", aux_palette_index_by_y[y]
                print "ulapal", palette_by_y[y].crystallised_palette
                print "origpixels", SFTODOorigpixels
                print "pixels", pixels
                print "palidx", palette_index
                print "adjpix", adjusted_pixels                
                print "bbcpix", bbc_pixels
                #assert False

            #assert bbc_colour_map[adjusted_pixels[0]]/4 == bbc_colour_map[adjusted_pixels[1]]/4
            #assert bbc_colour_map[adjusted_pixels[1]]/4 == bbc_colour_map[adjusted_pixels[2]]/4
            #attribute_value = bbc_colour_map[adjusted_pixels[0]] / 4
            def adjust_bbc_pixel(n):
                assert 0 <= n <= 3
                return ((n & 2) << 3) | (n & 1)
            bbc_byte = ((adjust_bbc_pixel(bbc_pixels[0]) << 3) |
                        (adjust_bbc_pixel(bbc_pixels[1]) << 2) |
                        (adjust_bbc_pixel(bbc_pixels[2]) << 1) |
                        adjust_bbc_pixel(palette_index))
            image_data += chr(bbc_byte)


# Save the attribute-constrained version of the image.
simulated_image = image.resize((1280, 1024), resample=PIL.Image.NEAREST)
if len(sys.argv) == 4:
    simulated_image.save(sys.argv[3])
else:
    simulated_image.save("z.png")

assert len(ula_palette) == 16
assert len(nula_palette) == 32
assert len(palette_changes) == changes_per_line * 256
assert len(image_data) == 20*1024
with open(sys.argv[2], "wb") as bbc_image:
    bbc_image.write(ula_palette)
    bbc_image.write(nula_palette)
    bbc_image.write(palette_changes)
    bbc_image.write(image_data)
component = os.path.splitext(sys.argv[2])
with open(component[0] + ".txt", "w") as bbc_image_txt:
    bbc_image_txt.write("Min/max ULA palette changes per line: %d/%d\n" % (stat_min_ula_changes, stat_max_ula_changes))
    bbc_image_txt.write("Min/max aux palette changes per line: %d/%d\n" % (stat_min_aux_changes, stat_max_aux_changes))
