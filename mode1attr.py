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
import math
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
    #return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2) + math.pow(a[2] - b[2], 2))
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

def best_effort_palette_group_lookup(local_map, desired_colour, palette_group):
    if desired_colour in local_map:
        return local_map[desired_colour]
    already_used_colours = set(c[0] for c in local_map.values())
    best_colour = None
    for colour in palette_group:
        error = colour_error(desired_colour, colour)
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

# TODO: I wonder if when we can't get the exact colour, we should avoid the closest match *if* it would result in making two pixels in this triplet identical when they previously weren't; in that case take the second closest match (perhaps not if it's "very different"). My thinking here is that while we might change the colour of things, we'd hopefully do so with some consistency and this would avoid losing detail in dithering and replacing it with flat colours giving that ugly-ish horizontal mini-stripe attribute appearance.
def best_effort_pixel_representation(pixels, palette):
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
            adjusted_pixel, error = best_effort_palette_group_lookup(local_map, pixel, palette_group)
            local_map[pixel] = (adjusted_pixel, error)
            adjusted_pixels.append(adjusted_pixel)
            total_error += error
        #print "QQQ", local_map, total_error
        if best_palette_group is None or total_error < best_total_error:
            best_palette_group = i
            best_total_error = total_error
            best_adjusted_pixels = adjusted_pixels
    return best_palette_group, best_adjusted_pixels

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



if len(sys.argv) < 3 or len(sys.argv) > 4:
    sys.stderr.write('Usage: %s INFILE OUTFILE [OUTFILESIM]\n' % sys.argv[0])
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

palette = [set() for i in range(0, 4)]
if True: # SFTODO: Optional clustering palette generation as first step
    from numpy import array
    from scipy.cluster.vq import vq, kmeans, kmeans2, whiten # SFTODO DON'T NEED KMEANS AND KMEANS2 AND MAY NOT NEED VQ OR WHITEN
    p = image.getpalette()
    features = []
    for colour in range(0, 16):
        features.append([float(SFTODO) for SFTODO in image_palette_rgb(colour)])
    features = array(features)
    print features
    if False:
        whitened = whiten(features)
        best_codebook = None
        for k in range(2, 17):
            codebook, distortion = kmeans(whitened, k)
            if best_codebook is None or distortion < best_distortion:
                best_codebook = codebook
                best_distortion = distortion
            print k, distortion
            if k == 6:
                break # SFTODO TEMP HACK - CURRENT CODE IS SILLY THO BECAUSE 16 GROUPS WILL ALWAYS GIVE BEST (0) DISTORTION!
        print best_codebook
        print distortion

        colour_to_colour_class_map = {}
        colour_class_to_colour_map = defaultdict(set)
        for colour in range(0,16):
            best_colour_class = None
            for colour_class, centroid in enumerate(best_codebook):
                d = distance(centroid, image_palette_rgb(colour))
                if best_colour_class is None or d < best_distance:
                    best_colour_class = colour_class
                    best_distance = d
            colour_to_colour_class_map[colour] = best_colour_class
            colour_class_to_colour_map[best_colour_class].add(colour)
        print colour_to_colour_class_map
        print colour_class_to_colour_map
    elif False:
        k = 5 # SFTODO
        centroid, label = kmeans2(features, k, minit="points", iter=10000000)
        colour_class_to_colour_map = defaultdict(set)
        for colour, colour_class in enumerate(label):
            colour_class_to_colour_map[colour_class].add(colour)
        print centroid
        print label
        print colour_class_to_colour_map
    elif False:
        from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
        from scipy.spatial.distance import pdist
        from matplotlib import pyplot as plt
        #Z = linkage(features, method='ward') # SFTODO EXPERIMENT WITH METHOD
        y = pdist(features, metric=distance)
        Z = linkage(y, method='average') # SFTODO EXPERMENT WITH METHOD - NOTE NOT ALL VALID WHEN WE PASS y
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)
        plt.show()
        colour_to_colour_class_map = fcluster(Z, t=8, criterion='distance') # SFTODO EXPERIMENT WITH T, CRITERION
        # fcluster() starts group numbers at 1; adjust to start at 0.
        colour_to_colour_class_map = [n-1 for n in colour_to_colour_class_map]
        colour_class_to_colour_map = defaultdict(set)
        for colour, colour_class in enumerate(colour_to_colour_class_map):
            colour_class_to_colour_map[colour_class].add(colour)
        print colour_to_colour_class_map
        print colour_class_to_colour_map

        data = list(image.getdata())
        hist_class = defaultdict(int)
        for i in range(0, len(data), 3):
            pixel_triple = [colour_to_colour_class_map[colour] for colour in data[i:i+3]]
            def do_pair(i, j):
                # We use a set because the order of the two colours is irrelevant.
                if pixel_triple[i] != pixel_triple[j]:
                    hist_class[frozenset([pixel_triple[i], pixel_triple[j]])] += 1
            do_pair(0, 1)
            do_pair(0, 2)
            do_pair(1, 2)
        hist_class = sorted(hist_class.items(), key=lambda x: x[1], reverse=True)
        for hist_class_entry in hist_class:
            print "%s\t%s" % (hist_class_entry[0], hist_class_entry[1])
        # We don't process the entire colour class histogram; once we've covered "a reasonable
        # fraction" of the "votes", we stop. This avoids packing out the palette to handle
        # relatively rare cases and leaves more scope for the colour histogram to influence
        # the palette directly. TODO: 0.8 IS OBVIOUSLY MAGIC NUM
        cutoff_cumulative_frequency = 1.0 * sum(hist_class_entry[1] for hist_class_entry in hist_class)
        cumulative_frequency = 0
        for hist_class_entry in hist_class:
            colour_class_set = hist_class_entry[0]
            cumulative_frequency += hist_class_entry[1]
            if cumulative_frequency > cutoff_cumulative_frequency:
                break
            print
            print "A", colour_class_set
            assert len(colour_class_set) == 2
            #palette_union = set(colour_to_colour_class_map[colour] for colour in set.union(*palette))
            palette_union = set.union(*palette)
            print "X", palette
            if len(palette_union) >= 15:
                # Just a minor optimisation; if we've already got 15 colours in the
                # palette there's no choice to be made any more, because we insist
                # all 16 colours are present.
                break
            palette_colour_class = [set(colour_to_colour_class_map[colour] for colour in palette_group) for palette_group in palette]
            print "Y", palette_colour_class
            done = False
            for palette_colour_class_group in palette_colour_class:
                if colour_class_set.issubset(palette_colour_class_group):
                    # The palette already has a group which contains a colour from both of these colour classes, so
                    # don't do anything else with it here.
                    done = True
            if not done:
                # If there's a palette group which contains a colour from one of these colour classes,
                # we will add a colour from the other colour class. Obviously we need a palette group
                # which has space, and if there's more than one we pick the emptiest. TODO: We could
                # probably be a lot smarter about picking when there are multiple possibilities.
                best_palette_group = None
                for palette_group, palette_colour_class_group in zip(palette, palette_colour_class):
                    intersection = colour_class_set.intersection(palette_colour_class_group)
                    if len(intersection) == 1 and len(palette_group) < 4:
                        colour_group_to_add = colour_class_set - intersection
                        assert len(colour_group_to_add) == 1
                        # If the colour group we would be adding to this palette entry has no free
                        # colours left, it's no good to us - keep going and we might find one where
                        # the other colour group is to be added instead.
                        if pick_colour_from_colour_class(palette, palette_group, tuple(colour_group_to_add)[0]) is not None:
                            if (best_palette_group is None or len(palette_group) < len(best_palette_group)):
                                best_palette_group = palette_group
                                best_palette_group_intersection = intersection
                if best_palette_group is not None:
                    colour_group_to_add = colour_class_set - best_palette_group_intersection
                    print "Q", colour_group_to_add
                    assert len(colour_group_to_add) == 1
                    colour_to_add = pick_colour_from_colour_class(palette, palette_group, tuple(colour_group_to_add)[0])
                    assert colour_to_add is not None # SFTODO NOT SURE THIS ASSERTION CAN'T TRIGGER IN PRINCIPLE, IN WHICH CASE WE JUST DON'T DO ANYTHING I GUESS
                    best_palette_group.add(colour_to_add)
                    done = True
            if not done:
                # No palette group contains a colour from both of these colour classes, nor could
                # we find an existing palette group with a colour from one of the classes to which
                # we could add a colour from the other class, for whatever reason. If there's at
                # least one unused colour in each colour class and a palette group with at least
                # two spaces free we will add them. If there's more than one we prefer the emptiest.
                # TODO: We could probably be smarter about picking one where we have a choice.
                # If we can't find one there's nothing we can do.
                emptiest_palette_group = None
                for palette_group in palette:
                    if len(palette_group) <= 2 and (
                            emptiest_palette_group is None or 
                            len(palette_group) < len(emptiest_palette_group)):
                        emptiest_palette_group = palette_group
                        emptiest_palette_group_len = len(palette_group)
                if emptiest_palette_group is not None:
                    colours_to_add = set(pick_colour_from_colour_class(palette, emptiest_palette_group, colour_class) for colour_class in colour_class_set)
                    if None not in colours_to_add:
                        emptiest_palette_group.update(colours_to_add)



        visualise_palette(palette, "zpal2.png")
        #SFTODO = raw_input()
        #assert False

for hist_entry in hist:
    print "%s\t%s" % (hist_entry[0], hist_entry[1])
#assert False

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
simulated_image = image.resize((1280, 1024), resample=PIL.Image.NEAREST)
if len(sys.argv) == 4:
    simulated_image.save(sys.argv[3])
else:
    simulated_image.save("z.png")
