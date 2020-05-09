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




if len(sys.argv) != 3:
    sys.stderr.write('Usage: %s INFILE OUTFILE\n' % sys.argv[0])
    sys.exit(1)

image = PIL.Image.open(sys.argv[1])
xsize, ysize = image.size
assert xsize == 240
assert ysize == 256
# TODO: verify it's an indexed colour image with 16 or fewer colours
pixel_map = image.load()

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
        old_palette_set = [set(palette_group) for palette_group in old_palette]
        original_new_palette = new_palette
        new_palette = []
        for old_palette_group in old_palette_set:
            best_new_palette_group = None
            for new_palette_group in original_new_palette:
                if best_new_palette_group is None or len(old_palette_group.intersection(new_palette_group)) > len(old_palette_group.intersection(best_new_palette_group)):
                    best_new_palette_group = new_palette_group
            new_palette.append(best_new_palette_group)
            original_new_palette.remove(best_new_palette_group)

        # Assign an index to the elements of the palette groups in the new palette,
        # re-using the index from the old palette where possible.
        new_palette_list = []
        for old_palette_group, new_palette_group_set in zip(old_palette, new_palette):
            new_palette_group_list = []
            for i, old_colour in enumerate(old_palette_group):
                if old_colour in new_palette_group_set or (
                        len(new_palette_group_set) < 4 and old_colour in pending_colours):
                    new_palette_group_list.append(old_colour)
                    pending_colours.discard(old_colour)
                else:
                    if len(new_palette_group_set) > 0:
                        new_palette_group_list.append(min(new_palette_group_set))
                        new_palette_group_set.remove(min(new_palette_group_set))
                    else:
                        new_palette_group_list.append(None)
            new_palette_list.append(new_palette_group_list)
        new_palette = new_palette_list
        new_palette_list = None

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

    def crystallise(self, current_palette):
        assert current_palette is None or current_palette.crystallised
        assert not self.crystallised # make this case a no-op instead?

        if current_palette is None:
            # We need a "dummy" current_palette to start from; we set max_changes to 16
            # so it can be completely replaced.
            current_palette = Palette.default_palette()
            max_changes = 16
            changes_weight = 0
        else:
            max_changes = 8
            changes_weight = 1.1

        old_palette = current_palette.crystallised_palette
        new_palette = [set() for i in range(0, 4)]
        pending_colours = set()

        # We iterate over the histogram using while/pop because we want to modify it as we
        # go through.
        while len(self.hist) > 0:
            colour_set, freq = self.hist.pop(0)
            print "H", colour_set, freq, len(self.hist)
            pending_colours -= set.union(*new_palette)
            print "P", new_palette, pending_colours

            _, changes = Palette.diff(old_palette, new_palette, pending_colours)
            assert len(changes) <= max_changes

            new_palette_entries_used = Palette.entries_used(new_palette, pending_colours) 
            if new_palette_entries_used == 16:
                # If we get here we have no further choices to make, so no point
                # examining further histogram entries.
                break

            #print "AAA"
            if (any(colour_set.issubset(palette_group) for palette_group in new_palette) or
                    (len(colour_set) == 1 and colour_set in pending_colours)):
                # Nothing to do, we can represent this perfectly
                continue

            #print "BBB"
            if len(colour_set) == 1:
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
            for palette_group in new_palette:
                #print "D", palette_group
                if len(colour_set.union(palette_group)) > 4:
                    continue

                pgi = new_palette.index(palette_group)
                new_palette_copy = copy.deepcopy(new_palette)
                new_palette_copy[pgi].update(colour_set)
                if Palette.entries_used(new_palette_copy, pending_colours) > 16:
                    continue
                #print "EEE", new_palette_copy

                assert len(new_palette_copy[pgi]) <= 4
                _, changes = Palette.diff(old_palette, new_palette_copy, pending_colours)
                #print "FFF", changes
                changes = len(changes)

                if changes > max_changes:
                    continue

                new_colours_in_group = len(colour_set - palette_group)

                # We give changes a slightly higher weighting so if two alternatives
                # both add the same number of new colours, we prefer the one which can
                # be done with fewest changes.
                # TODO: could probably tweak weightings here
                # TODO: We could possibly try to consider "how much space is left
                # in the palette group" as a factor here - we currently add positive
                # weight for adding as few colours as possible to an existing palette
                # group, which will encourage "clumping", but we might also want to
                # add positive weight for "leaving space free for later additions". I
                # think these are obviously opposing tendencies but I don't think they're
                # the same, e.g. suppose we want to add (1, 2) and the palette looks
                # like this: [(4, 3, 2), (), (), ()]. Adding as few colours as possible
                # favours putting 1 in with (4, 3, 2), but then that group is full so
                # maybe we should put (1, 2) in one of the empty groups.
                palette_group_score = -(changes*changes_weight + new_colours_in_group)

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
            new_hist = defaultdict(int)
            for colour_set, freq in self.hist:
                new_hist[colour_set] += freq
            t = tuple(colour_set)
            f = freq / float(len(colour_set))
            for i in range(0, len(colour_set)):
                for j in range(i+1, len(colour_set)):
                    new_hist[frozenset([t[i], t[j]])] += f
            self.hist = sorted(new_hist.items(), key=lambda x: x[1], reverse=True)


            



        pending_colours -= set.union(*new_palette)
        assert len(pending_colours) == 0 # TODO CAN PROBABLY FAIL BUT WORRY WHEN IT HAPPENS, WE JUSTNEED TO STUFF THE COLOURS IN TAKING INTO ACCOUNT OLD PALETTE








        self.crystallised_palette, changes = Palette.diff(old_palette, new_palette, pending_colours)
        self.crystallised = True
        return changes





raw_hist_by_y = [None]*ysize
for y in range(0, ysize):
    hist = defaultdict(int)
    for x in range(0, xsize, 3):
        pixel_triple = (pixel_map[x,y], pixel_map[x+1,y], pixel_map[x+2,y])
        hist[frozenset(pixel_triple)] += 1
    hist = sorted(hist.items(), key=lambda x: x[1], reverse=True)
    raw_hist_by_y[y] = hist
    #print y, hist

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

window_size = 5
palette_by_y = [None]*ysize
for y in range(0, ysize):
    # When we get to the last few lines we will be considering fewer than window_size
    # lines. This is OK from a palette changing perspective as we won't back ourselves
    # into a corner as there *are* no more lines. From an "avoiding visual glitches"
    # point of view it may not be so good, as the palette may adapt dramatically and
    # suddenly new colours will appear and cause horizontal striping; if this seems to
    # happen we could always constrain the window to be the bottom window_size lines
    # if it would otherwise be smaller.
    window_hist = merge_hist(raw_hist_by_y[y:y+window_size])
    palette_by_y[y] = Palette(window_hist)


for y in range(0, ysize):
    print "Y", y
    palette_actions = palette_by_y[y].crystallise(None if y == 0 else palette_by_y[y-1])
    print "Y", y, palette_by_y[y].crystallised_palette





assert False




hist_all = defaultdict(int)
for y in range(0, ysize):
    for colour_set, freq in hist_by_y[y]:
        hist_all[colour_set] += freq
hist_all = sorted(hist_all.items(), key=lambda x: x[1], reverse=True)
#for colour_set, freq in hist_all:
#    print colour_set, freq

common_palette = [set() for i in range(0, 4)]
common_colours = 8
pending_unpaired_colours = set()
for colour_set, freq in hist_all:
    palette_union = set.union(*common_palette)
    palette_union_with_unpaired = palette_union.union(pending_unpaired_colours)
    print
    print len(palette_union_with_unpaired), palette_union, pending_unpaired_colours
    print "A", colour_set, freq
    assert len(palette_union_with_unpaired) <= common_colours
    palette_full = (len(palette_union_with_unpaired) == common_colours)
    if palette_full and len(pending_unpaired_colours) == 0:
        break
    assert len(colour_set) <= 2
    if colour_set.issubset(palette_union):
        # Nothing to do; we only allow each colour to appear once in the common
        # palette.
        pass
    elif len(colour_set) == 1:
        if not palette_full:
            # Colours used unpaired must appear *somewhere*. but we don't commit to
            # anywhere yet to keep options open for paired colours.
            pending_unpaired_colours.add(tuple(colour_set)[0])
    else:
        new_colours = colour_set - palette_union
        colours_needed = len(new_colours)
        assert 1 <= colours_needed <= 2
        if colours_needed == 1:
            new_colour = tuple(new_colours)[0]
            old_colour = tuple(colour_set - set([new_colour]))[0]
            if not palette_full or new_colour in pending_unpaired_colours:
                # If there's space in the palette group containing old_colour, add
                # new_colour to it. Otherwise there's not much we can do.
                # TODO: Should we include the colour as a single colour? Not sure.
                for palette_group in common_palette:
                    if old_colour in palette_group:
                        if len(palette_group) < 4:
                            palette_group.add(new_colour)
                        break
        else:
            if len(palette_union_with_unpaired.union(new_colours)) <= common_colours:
                # Add both colours to the emptiest palette group we can find. If
                # there isn't one (unlikely, but if common_colours is high it may
                # happen) there's not much we can do.
                best_palette_group = None
                for palette_group in common_palette:
                    if len(palette_group) <= 2 and (best_palette_group is None or len(palette_group) < len(best_palette_group)):
                        best_palette_group = palette_group
                if best_palette_group is not None:
                    best_palette_group.update(new_colours)

        palette_union = set.union(*common_palette)
        pending_unpaired_colours -= palette_union

# TODO: I think this assert could fail in principle, and we would need to pick an arbitrary
# palette group (should we prefer a full or empty one?) to put it in. Not a big deal, but
# I'm going to wait until it happens before I worry about that.
assert len(pending_unpaired_colours) == 0

print "PUC", pending_unpaired_colours
print "Common palette:", common_palette
#visualise_palette(common_palette, "zpal.png")
common_palette_union = set.union(*common_palette)
#SFTODO = raw_input("")

if False:
    palette = []
    for i in range(0, 4):
        palette.append(set(range(i*4, (i+1)*4)))
    visualise_palette(palette, "zpal.png")
    palette = None

# TODO: Code very similar to previous common_palette build up, but let's just write it
# out again for the moment to ease tinkering
# Build up per-line palettes based on the common palette. Note that because the remaining
# palette entries are changed on a per-line basis, we don't insist on each colour appearing
# no more than once here; if it's helpful, we allow duplicates.
palette_by_y = [None]*ysize
for y in range(0, ysize):
    print
    print "Y", y
    palette = copy.deepcopy(common_palette)
    pending_unpaired_colours = set()
    for colour_set, freq in hist_by_y[y]:
        palette_union = set.union(*palette) # note that while valid, this may lose the fact some colours are in the palette multiple times, so its *length* is kind of meaningless
        palette_entries_used = sum(len(palette_group) for palette_group in palette) + len(pending_unpaired_colours)
        print
        print palette_entries_used, palette, pending_unpaired_colours
        print "B", colour_set, freq
        assert palette_entries_used <= 16
        palette_full = (palette_entries_used == 16)
        if palette_full and len(pending_unpaired_colours) == 0:
            break
        # If the current palette perfectly handles this colour set, no action is needed.
        if any(colour_set.issubset(palette_group) for palette_group in palette):
            continue
        assert 1 <= len(colour_set) <= 2
        if len(colour_set) == 1:
            new_colour = tuple(colour_set)[0]
            if new_colour not in palette_union and new_colour not in pending_unpaired_colours and not palette_full:
                pending_unpaired_colours.add(new_colour)
        else:
            best_palette_group = None
            colour_a = tuple(colour_set)[0]
            colour_b = tuple(colour_set)[1]
            print "AAA", colour_a, colour_b
            for palette_group in palette:
                if colour_a in palette_group:
                    print "QQ"
                    assert colour_b not in palette_group
                    if len(palette_group) < 4 and (best_palette_group is None or len(palette_group) < len(best_palette_group)):
                        best_palette_group = palette_group
                elif colour_b in palette_group:
                    print "ZZ"
                    assert colour_a not in palette_group
                    if len(palette_group) < 4 and (best_palette_group is None or len(palette_group) < len(best_palette_group)):
                        best_palette_group = palette_group
            if best_palette_group is not None:
                print "WW", palette_full
                # We found a palette group with one of our two colours and room for the other.
                new_colour = tuple(colour_set - best_palette_group)[0]
                print "WW2", new_colour, colour_set - best_palette_group
                if not palette_full or new_colour in pending_unpaired_colours:
                    best_palette_group.add(new_colour)
            else:
                # The only way we're going to be able to use this colour pair is by adding them
                # both to the same palette group.
                best_palette_group = None
                for palette_group in palette:
                    if len(palette_group) <= 2 and (best_palette_group is None or len(palette_group) < len(best_palette_group)):
                        best_palette_group = palette_group
                if best_palette_group is not None:
                    new_palette_entries_taken = len(colour_set - pending_unpaired_colours)
                    if palette_entries_used + new_palette_entries_taken <= 16:
                        best_palette_group.update(colour_set)

        palette_union = set.union(*palette)
        pending_unpaired_colours -= palette_union

    for colour in pending_unpaired_colours:
        for palette_group in palette:
            if len(palette_group) < 4:
                palette_group.add(colour)
                break

    # Just to keep things simple (since it won't hurt for now) if we have some spare entries
    # in the palette we force a colour into them so all palettes do have 16 entries.
    for palette_group in palette:
        while len(palette_group) < 4:
            new_colour = tuple(set(range(0, 16)) - palette_group)[0]
            palette_group.add(new_colour)



    print "PUC", pending_unpaired_colours
    print "Y palette:", palette
    palette_by_y[y] = palette

    #visualise_palette(palette, "zpal.png")
    #SFTODO = raw_input("")


nula_palette = bytearray()
for original_colour in range(0, 16):
    p = image.getpalette()
    r = p[original_colour*3+0] >> 4
    g = p[original_colour*3+1] >> 4
    b = p[original_colour*3+2] >> 4
    nula_palette.extend(bytearray([(g<<4) | b, (original_colour<<4) | r]))

def SFTODORENAME(palette, common_palette):
    assert len(palette) == 4
    bbc_to_original_colour_map = []
    original_to_bbc_colour_map = defaultdict(set)
    for i, (palette_group, common_palette_group) in enumerate(zip(palette, common_palette)):
        assert len(palette_group) == 4
        ordered = sorted(common_palette_group) + sorted(palette_group - common_palette_group)
        assert len(ordered) == 4
        bbc_to_original_colour_map.extend(ordered)
        for j, original_colour in enumerate(ordered):
            #print "Q", i, j, original_colour
            original_to_bbc_colour_map[original_colour].add(i*4+j)
    return bbc_to_original_colour_map, original_to_bbc_colour_map

bbc_to_original_colour_map, original_to_bbc_colour_map = SFTODORENAME(palette_by_y[0], common_palette)
ula_palette = bytearray()
for bbc_colour, original_colour in enumerate(bbc_to_original_colour_map):
    ula_palette += chr((bbc_colour<<4) | (original_colour ^ 7))

changes_per_line = 8 # in file, not "can be done without flicker"
ula_palette_changes = bytearray(ula_palette[0:changes_per_line]) # line 0, won't be read but we use "realistic" data so we can copy from it to subsequent lines safely
previous_bbc_to_original_colour_map = bbc_to_original_colour_map
for y in range(1, ysize):
    bbc_to_original_colour_map, original_to_bbc_colour_map = SFTODORENAME(palette_by_y[y], common_palette)
    line_changes = bytearray()
    for bbc_colour, (previous_original_colour, new_original_colour) in enumerate(zip(previous_bbc_to_original_colour_map, bbc_to_original_colour_map)):
        if previous_original_colour != new_original_colour:
            if y == 110:
                print "EZ", bbc_colour, previous_original_colour, new_original_colour
            line_changes += chr((bbc_colour<<4) | (new_original_colour ^ 7))
    #print y, len(line_changes)
    assert len(line_changes) <= 16 - common_colours
    previous_bbc_to_original_colour_map = bbc_to_original_colour_map
    if y == 110:
        print "LC", line_changes
        #assert False
    if False:
        # HACK FOR DEBUGGING (IN COMBINATION WITH PROCstripes)
        if y<=10:
            line_changes = bytearray([0x60 | (15^7)])
        else:
            line_changes = bytearray([0x60 | ((y % 15) ^ 7)]) # TODO: "SHOULD" BE % 16 BUT LEAVE IT FOR NOW

    # We always make all the changes, so if we aren't using the maximum number of changes
    # we must provide some safe no-op data. If we have no changes at all we copy the
    # previous line's changes, otherwise we just redo the last change several times.
    if len(line_changes) == 0:
        line_changes = ula_palette_changes[-changes_per_line:]
    else:
        while len(line_changes) < changes_per_line:
            line_changes += chr(line_changes[-1])
    assert len(line_changes) == changes_per_line
    ula_palette_changes.extend(line_changes)
assert len(ula_palette_changes) == ysize * changes_per_line
with open('zrawchange', 'wb') as f:
    f.write(ula_palette_changes)
def interleave_changes(raw_changes):
    interleaved_changes = bytearray([0])*changes_per_line*ysize
    for y in range(0, ysize):
        for i in range(0, changes_per_line):
            interleaved_changes[y+i*256] = raw_changes[y*changes_per_line+i]
    return interleaved_changes
ula_palette_changes = interleave_changes(ula_palette_changes)


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
    for x in range(0, xsize, 3):
        for y in range(y_block, y_block+8):
            #print "Y2:", y
            #if y == 2:
            #    assert False
            if False and y_block == 13*8 and (x >= 153 and x <= 155):
                pixels = (15, 15, 15)
            else:
                pixels = (pixel_map[x,y], pixel_map[x+1,y], pixel_map[x+2,y])
            palette_index, adjusted_pixels = best_effort_pixel_representation(pixels, palette_by_y[y])
            pixel_map[x,y] = adjusted_pixels[0]
            pixel_map[x+1,y] = adjusted_pixels[1]
            pixel_map[x+2,y] = adjusted_pixels[2]
            bbc_to_original_colour_map, original_to_bbc_colour_map = SFTODORENAME(palette_by_y[y], common_palette)
            bbc_pixels = []
            bbc_colour_range = set(range(palette_index*4, (palette_index+1)*4))
            for original_colour in adjusted_pixels:
                bbc_pixels.append(tuple(bbc_colour_range.intersection(original_to_bbc_colour_map[original_colour]))[0] % 4)
            if y <= 1:
                print "pal", palette_by_y[y]
                print "pixels", pixels
                print "palidx", palette_index
                print "adjpix", adjusted_pixels                
                print "bbcpix", bbc_pixels
                #if y == 1:
                #    assert False

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
simulated_image.save("z.png")

assert len(ula_palette) == 16
assert len(nula_palette) == 32
assert len(ula_palette_changes) == changes_per_line * 256
assert len(image_data) == 20*1024
with open(sys.argv[2], "wb") as bbc_image:
    bbc_image.write(ula_palette)
    bbc_image.write(nula_palette)
    bbc_image.write(ula_palette_changes)
    bbc_image.write(image_data)
