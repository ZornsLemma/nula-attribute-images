from collections import defaultdict
import PIL.Image
import argparse
import heapq
import itertools
import math
import os
import sys


# Terminology:
#
# ULA palette - the palette present in the standard ULA, programmed by writing to &FE21
# Auxiliary palette - the VideoNuLA auxiliary palette, programmed by writing to &FE23
# Source palette - the palette on the source image we are converting


# Since VideoNuLA uses 4-bit RGB, we use that internally.
max_rgb = 15

writes_per_line = 9
writes_per_aux_change = 2
aux_palette_size = 16

# TODO: I did experiment with swapping the fractions for (1, 0) and (0, 1), since our
# horizontal triplets make it "hard" to distribute errors to the right, whereas the
# next line will be addressed at least semi-fresh. It had debatable value at the time,
# as the code evolves and gets tweaked it might be worth trying this again later.
error_dist = [
    ( 1, 0, 7.0/16),
    ( 1, 1, 1.0/16),
    ( 0, 1, 5.0/16),
    (-1, 1, 3.0/16)]

# TODO: Magic/tweakable parameter
single_boost = 2


def error_unless(b, message):
    if not b:
        sys.stderr.write(message + '\n')
        sys.exit(1)


# TODO: Eventually this might be controlled by some kind of --verbose command line option
def info(s):
    print s


def elem(s):
    [e] = s
    return e


def build_source_palette():
    palette_size = 1 + max(source_pixel_map[x, y] for x in range(x_size) for y in range(y_size))
    p = source_image.getpalette()
    # TODO: Should we divide by 0x11 rather than shift right by 4? Almost certainly not, as that
    # way e.g. only 0xff would be mapped to 15.
    return tuple((r >> 4, g >> 4, b >> 4) for r, g, b in 
            ((p[i*3+0], p[i*3+1], p[i*3+2]) for i in range(palette_size)))


def build_image_rgb():
    result = []
    for y in range(y_size):
        row = []
        for x in range(x_size):
            row.append(source_palette[source_pixel_map[x, y]])
        result.append(row)
    return result


def squash_palette(source_palette, image_rgb):
    new_palette = []
    rgb_to_index = {}
    for i, rgb in enumerate(source_palette):
        if rgb in rgb_to_index:
            info('Source palette entry %d is a duplicate of entry %d in our RGB space; removing it' % (i, rgb_to_index[rgb]))
            # TODO: Will this cause any problems? I suspect not but if we ever try to
            # "refer back" to the original image we might run into difficulties.
        else:
            rgb_to_index[rgb] = i
            new_palette.append(rgb)

    # Use source_palette not new_palette here so reported index is meaningful.
    for i, rgb in enumerate(source_palette):
        used = False
        for row in image_rgb:
            used = used or rgb in row
            if used:
                break
        if not used:
            info('Source palette entry %d is not used in the image; removing it' % (i,))
            if i in new_palette:
                new_palette.remove(i)

    return tuple(new_palette)


# TODO: Rename? We may have different types of histogram, this is *not* by triples.
def build_hist(image_rows):
    hist = defaultdict(int)
    for row in image_rows:
        for pixel_rgb in row:
            hist[pixel_rgb] += 1
    return hist.items()


def add_rgb(rgb_a, rgb_b):
    return (rgb_a[0] + rgb_b[0], rgb_a[1] + rgb_b[1], rgb_a[2] + rgb_b[2])


def sub_rgb(rgb_a, rgb_b):
    return (rgb_a[0] - rgb_b[0], rgb_a[1] - rgb_b[1], rgb_a[2] - rgb_b[2])


def mult_rgb(rgb, f):
    return (rgb[0] * f, rgb[1] * f, rgb[2] * f)


def clamp_rgb(rgb):
    return (max(0, min(rgb[0], max_rgb)),
            max(0, min(rgb[1], max_rgb)),
            max(0, min(rgb[2], max_rgb)))


# TODO: We should maybe have command line options for these two distance measures,
# both with and without square root (for 4 possibilities), and we should maybe allow
# the power applied to the frequency to be specified on command line too.
def distance(rgb_a, rgb_b):
    if True: # TODO EXPERIMENT
        return math.sqrt(math.pow(rgb_a[0] - rgb_b[0], 2) +
                         math.pow(rgb_a[1] - rgb_b[1], 2) +
                         math.pow(rgb_a[2] - rgb_b[2], 2))

    # https://bisqwit.iki.fi/story/howto/dither/jy/
    luma1 = (rgb_a[0]*299 + rgb_a[1]*587 + rgb_a[2]*114) / (max_rgb*1000)
    luma2 = (rgb_b[0]*299 + rgb_b[1]*587 + rgb_b[2]*114) / (max_rgb*1000)
    lumadiff = luma1-luma2
    diffR = (rgb_a[0]-rgb_b[0])/float(max_rgb)
    diffG = (rgb_a[1]-rgb_b[1])/float(max_rgb)
    diffB = (rgb_a[2]-rgb_b[2])/float(max_rgb)
    # TODO: I ADDED SQRT - AS WITH EUCLIDEAN DEBATABLE WHETHER OR NOT SQRT IS HELPFUL
    return math.sqrt((diffR*diffR*0.299 + diffG*diffG*0.587 + diffB*diffB*0.114)*0.75 + lumadiff*lumadiff)


def aux_palette_best_match(pixel_rgb, aux_palette):
    best_palette_rgb = None
    for palette_rgb in aux_palette:
        error = distance(pixel_rgb, palette_rgb)
        if best_palette_rgb is None or error < best_palette_error:
            best_palette_rgb = palette_rgb
            best_palette_error = error
    return best_palette_rgb


# TODO: Awkward duplication of code with aux_palette_best_match
def aux_palette_best_match_index(pixel_rgb, aux_palette):
    best_palette_index = None
    for i, palette_rgb in enumerate(aux_palette):
        error = distance(pixel_rgb, palette_rgb)
        if best_palette_index is None or error < best_palette_error:
            best_palette_index = i
            best_palette_error = error
    return best_palette_index


def aux_palette_error(hist, aux_palette):
    error_by_pixel_rgb = {}
    for pixel_rgb, freq in hist:
        best_aux_palette_rgb = aux_palette_best_match(pixel_rgb, aux_palette)
        # TODO: Is 'distance * freq' the best measure here? For example, just maybe we
        # should square freq? Or if we remove the sqrt() inside distance, we would get a
        # different effect too.
        error_by_pixel_rgb[pixel_rgb] = distance(pixel_rgb, best_aux_palette_rgb) * freq
    return error_by_pixel_rgb.items()


# We represent an auxiliary palette as a set of RGB tuples. While at the moment these will
# always be RGB tuples which appear in the source palette, it is at least conceivable that
# eventually we will say "since we can only select 16 colours per row, we will make one
# of our colours an intermediate shade between two of the source palette colours as a
# compromise" or similar. If we used indices into the source palette, we'd lose the
# flexibility to support that.
def build_aux_palette_by_y():
    # TODO: This is another tweakable parameter and needs experimentation.
    aux_palette_window_size = 2

    aux_palette_set_by_y = []
    for y in range(y_size):
        hist = build_hist(image_rgb[y:y+aux_palette_window_size])
        aux_palette = set([image_rgb[0][0]]) if y == 0 else aux_palette_set_by_y[y-1].copy()
        #print "PX", y, len(aux_palette)

        # Add colours into aux_palette one at a time, picking the one which is causing
        # the biggest overall error in the representation each time. We don't do all n
        # colours at once because adding one colour may mean another colour can suddenly
        # be well approximated and isn't all that valuable after all. In the y == 0 case
        # we allow one extra change to replace the single dummy entry we put in aux_palette
        # above.
        while len(aux_palette) < aux_palette_size + (1 if y == 0 else max_aux_changes_per_line):
            error_by_pixel_rgb = aux_palette_error(hist, aux_palette)
            error_by_pixel_rgb = ((rgb, error) for rgb, error in error_by_pixel_rgb if error > 0)
            error_by_pixel_rgb = sorted(error_by_pixel_rgb, key=lambda x: x[1], reverse=True)
            if len(error_by_pixel_rgb) == 0:
                break
            aux_palette.add(error_by_pixel_rgb[0][0])

        # If aux_palette isn't full yet, add the first distinct colours we find in
        # the image. There's no advantage to leaving palette slots empty anyway, and the
        # code isn't written to handle this very well.
        if len(aux_palette) < aux_palette_size:
            for image_row in image_rgb[y:]:
                for pixel_rgb in image_row:
                    if pixel_rgb not in aux_palette:
                        aux_palette.add(pixel_rgb)
                        if len(aux_palette) == aux_palette_size:
                            break
        # TODO: This will go wrong if we have fewer than 16 distinct colours in the
        # source image. But let's handle that when it happens.
        assert len(aux_palette) >= aux_palette_size


        # Now remove colours from aux_palette one at a time until it's an acceptable size,
        # picking the one which causes the least overall error to remove each time. Again,
        # we don't do all n at once because removing one colour might make another colour
        # more important than it initially seemed.
        while len(aux_palette) > aux_palette_size:
            best_removal_candidate_rgb = None
            for removal_candidate_rgb in aux_palette:
                error_by_pixel_rgb = aux_palette_error(hist, aux_palette - set([removal_candidate_rgb]))
                error = sum(error for _, error in error_by_pixel_rgb)
                if best_removal_candidate_rgb is None or error < best_removal_candidate_error:
                    best_removal_candidate_rgb = removal_candidate_rgb
                    best_removal_candidate_error = error
            aux_palette.remove(best_removal_candidate_rgb)
        assert len(aux_palette) <= aux_palette_size
        aux_palette_set_by_y.append(aux_palette)
        #print "QQA", y, aux_palette

    # Now turn each palette into a list so each unchanged colour appears at the same
    # index each time; indices into these lists will be the auxiliary palette indices.
    aux_palette_by_y = [sorted(aux_palette_set_by_y[0])]
    aux_changes_by_y = [()]
    for y in range(1, y_size):
        aux_palette = []
        aux_changes = []
        #print "Y", y, aux_palette_set_by_y[y]
        for i, previous_rgb in enumerate(aux_palette_by_y[y-1]):
            #print len(aux_palette), aux_palette
            if previous_rgb in aux_palette_set_by_y[y]:
                rgb = previous_rgb
            else:
                # TODO: I don't know if it would make much difference in practice, but we could
                # potentially try to preserve position across "recent" auxiliary palettes
                # here. For example, if (5, 3, 2) was at index 14 a few lines ago, it
                # dropped out of the palette but now it is being added back in and it
                # happens that indices 9 and 14 are going to be replaced, we should
                # prefer to put (5, 3, 2) in at index 14 not index 9. The thinking here is
                # that this might increase the chances of the ULA palette already pairing
                # up (5, 3, 2) with "useful" colours without us having to burn a ULA change
                # on it.
                # TODO: *Or* maybe we should choose the "most similar" colour to the one currently
                # in the palette index we're updating, on the grounds this might allow the current
                # ULA palette to do a better job (thus saving changes) with the new aux palette.
                candidates = aux_palette_set_by_y[y] - aux_palette_set_by_y[y-1] - set(aux_palette)
                best_candidate = None
                for candidate_rgb in candidates:
                    error = distance(candidate_rgb, previous_rgb)
                    if best_candidate is None or error < best_candidate_error:
                        best_candidate = candidate_rgb
                        best_candidate_error = error
                #print "Y", y, "Q", len(candidates), candidates
                rgb = best_candidate
                #rgb = min(candidates)
                aux_changes.append((i, rgb))
            aux_palette.append(rgb)
        aux_palette_by_y.append(aux_palette)
        aux_changes_by_y.append(aux_changes)
        #print "A1", aux_palette_set_by_y[y]
        #print "A2", set(aux_palette_by_y[y])
        #print "A3", aux_palette_set_by_y[y] - set(aux_palette_by_y[y])
        #print "A4", set(aux_palette_by_y[y]) - aux_palette_set_by_y[y]
        assert aux_palette_set_by_y[y] == set(aux_palette_by_y[y])
        #print "QQB", y, aux_palette

    return aux_palette_by_y, aux_changes_by_y


def build_ula_palette_by_y():
    # If we're making less than max_aux_changes_per_line on a particular scan line, we
    # can steal two aux writes (i.e. one aux change) and use them for additional ULA
    # writes.
    max_ula_changes_by_y = [nominal_max_ula_changes_per_line] * y_size
    if max_aux_changes_per_line > 0:
        for y, aux_changes in enumerate(aux_changes_by_y):
            if len(aux_changes) < max_aux_changes_per_line:
                max_ula_changes_by_y[y] += writes_per_aux_change

    ula_palette_by_y = []
    ula_changes_by_y = []
    previous_ula_palette = None
    for y in range(y_size):
        ula_palette, ula_changes = build_ula_palette(y, max_ula_changes_by_y[y], previous_ula_palette)
        ula_palette_by_y.append(ula_palette)
        ula_changes_by_y.append(ula_changes)
        previous_ula_palette = ula_palette
        #print "UP", y, len(ula_changes), ula_changes, ula_palette
    return ula_palette_by_y, ula_changes_by_y


# We use this when we get to pick an arbitrary colour to add to a palette group; we used
# to take the one with the smallest index in aux_palette but that's obviously rather
# arbitrary. Some experimentation suggest this works fairly well; we pick the colour
# which is furthest away, on average, from the other colours in the palette group already.
# My possibly incorrect justification for why this works well is that it opens up the
# prospect of dithering by creating some variation in the palette group and reducing the
# chances of solid-but-liney pixel triplets ocurring.
def pick_colour(colour_set, palette_group, aux_palette):
    best_colour = None
    for colour in colour_set:
        colour_rgb = aux_palette[colour]
        error = 0
        for aux_index in palette_group:
            if aux_index is None:
                continue
            group_rgb = aux_palette[aux_index]
            error += distance(colour_rgb, group_rgb)
        if best_colour is None or error > best_error:
            best_colour = colour
            best_error = error
    colour_set.remove(best_colour)
    return best_colour


class UlaPalette(object):
    def __init__(self):
        self.palette = [set(), set(), set(), set()]
        self.pending_colours = set()
        assert self.valid()

    def valid(self):
        assert all(len(palette_group) <= 4 for palette_group in self.palette)
        assert all(len(self.pending_colours.intersection(palette_group)) == 0 for palette_group in self.palette)
        assert sum(len(palette_group) for palette_group in self.palette) + len(self.pending_colours) <= 16
        return True

    def space_left(self):
        assert self.valid()
        return 16 - (sum(len(palette_group) for palette_group in self.palette) + len(self.pending_colours))

    def palette_handles_colour_set(self, colour_set):
        return any(colour_set.issubset(palette_group) for palette_group in self.palette)

    def handles_colour_set(self, colour_set):
        return (self.palette_handles_colour_set(colour_set) or 
                (len(colour_set) == 1 and colour_set in self.pending_colours))

    def try_add_colour_set(self, colour_set, max_ula_changes, previous_ula_palette, aux_palette):
        assert self.valid()

        if self.handles_colour_set(colour_set):
            return True

        if len(colour_set) == 1:
            if self.space_left() > 0:
                self.pending_colours.add(elem(colour_set))
                if previous_ula_palette is None:
                    return True
                _, change_list = self.changes(previous_ula_palette, aux_palette)
                if len(change_list) <= max_ula_changes:
                    return True
                self.pending_colours.remove(elem(colour_set))
            return False

        best_palette_group = None
        space_left = self.space_left()
        for palette_group in self.palette:
            extra_colours = colour_set - palette_group
            if len(palette_group) + len(extra_colours) > 4 or len(extra_colours) > space_left:
                continue
            if previous_ula_palette is None:
                changes = 0
            else:
                old_pending_colours = self.pending_colours.copy()
                palette_group.update(extra_colours)
                self.pending_colours -= extra_colours
                _, change_list = self.changes(previous_ula_palette, aux_palette)
                changes = len(change_list)
                palette_group -= extra_colours
                self.pending_colours = old_pending_colours
                if changes > max_ula_changes:
                    continue
                #print "OKULA", changes, max_ula_changes
            new_palette_group_size = len(palette_group.union(colour_set))
            # TODO: lots of scope for tweaking weightings here
            # TODO: Maybe the change weight should be non-linear? As changes appraches max_changes we
            # weight it increasingly heavily? Not sure this is sensible but not tried it yet.
            score = -(changes*0.3 + len(extra_colours) + 0.1*new_palette_group_size)
            if best_palette_group is None or score > best_palette_group_score:
                best_palette_group = palette_group
                best_palette_group_score = score

        if best_palette_group is not None:
            best_palette_group.update(colour_set)
            self.pending_colours -= set.union(*self.palette)
        assert self.valid()
        return best_palette_group is not None

    def changes(self, previous_ula_palette, aux_palette):
        assert self.valid()
        #print "CE", previous_ula_palette
        assert isinstance(previous_ula_palette, list)
        assert all(isinstance(palette_group, list) for palette_group in previous_ula_palette)

        # We need to decide how to map our four palette groups to those in
        # previous_ula_palette. We consider all possible reorderings and take the one
        # which requires fewest changes.
        reorderings = list(itertools.permutations(range(len(self.palette))))
        best_rpg = None
        for reordering in reorderings:
            rpg = [self.palette[i] for i in reordering]
            flattened_palette, changes = UlaPalette.changes2(rpg, self.pending_colours, previous_ula_palette, aux_palette)
            if best_rpg is None or len(changes) < len(best_rpg_changes):
                best_rpg = rpg
                best_rpg_flattened_palette = flattened_palette
                best_rpg_changes = changes
                if len(best_rpg_changes) == 0:
                    break
        return best_rpg_flattened_palette, best_rpg_changes

    # TODO: Crappy name
    @staticmethod
    def changes2(palette, pending_colours, previous_ula_palette, aux_palette):
        pending_colours = pending_colours.copy()

        new_palette = []
        for new_palette_group_set, old_palette_group in zip(palette, previous_ula_palette):
            #print "Q", old_palette_group
            assert len(set(old_palette_group)) == len(old_palette_group)
            new_palette_group = []
            colours_to_keep = set(old_palette_group).intersection(new_palette_group_set)
            colours_to_add = new_palette_group_set - set(old_palette_group)
            #print "old/new", old_palette_group, new_palette_group_set
            #print "keep", colours_to_keep
            #print "add", colours_to_add
            assert len(colours_to_keep) + len(colours_to_add) <= 4
            while len(colours_to_keep) + len(colours_to_add) < 4:
                SFTODOA = set.intersection(set(old_palette_group), pending_colours)
                if len(SFTODOA) > 0:
                    colours_to_keep.add(pick_colour(SFTODOA, new_palette_group_set, aux_palette))
                    pending_colours -= colours_to_keep
                else:
                    break
            for old_colour in old_palette_group:
                if old_colour in colours_to_keep:
                    new_palette_group.append(old_colour)
                    colours_to_keep.remove(old_colour)
                elif len(colours_to_add) > 0:
                    new_palette_group.append(pick_colour(colours_to_add, new_palette_group_set, aux_palette))
                else:
                    new_palette_group.append(None)
            new_palette.append(new_palette_group)

        changes = []
        for i, (new_palette_group, old_palette_group) in enumerate(zip(new_palette, previous_ula_palette)):
            #print "Q", i, new_palette_group, old_palette_group
            while None in new_palette_group:
                pgi = new_palette_group.index(None)
                if len(pending_colours) > 0:
                    new_palette_group[pgi] = pick_colour(pending_colours, new_palette_group, aux_palette)
                else:
                    new_palette_group[pgi] = old_palette_group[pgi]

            for j, (new_colour, old_colour) in enumerate(zip(new_palette_group, old_palette_group)):
                if new_colour is not None and old_colour != new_colour:
                    changes.append((i * 4 + j, new_colour))
        assert len(pending_colours) == 0
        return list(itertools.chain(new_palette)), changes


def colour_set_hist_key(hist_entry):
    _, freq = hist_entry
    return -freq


def build_colour_set_hist(image_rows, aux_palette_list):
    hist = defaultdict(int)
    error_rgb = [[(0, 0, 0) for x in range(x_size)] for y in range(len(image_rows))]
    for y in range(len(image_rows)):
        aux_palette = aux_palette_list[y]
        quantised_row = []
        for x in range(0, x_size):
            adjusted_rgb = add_rgb(image_rows[y][x], error_rgb[y][x])
            #print "EC", y, x, error_rgb[y][x]
            aux_index = aux_palette_best_match_index(clamp_rgb(adjusted_rgb), aux_palette)
            quantised_row.append(aux_index)
            aux_rgb = aux_palette[aux_index]
            new_error = sub_rgb(adjusted_rgb, aux_rgb)
            for xoff, yoff, fraction in error_dist:
                if 0 <= x+xoff < x_size and y+yoff < len(image_rows):
                    error_rgb[y+yoff][x+xoff] = add_rgb(error_rgb[y+yoff][x+xoff], mult_rgb(new_error, fraction))
        for x in range(0, x_size, 3):
            t = frozenset(quantised_row[x:x+3])
            hist[t] += single_boost if len(t) == 1 else 1
    return sorted(hist.items(), key=colour_set_hist_key)
        


def build_ula_palette(y, max_ula_changes, previous_ula_palette):
    # TODO: This is another tweakable parameter and needs experimentation.
    ula_palette_window_size = 2
    hist = build_colour_set_hist(image_rgb[y:y+ula_palette_window_size], aux_palette_by_y[y:y+ula_palette_window_size])
    orig_sum_freq = sum(freq for _, freq in hist)
    while True:
        decomposed = False
        ula_palette = UlaPalette()
        new_hist = defaultdict(int)
        while len(hist) > 0:
            colour_set, freq = hist.pop(0)
            #print "P", ula_palette.palette, ula_palette.pending_colours
            #print "AX", colour_set, freq

            if ula_palette.try_add_colour_set(colour_set, max_ula_changes, previous_ula_palette, aux_palette_by_y[y]):
                new_hist[colour_set] += freq
                continue

            if len(colour_set) == 1:
                new_hist[colour_set] += freq
                continue

            decomposed = True
            t = tuple(colour_set) 
            hist = defaultdict(int, {colour_set: freq for colour_set, freq in hist})
            if len(colour_set) == 2:
                hist[frozenset([t[0]])] += single_boost * freq / 2.0
                hist[frozenset([t[1]])] += single_boost * freq / 2.0
                orig_sum_freq += (single_boost - 1) * freq
            else:
                assert len(colour_set) == 3
                hist[frozenset([t[0], t[1]])] += freq / 3.0
                hist[frozenset([t[0], t[2]])] += freq / 3.0
                hist[frozenset([t[1], t[2]])] += freq / 3.0
            hist = sorted(hist.items(), key=colour_set_hist_key)

        if not decomposed:
            break
        hist = sorted(new_hist.items(), key=colour_set_hist_key)
        sum_freq = sum(freq for _, freq in hist)
        #print "EE", sum_freq, orig_sum_freq
        # TODO: It's not a big deal, but we could avoid floating point if we scaled all
        # original frequencies by six (to allow for division by three and then a further
        # division by two).
        # assert abs(sum_freq - orig_sum_freq) < 0.1

    if previous_ula_palette is None:
        previous_ula_palette = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        flattened_palette, changes = ula_palette.changes(previous_ula_palette, aux_palette_by_y[y])
        changes = []
    else:
        flattened_palette, changes = ula_palette.changes(previous_ula_palette, aux_palette_by_y[y])
    #print "CD", y, changes, max_ula_changes, previous_ula_palette
    assert len(changes) <= max_ula_changes
    return flattened_palette, changes


def build_attribute_data():
    attribute_data = []
    for y in range(y_size):
        #print "XF", y
        aux_palette = aux_palette_by_y[y]
        ula_palette = ula_palette_by_y[y]
        attribute_row = []
        # We process the image in a serpentine fashion; most of the code ignores this, to
        # get this behaviour we do some reversal here and then fix it up later. We reverse
        # the following line too so that error distribution down into that line goes into
        # the correct place.
        if args.serpentine and y % 2 == 0:
            image_rgb[y].reverse()
            image_rgb[y+1].reverse()
        for x in range(0, x_size, 3):
            triple_attribute_data, error_dict = build_triple_attribute_data(image_rgb[y][x:x+3], aux_palette, ula_palette)
            assert triple_attribute_data[0] / 4 == triple_attribute_data[1] / 4
            assert triple_attribute_data[0] / 4 == triple_attribute_data[2] / 4
            attribute_row.extend(triple_attribute_data)
            for (xoff, yoff), error in error_dict.items():
                assert 0 <= yoff <= 1 # serpentine assumes this
                if 0 <= x+xoff < x_size and y+yoff < y_size:
                    image_rgb[y+yoff][x+xoff] = add_rgb(image_rgb[y+yoff][x+xoff], error)
        if args.serpentine and y % 2 == 0:
            image_rgb[y].reverse()
            image_rgb[y+1].reverse()
            attribute_row.reverse()
        attribute_data.append(attribute_row)
    return attribute_data


def build_triple_attribute_data(triple_rgb, aux_palette, ula_palette):
    best_triple_attribute_data = None
    for i, palette_group in enumerate(ula_palette):
        triple_attribute_data, error_dict, total_error = build_triple_attribute_data_for_group(triple_rgb, aux_palette, i * 4, palette_group)
        if best_triple_attribute_data is None or total_error < best_total_error:
            best_triple_attribute_data = triple_attribute_data
            best_error_dict = error_dict
            best_total_error = total_error
    return best_triple_attribute_data, best_error_dict


def build_triple_attribute_data_for_group(triple_rgb, aux_palette, palette_group_base, palette_group):
    triple_attribute_data = []
    total_error = (0, 0, 0)
    error_dict = defaultdict(lambda: (0, 0, 0))
    for x, pixel_rgb in enumerate(triple_rgb):
        best_aux_index = None
        for aux_index in palette_group:
            adjusted_rgb = add_rgb(pixel_rgb, error_dict[(x, 0)])
            error = distance(clamp_rgb(adjusted_rgb), aux_palette[aux_index])
            if best_aux_index is None or error < best_aux_index_error:
                best_aux_index = aux_index
                best_aux_index_error = error
        triple_attribute_data.append(palette_group_base + palette_group.index(best_aux_index))
        this_error = sub_rgb(adjusted_rgb, aux_palette[best_aux_index])
        total_error = add_rgb(total_error, this_error)
        for xoff, yoff, fraction in error_dist:
            error_dict[(x+xoff, yoff)] = add_rgb(error_dict[(x+xoff, yoff)], mult_rgb(this_error, fraction))
    total_error = distance((0, 0, 0), total_error)
    return triple_attribute_data, error_dict, total_error


def generate_proportions(label, d):
    info(label)
    hist = sorted(d.items(), key=lambda x: x[0])
    max_item_str_len = len(str(hist[-1][0]))
    total_freq = sum(x[1] for x in hist)
    for item, freq in hist:
        info("    %*d:%5.1f%%" % (max_item_str_len, item, 100 * freq / float(total_freq)))


def generate_percentiles(label, d):
    info(label)
    hist = sorted(d.items(), key=lambda x: x[1], reverse=True)
    hist_len = len(hist)
    hist_str_len = len(str(hist_len))
    hist.append((None, 0))
    sum_all_freq = sum(x[1] for x in hist)
    chunk = []
    total = 0
    old_percentile = 0
    for i, (item, freq) in enumerate(hist):
        chunk.append(item)
        total += freq
        new_percentile = 100 * total / sum_all_freq
        if (new_percentile // 10) > (old_percentile // 10):
            info("    %3d%%: %*d/%d %s" % (new_percentile, hist_str_len, i + 1, hist_len, chunk))
            chunk = []
        old_percentile = new_percentile



parser = argparse.ArgumentParser(description='Convert an image for display on a BBC Micro with VideoNuLA using per-scan-line palette programming.')
parser.add_argument('input', help='paletted 240x256 bitmap image to convert (ideally a PNG image)')
parser.add_argument('output', nargs='?', help='converted image suitable for use with showotf code (defaults to input filename but with .bbc extension)')
parser.add_argument('-s', '--simulated-output', nargs=1, help='generate a PNG image which simulates the effect of displaying the output image on real hardware')
parser.add_argument('-e', '--serpentine', action='store_true', help='use serpentine scanning when dithering the output')
parser.add_argument('-a', '--max-aux-changes', type=int, default=2, help='maximum number of auxiliary palette changes per scan line (default: 2)')

args = parser.parse_args()
if args.output is None:
    args.output = os.path.splitext(args.input)[0] + '.bbc'
# TODO: We should support max_aux_changes of 0 and 1, but these will require some tweaks to
# showotf.beebasm and this code to work properly and I don't want to get into that now.
# (showotf.beebasm needs to patch the palette updates correctly in all cases, and the code
# in here to handle "stealing" an unneeded aux update needs to behave correctly too.)
error_unless(2 <= args.max_aux_changes <= 4, "--max_aux_changes must be in the range 2-4")

# TODO: This calculation will need tweaking when we allow max_aux_changes_per_line to be 0
max_aux_changes_per_line = args.max_aux_changes
nominal_max_ula_changes_per_line = writes_per_line - max_aux_changes_per_line * writes_per_aux_change
assert nominal_max_ula_changes_per_line >= 1

source_image = PIL.Image.open(args.input)
x_size, y_size = source_image.size
# TODO: We might eventually want to expand this tool to handle mode 2 images (i.e. do
# per-scan-line colour changes and associated dithering, but without any attribute mode
# shenanigans).
# TODO: This tool should probably handle 16 colour source images, i.e. ones where there's
# no real desire to alter the aux palette at all. Ideally it would also handle <16 colour
# source images without breaking too.
error_unless(x_size == 240 and y_size == 256, 'Source image must be 240x256')
error_unless(source_image.mode == 'P', 'Source image must have a palette')
source_pixel_map = source_image.load()

source_palette = build_source_palette()
info('Source palette size: %d' % (len(source_palette),))
# image_rgb starts off as a representation of the source image but is mutated during
# conversion as errors are distributed.
image_rgb = build_image_rgb()
original_image_rgb = image_rgb[:]
source_palette = squash_palette(source_palette, image_rgb)

aux_palette_by_y, aux_changes_by_y = build_aux_palette_by_y()
aux_stats = defaultdict(int)
for y in range(y_size):
    aux_stats[len(aux_changes_by_y[y])] += 1
    #print "AP", y, len(aux_changes_by_y[y]), aux_changes_by_y[y], aux_palette_by_y[y]
generate_proportions("Auxiliary palette changes by percentage of lines:", aux_stats)

# TODO: This code is hacky, maybe keep a tidied up version around eventually (controlled by
# a command line option of course) moved into a helper function. We're just dumping the
# image out with no error diffusion using the best match for each pixel in that line's
# auxiliary palette. This a) acts as a kind of baseline to compare the output we get once
# we start taking attribute mode restrictions into account b) allows us to verify that the
# auxiliary palettes we've determined are reasonable and therefore the code to create them
# is probably working correctly.
output_image = PIL.Image.new('P', (x_size, y_size))
output_palette = []
union_aux_palette = set()
for aux_palette in aux_palette_by_y:
    union_aux_palette.update(set(aux_palette))
union_aux_palette = tuple(union_aux_palette)
#print "Q", list(itertools.chain(*union_aux_palette))
output_image.putpalette(list(x * 0x11 for x in itertools.chain(*union_aux_palette)))
output_pixel_map = output_image.load()
for y in range(y_size):
    for x in range(x_size):
        pixel_rgb = aux_palette_best_match(image_rgb[y][x], aux_palette_by_y[y])
        output_pixel_map[x, y] = union_aux_palette.index(pixel_rgb)
        #print x, y, output_pixel_map[x, y]
#output_image.show()
output_image = output_image.resize((1280, 1024), resample=PIL.Image.NEAREST)
output_image.save('znoatt.png')

ula_palette_by_y, ula_changes_by_y = build_ula_palette_by_y()
ula_stats = defaultdict(int)
for y in range(y_size):
    ula_stats[len(ula_changes_by_y[y])] += 1
generate_proportions("ULA palette changes by percentage of lines:", ula_stats)

attribute_data = build_attribute_data()

if args.simulated_output is not None:
    # TODO: This code is hacky, maybe keep a tidied up version around eventually (controlled by
    # a command line option of course) moved into a helper function. We're just dumping the
    # image out with no error diffusion using the best match for each pixel in that line's
    # auxiliary palette. This a) acts as a kind of baseline to compare the output we get once
    # we start taking attribute mode restrictions into account b) allows us to verify that the
    # auxiliary palettes we've determined are reasonable and therefore the code to create them
    # is probably working correctly.
    output_image = PIL.Image.new('P', (x_size, y_size))
    output_palette = []
    union_aux_palette = set()
    for aux_palette in aux_palette_by_y:
        union_aux_palette.update(set(aux_palette))
    union_aux_palette = tuple(union_aux_palette)
    #print "Q", list(itertools.chain(*union_aux_palette))
    output_image.putpalette(list(x * 0x11 for x in itertools.chain(*union_aux_palette)))
    output_pixel_map = output_image.load()
    colour_stats = defaultdict(int)
    cumulative_error = 0
    for y in range(y_size):
        aux_palette = aux_palette_by_y[y]
        ula_palette = ula_palette_by_y[y]
        for x, aux_index in enumerate(attribute_data[y]):
            pixel_rgb = aux_palette[ula_palette[aux_index / 4][aux_index % 4]]
            output_pixel_map[x, y] = union_aux_palette.index(pixel_rgb)
            colour_stats[pixel_rgb] += 1
            cumulative_error += distance(pixel_rgb, original_image_rgb[y][x])

            #print x, y, output_pixel_map[x, y]
    #output_image.show()
    # TODO: We should probably be showing this info() even if no simulated output is requested
    info("Mean error per pixel: %.2f" % (cumulative_error / float(x_size * y_size),))
    output_image = output_image.resize((1280, 1024), resample=PIL.Image.NEAREST)
    output_image.save(args.simulated_output[0])
    # TODO: As with info() above, this should always occur regardless of simulation
    generate_percentiles("Colours by cumulative percentage of pixels:", colour_stats)

# TODO: Move all the following into a function

initial_aux_palette_data = bytearray()
for aux_colour, (r, g, b) in enumerate(aux_palette_by_y[0]):
    initial_aux_palette_data += bytearray([(g << 4) | b, (aux_colour << 4) | r])
assert len(initial_aux_palette_data) == 32

initial_ula_palette_data = bytearray()
for i, palette_group in enumerate(ula_palette_by_y[0]):
    for j, aux_colour in enumerate(palette_group):
        ula_colour = i * 4 + j
        initial_ula_palette_data += chr((ula_colour << 4) | (aux_colour ^ 7))
assert len(initial_ula_palette_data) == 16

# We don't use the Y=0 data in {ula,aux}_writes_by_y on the BBC, but we need something
# sane here temporarily so the "copy data from previous line if necessary" code in the
# following loop has something to work with. We need to reverse the aux data because
# the initial write is done backwards but here the writing is done forwards.
ula_writes_by_y = [initial_ula_palette_data[:nominal_max_ula_changes_per_line]]
aux_writes_by_y = [bytearray(reversed(initial_aux_palette_data[:max_aux_changes_per_line * writes_per_aux_change]))]
for y in range(1, y_size):
    ula_writes = bytearray()
    for ula_colour, aux_colour in ula_changes_by_y[y]:
        ula_writes += chr((ula_colour << 4) | (aux_colour ^ 7))
    aux_writes = bytearray()
    for aux_colour, (r, g, b) in aux_changes_by_y[y]:
        aux_writes += bytearray([(aux_colour << 4) | r, (g << 4) | b])

    dynamic_ula = len(ula_changes_by_y[y]) > nominal_max_ula_changes_per_line
    needed_aux_writes = (max_aux_changes_per_line - (1 if dynamic_ula else 0)) * writes_per_aux_change
    needed_ula_writes = nominal_max_ula_changes_per_line + (writes_per_aux_change if dynamic_ula else 0)
    assert needed_aux_writes + needed_ula_writes == writes_per_line

    while len(ula_writes) < needed_ula_writes:
        if len(ula_writes) == 0:
            ula_writes += chr(ula_writes_by_y[y-1][0])
        else:
            ula_writes += chr(ula_writes[-1])
    assert len(ula_writes) == needed_ula_writes
    ula_writes_by_y.append(ula_writes)

    while len(aux_writes) < needed_aux_writes:
        if len(aux_writes) == 0:
            aux_writes += aux_writes_by_y[y-1][0:writes_per_aux_change]
        else:
            aux_writes += aux_writes[-writes_per_aux_change:]
    assert len(aux_writes) == needed_aux_writes
    aux_writes_by_y.append(aux_writes)

misc_data = bytearray([0] * (writes_per_line + 1))
misc_data[0] = chr(nominal_max_ula_changes_per_line)
mixed_writes_by_y = [misc_data]
aux_palette_addr = 0xfe23
ula_palette_addr = 0xfe21
for y in range(1, y_size):
    mixed_writes = aux_writes_by_y[y] + ula_writes_by_y[y]
    dynamic_ula = len(ula_writes_by_y[y]) > nominal_max_ula_changes_per_line
    mixed_writes += chr(ula_palette_addr & 0xff) if dynamic_ula else chr(aux_palette_addr & 0xff)
    assert len(mixed_writes) == writes_per_line + 1
    mixed_writes_by_y.append(mixed_writes)
with open('z.dat', 'wb') as f:
    for y in range(0, y_size):
        f.write(mixed_writes_by_y[y])

interleaved_writes = bytearray()
for i in range(writes_per_line + 1):
    for y in range(y_size):
        interleaved_writes += chr(mixed_writes_by_y[y][i])
assert len(interleaved_writes) == y_size * (writes_per_line + 1)

video_data = bytearray()
for y_block in range(0, y_size, 8):
    for x in range(0, x_size, 3):
        for y in range(y_block, y_block + 8):
            t = attribute_data[y][x:x+3]
            attribute_set = t[0] / 4
            bbc_pixels = (t[0] % 4, t[1] % 4, t[2] % 4)
            def adjust_bbc_pixel(n):
                assert 0 <= n <= 3
                return ((n & 2) << 3) | (n & 1)
            video_data += chr((adjust_bbc_pixel(bbc_pixels[0]) << 3) |
                              (adjust_bbc_pixel(bbc_pixels[1]) << 2) |
                              (adjust_bbc_pixel(bbc_pixels[2]) << 1) |
                               adjust_bbc_pixel(attribute_set))
assert len(video_data) == y_size * x_size / 3

with open(args.output, "wb") as bbc_image:
    bbc_image.write(initial_ula_palette_data)
    bbc_image.write(initial_aux_palette_data)
    bbc_image.write(interleaved_writes)
    bbc_image.write(video_data)





#hardware_writes_by_y = []
#for y in range(y_size):
#    hardware_writes = []
#    for 





# TODO: *If* we used window sizes of 1 for aux and ULA palettes, it might be possible/worthwhile
# to calculate the aux and ULA palettes as we go. That is, 1) aux palette for line 0 2) ULA palette for line 0 3) pixel data for line 0 (which will distribute some error into line 1) 4) aux palette for line 1 (thus taking account of distributed error) 5) ULA palette for line 1 etc. As opposed to the current approach where we do aux palette for all lines, then ULA palette for all lines, then pixel data for all lines. This still isn't perfect (even ignoring the downside of using a window size of 1 generally) as error distribution occurs "rightwards" within a line and that "could/should" interact with the optimal ULA palette choice, but won't.

# TODO: Just *maybe* we should a) do error distribution and ULA palette generation mixed in together b) use alternating direction X scanning to compensate for any bias that would give towards one side of the screen
