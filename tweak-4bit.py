import PIL.Image
import sys


def error_unless(b, message):
    if not b:
        sys.stderr.write(message + '\n')
        sys.exit(1)


if len(sys.argv) != 3:
    sys.stderr.write('Usage: %s INFILE OUTFILE [OUTFILESIM]\n' % sys.argv[0])
    sys.exit(1)

source_image = PIL.Image.open(sys.argv[1])
error_unless(source_image.mode == 'P', 'Source image must have a palette')
output_image = source_image.resize((1280, 1024), resample=PIL.Image.NEAREST)
palette = output_image.getpalette()
output_image.putpalette(list((x >> 4) * 0x11 for x in palette))
output_image.save(sys.argv[2])
