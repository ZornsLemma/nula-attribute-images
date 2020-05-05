# nula-attribute-images

This is some experimental code to generate images for display on the BBC Micro using VideoNuLA's attribute modes. At the moment only the attribute variation on mode 1 is supported, i.e. a 240x256 colour mode with 16 colours split into four groups of four colours, with the restriction that each pixel triple on the screen can use any three colours from a single colour group. No on-the-fly palette reprogramming is being done so there's a single palette for the entire image.

Some sample images are included and there's a crude shell script (make.sh) to build a slideshow disc image with some of them. The disc image is not checked into the repository so you'll either need to build it yourself or get it from the stardot thread where I'll upload it.

I've only been able to test this on b-em as I don't have any real hardware, but it will probably work. :-) The images should look slightly better on real hardware as b-em's attribute mode pixels are unevenly sized; it needs to fit six pixels in the space where eight normally go and it can't use fractionally sized pixels.

The algorithm used is described in a big comment inside mode1attr.py (search for "We need to build up") so I won't repeat it here.

If you want to try this on your own images, you'll need python and PIL (Python Imaging Library) installed as well as some way to get the output files onto a disc image for use in an emulator or on real hardware. I'm using Ubuntu, but since this is Python code you should be able to run it on pretty much any operating system, although the little utility shell scripts in the repository which I use to build the slideshow probably won't work on anything except Unix systems.

If you can type "python" at a command prompt on your machine and do "import PIL.Image" at the Python prompt without an error occurring, you're probably fine. If not you'll need to install the relevant software; on Ubuntu it's probably enough to do "apt install python-pil".

Given all that, the process for converting your own images is roughly as follow. I'll describe the process using gimp, but you can use whatever works best for you.
1. Take your image and (if necessary) crop it (or part of it) to a 4:3 aspect ratio to match the BBC screen.
1. Optionally tweak the image to increase the saturation; I find this sometimes gives better results. There's probably scope for experimentation here.
1. Scale the image (using interpolation) to 240x256; this will distort the image, but the distortion will be undone when it's displayed because attribute mode 1 pixels are (approximately) 33% wider than they are high. In gimp, choose Image->Scale Image, unlock the relationship between width and height, enter 240 and 256 respectively and choose something other than "None" for interpolation.
1. Convert the image to 16 colours, probably with dithering (though it might depend on the image you're using). In gimp, choose Image->Mode->Indexed, specify "Generate optimum palette" with 16 colours and choose a dithering option.
1. Optionally tweak the image further; if the dithering introduces some noise in an area of flat colour you might want to paint over that with the flat colour, for example.
1. Save the image as a .png file. Let's pretend you call it foo.png.
1. Execute the command "python mode1attr.py foo.png foo.bbc" to convert the image into BBC format. (This is very slow at the moment.)
1. As well as foo.bbc, a file called z.png will be generated which is (approximately) the same as foo.bbc but in png format. You can examine this to see if the conversion process worked well or not without needing to get foo.bbc onto a disc image and use an emulator or real hardware to view it. It's a 240x256 image so you'll need a viewer which can scale it up to undo the distortion; I find opening it in gimp and scaling to 1024x768 with no interpolation works well.
1. You need to get the resulting foo.bbc file onto a disc image, along with a tokenised version of the slideshow.bas display program. The simplest way to do this is to take the existing slideshow.ssd disc image and replace one of the existing image files (eg JAFFA, which is the first image displayed) with yours, keeping the name the same. You can of course use your own filenames and tweak the list of names in the DATA statements at the end of the BASIC program accordingly.
