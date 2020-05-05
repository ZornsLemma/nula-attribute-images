#!/bin/bash
./convert.sh
beebasm -i slideshow.beebasm -do slideshow.ssd -opt 3
