#!/bin/bash
./convertc.sh
beebasm -i otfc.beebasm -do otfc.ssd -opt 3
