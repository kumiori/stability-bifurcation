#!/bin/bash
docker run --rm -ti -v $(pwd):/home/fenics/shared -w /home/fenics/shared quay.io/83252/varfrac ". sourceme.sh; /bin/bash -i"