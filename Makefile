# GSLROOT=/opt/apps/gsl/1.9
# GSLROOT=/usr
# use this if on 64-bit machine with 64-bit GSL libraries
# ARCH=x86_64
# use this if on 32-bit machine with 32-bit GSL libraries
# ARCH=i386
MPICC=mpicc
ifndef MATLABHOME
MATLABHOME=/opt/MATLAB/R2017b
endif
LD_LIBRARY_PATH=$(MATLABHOME)/bin/glnxa64

MEX=$(MATLABHOME)/bin/glnxa64/mex
CC=gcc
MEXFLAGS=-fPIC -shared -Wl,--no-undefined -Wl,-rpath-link,$(MATLABHOME)/bin/glnxa64 -L$(MATLABHOME)/bin/glnxa64 -lmx -lmex -lmat -lm
CFLAGS=-Wall -ggdb -O0 -g3 -std=c99 -fopenmp -I/usr/include -I$(MATLABHOME)/extern/include
LDFLAGS=-L/usr/lib/x86_64-linux-gnu -L/usr/lib -lgsl -lgslcblas -lm -lmatio -fopenmp

all: pFistaLasso pFistaLassoLib

pFistaLasso: pFistaLasso.o mmio.o
	$(CC) $(CFLAGS) pFistaLasso.o mmio.o -o pFistaLasso $(LDFLAGS)
pFistaLassoLib:
	$(MEX) pFistaLassoLib.c -output pFistaLasso.mexa64 CFLAGS="$(CFLAGS)" LINKLIBS="$(MEXFLAGS) $(LDFLAGS)" 
	
# gam: gam.o mmio.o
#	$(MPICC) $(CFLAGS) $(LDFLAGS) gam.o mmio.o -o gam

# gam.o: gam.c mmio.o
#	$(MPICC) $(CFLAGS) -c gam.c

mmio.o: mmio.c
	$(CC) $(CFLAGS) -c mmio.c

clean:
	rm -vf *.o pFistaLasso pFistaLasso.mexa64 
run:
	./pFistaLasso ./data 

