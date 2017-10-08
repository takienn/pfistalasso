# GSLROOT=/opt/apps/gsl/1.9
# GSLROOT=/usr
# use this if on 64-bit machine with 64-bit GSL libraries
# ARCH=x86_64
# use this if on 32-bit machine with 32-bit GSL libraries
# ARCH=i386
MPICC=mpicc
CC=gcc
CFLAGS=-Wall -ggdb -O0 -g3 -std=c99 -I/usr/include
LDFLAGS=-L/usr/lib -lgsl -lgslcblas -lm

all: pFistaLasso pFistaLasso_wen

pFistaLasso_wen: pFistaLasso_wen.o mmio.o
	$(MPICC) $(CFLAGS) pFistaLasso_wen.o mmio.o -o pFistaLasso_wen $(LDFLAGS)

pFistaLasso: pFistaLasso.o mmio.o
	$(MPICC) $(CFLAGS) pFistaLasso.o mmio.o -o pFistaLasso $(LDFLAGS)

# gam: gam.o mmio.o
#	$(MPICC) $(CFLAGS) $(LDFLAGS) gam.o mmio.o -o gam

pFistaLasso_wen.o: pFistaLasso_wen.c mmio.o
	$(MPICC) $(CFLAGS)  -c pFistaLasso_wen.c

# gam.o: gam.c mmio.o
#	$(MPICC) $(CFLAGS) -c gam.c

mmio.o: mmio.c
	$(CC) $(CFLAGS) -c mmio.c

clean:
	rm -vf *.o pFistaLasso pFistaLasso_wen
run:
	mpiexec -n 1 ./pFistaLasso big1

