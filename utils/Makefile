# Build: make            -> capture, pipeline, prep_image
#        make capture    -> just the live binary
#        make clean

DARKNET = ../darknet
NCNN    = ../ncnn/install

CFLAGS   = -O2 -Wall -Wextra -g -I$(DARKNET)/include
CXXFLAGS = -O2 -Wall -Wextra -g -I$(NCNN)/include/ncnn
LIBS     = $(DARKNET)/libdarknet.a $(NCNN)/lib/libncnn.a -fopenmp -lm -lpthread

COMMON = decode.o preprocess.o detect_darknet.o recognize.o

all: capture pipeline prep_image

capture: capture.o $(COMMON)
	g++ $^ $(LIBS) -o $@

pipeline: pipeline.o $(COMMON)
	g++ $^ $(LIBS) -o $@

# prep_image needs neither darknet nor ncnn -- stb_image.h lives in darknet/src
prep_image: prep_image.o preprocess.o
	gcc $^ -lm -o $@

prep_image.o: prep_image.c
	gcc -O2 -Wall -Wextra -g -I$(DARKNET)/src -c $< -o $@

%.o: %.c
	gcc $(CFLAGS) -c $< -o $@

%.o: %.cpp
	g++ $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o capture pipeline prep_image

.PHONY: all clean
