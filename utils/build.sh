#!/bin/sh
set -e
DARKNET=../darknet
NCNN=../ncnn/install

gcc -O2 -Wall -Wextra -g -I$DARKNET/include -c \
  pipeline.c capture.c test.c decode.c preprocess.c detect_darknet.c
g++ -O2 -Wall -Wextra -g -I$NCNN/include/ncnn -c recognize.cpp

COMMON="decode.o preprocess.o"

g++ pipeline.o $COMMON detect_darknet.o recognize.o \
  $DARKNET/libdarknet.a $NCNN/lib/libncnn.a -o pipeline -fopenmp -lm -lpthread

gcc capture.o $COMMON -o capture -lm
# gcc prep_image.o $COMMON -o prep_image -lm
gcc test.o $COMMON -o test -lm
