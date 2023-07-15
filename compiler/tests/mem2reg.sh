#!/usr/bin/bash

SRC_FILE=$1

opt-15 -mem2reg -S $SRC_FILE.ll -o $SRC_FILE-mem2reg.ll


