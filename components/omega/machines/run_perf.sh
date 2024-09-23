#!/usr/bin/bash

arch=cpu
comp=gcc
opt="vel_sep"

dirname=frontier-$arch-$comp
mkdir $dirname

ln -sf ~/omega_test_meshes/icos-cvt-hi/icos1920.nc PerfMesh.nc
./test/testSteadyZonal.exe  > $dirname/log_${arch}_1920_${opt}.txt 2>&1
ln -sf ~/omega_test_meshes/icos-cvt-hi/icos960.nc PerfMesh.nc
./test/testSteadyZonal.exe  > $dirname/log_${arch}_960_${opt}.txt 2>&1
ln -sf ~/omega_test_meshes/icos-cvt-hi/icos480.nc PerfMesh.nc
./test/testSteadyZonal.exe  > $dirname/log_${arch}_480_${opt}.txt 2>&1
ln -sf ~/omega_test_meshes/icos-cvt-hi/icos240.nc PerfMesh.nc
./test/testSteadyZonal.exe  > $dirname/log_${arch}_240_${opt}.txt 2>&1
