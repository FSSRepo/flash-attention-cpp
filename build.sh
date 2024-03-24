#!/bin/bash
mkdir build
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout 751eb9a
cd ../build
cmake ..
cmake --build . --config Release