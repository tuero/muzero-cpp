#!/bin/bash
cd ../build
ctest -j$(nproc) --output-on-failure
