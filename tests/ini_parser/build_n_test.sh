#!/bin/bash
# Configure, build and run the tests
BUILD_TYPE=Release
cmake -S tests/ini_parser -B tests/ini_parser/build -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
cmake --build tests/ini_parser/build
ctest --test-dir tests/ini_parser/build --output-on-failure
