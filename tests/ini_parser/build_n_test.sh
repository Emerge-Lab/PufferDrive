#!/bin/bash
# Configure, build and run the tests
cmake -S tests/ini_parser -B tests/ini_parser/build
cmake --build tests/ini_parser/build
ctest --test-dir tests/ini_parser/build --output-on-failure
