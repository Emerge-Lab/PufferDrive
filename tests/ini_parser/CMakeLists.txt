cmake_minimum_required(VERSION 3.10)
project(ini_tester)

add_library(inih ${CMAKE_CURRENT_LIST_DIR}/../../inih-r62/ini.c)
target_include_directories(inih PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../inih-r62)

enable_testing()
add_executable(ini_tester ini_tester.c)
target_link_libraries(ini_tester PRIVATE inih)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ini_tester PRIVATE -fsanitize=address)
    target_link_options(ini_tester PRIVATE -fsanitize=address)
endif()
set_property(TARGET ini_tester PROPERTY COMPILE_WARNING_AS_ERROR ON)

function(AddTest testName)
  add_test(
    NAME ${testName}
    COMMAND ini_tester ${testName}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endfunction()

AddTest(test_values)
AddTest(test_full_line_comment)
AddTest(test_inline_comment)
AddTest(test_problematic_inline_comment)
