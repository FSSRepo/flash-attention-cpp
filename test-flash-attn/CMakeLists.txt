cmake_minimum_required(VERSION 3.18)

project(TestApp)

if (WIN32)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  set(CUDA_FLAGS -use_fast_math)
  set(CUDA_FILES main.cu)
  set(CUDA_LIBRARIES CUDA::cudart_static)

  set(CMAKE_CUDA_ARCHITECTURES "70")

  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>")
endif()

message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

if(WIN32)
  link_directories(../build/Release)
else()
  link_directories(../build)
endif()

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ..
    ../fa
    ../cutlass/include
)

add_executable(test-app main.cpp ${CUDA_FILES})

target_link_libraries(test-app PUBLIC ${CUDA_LIBRARIES} flash_attn)

if (WIN32)
  add_custom_command(TARGET test-app POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/../build/Release/flash_attn.dll $<TARGET_FILE_DIR:test-app>)
else()
  add_custom_command(TARGET test-app POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/../build/libflash_attn.so $<TARGET_FILE_DIR:test-app>)
endif()