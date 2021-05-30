cmake_minimum_required(VERSION 3.16)

project(SurroundView LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

enable_language(CUDA)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-9.0/bin/nvcc)
set(
         CMAKE_NVCC_FLAGS
         ${CMAKE_NVCC_FLAGS};
         -Xcompiler
         -fopenmp
         #-O3
         -use_fast_math
         -gencode arch=compute_62,code=sm_62
)
set(CMAKE_CUDA_FLAGS)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 14)


find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})


set(TEST_INCLUDE_DIRS)
set(TEST_LIBRARIES)

#set(Qt5Modules Core Widgets Gui)
#find_package(Qt5 COMPONENTS ${Qt5Modules} REQUIRED)

#set(CMAKE_AUTOMOC ON)

set(GLM_INCLUDE_DIR "/usr/include/glm")
set(3D_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/3dparty/")
set(GLFW_LIB "/usr/lib/aarch64-linux-gnu/libglfw.so.3.1")
#set(GLEW_LIB "/usr/lib/aarch64-linux-gnu/libGLEW.so")
set(GLES_LIB "/usr/lib/aarch64-linux-gnu/libEGL.so")
list(APPEND GLES_LIB "/usr/lib/aarch64-linux-gnu/libGLESv2.so")


find_package(OpenCV REQUIRED)
list(APPEND TEST_LIBRARIES ${OpenCV_LIBS})
list(APPEND TEST_LIBRARIES ${CUDA_LIBRARIES})
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND TEST_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS})
find_package(OpenGL REQUIRED)
list(APPEND TEST_INCLUDE_DIRS ${OPENGL_EGL_INCLUDE_DIR})

set(GLM_INCLUDE_DIR "/usr/include/glm/")
list(APPEND TEST_INCLUDE_DIRS ${GLM_INCLUDE_DIR})
list(APPEND TEST_INCLUDE_DIRS ${3D_INCLUDE_DIR})
list(APPEND TEST_LIBRARIES ${GLFW_LIB})
list(APPEND TEST_LIBRARIES ${GLES_LIB})

find_package(ASSIMP REQUIRED)
list(APPEND TEST_LIBRARIES ${ASSIMP_LIBRARIES})
list(APPEND TEST_INCLUDE_DIRS ${ASSIMP_INCLUDE_DIRS})


include_directories(${PROJECT_SOURCE_DIR}/include)


file(GLOB SOURCE_FILES 
	${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cusrc/*.cu
)


add_executable(SurroundView ${SOURCE_FILES})

set_target_properties(SurroundView 
			PROPERTIES OUTPUT_NAME "SurroundView" 
			CUDA_SEPARABLE_COMPILATION ON
)
set_property(TARGET SurroundView PROPERTY CUDA_STANDARD 14)

target_link_libraries(SurroundView PUBLIC ${TEST_LIBRARIES})
target_include_directories(SurroundView PUBLIC ${TEST_INCLUDE_DIRS})

target_compile_options(SurroundView PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_NVCC_FLAGS}>)
