cmake_minimum_required(VERSION 3.16)
project(PD_FSL_Camera_Handler)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================
# Find dependencies
# ============================================

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core videoio imgproc imgcodecs)
message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Mediapipe (adjust paths based on your installation)
# For pre-built Mediapipe or custom build
set(MEDIAPIPE_DIR "" CACHE PATH "Path to Mediapipe installation")

if(NOT MEDIAPIPE_DIR)
    # Try to find in common locations
    if(EXISTS "/usr/local/mediapipe")
        set(MEDIAPIPE_DIR "/usr/local/mediapipe")
    elseif(EXISTS "/opt/mediapipe")
        set(MEDIAPIPE_DIR "/opt/mediapipe")
    else()
        message(FATAL_ERROR "MEDIAPIPE_DIR not set and not found in common locations. "
                           "Set MEDIAPIPE_DIR=/path/to/mediapipe when running cmake")
    endif()
endif()

set(MEDIAPIPE_INCLUDE_DIR "${MEDIAPIPE_DIR}/include")
set(MEDIAPIPE_LIB_DIR "${MEDIAPIPE_DIR}/lib")

message(STATUS "Mediapipe directory: ${MEDIAPIPE_DIR}")
message(STATUS "Mediapipe include: ${MEDIAPIPE_INCLUDE_DIR}")
message(STATUS "Mediapipe lib dir: ${MEDIAPIPE_LIB_DIR}")

# ============================================
# Protobuf (required by Mediapipe)
# ============================================
find_package(Protobuf REQUIRED)
message(STATUS "Protobuf: ${PROTOBUF_INCLUDE_DIR}")

# ============================================
# Create shared library
# ============================================
add_library(camera_handler SHARED camera_handler.cpp)

# Include directories
target_include_directories(camera_handler PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${MEDIAPIPE_INCLUDE_DIR}
    ${PROTOBUF_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(camera_handler PRIVATE
    ${OpenCV_LIBS}
    ${MEDIAPIPE_LIB_DIR}/libmediapipe_framework.so
    ${MEDIAPIPE_LIB_DIR}/libmediapipe_graphs_holistic_holistic_landmark_cpu.so
    ${PROTOBUF_LIBRARIES}
)

# ============================================
# Create Python-callable shared library
# ============================================
# Optional: Build as Python extension
if(BUILD_PYTHON_MODULE)
    add_library(camera_handler_py SHARED camera_handler.cpp)
    target_include_directories(camera_handler_py PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${MEDIAPIPE_INCLUDE_DIR}
        ${PROTOBUF_INCLUDE_DIR}
    )
    target_link_libraries(camera_handler_py PRIVATE
        ${OpenCV_LIBS}
        ${MEDIAPIPE_LIB_DIR}/libmediapipe_framework.so
        ${MEDIAPIPE_LIB_DIR}/libmediapipe_graphs_holistic_holistic_landmark_cpu.so
        ${PROTOBUF_LIBRARIES}
    )
    set_target_properties(camera_handler_py PROPERTIES PREFIX "")
    if(APPLE)
        set_target_properties(camera_handler_py PROPERTIES SUFFIX ".so")
    endif()
endif()

# ============================================
# Installation
# ============================================
install(TARGETS camera_handler
        LIBRARY DESTINATION lib)

message(STATUS "Build configuration complete!")
message(STATUS "To build:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  cmake --build .")
