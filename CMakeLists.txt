CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(catboost_runtime CXX)

# nlohmann/json uses C++14. Later it is possible we switch to C++11
SET(CMAKE_CXX_STANDARD 14)

SET(CMAKE_VERBOSE_MAKEFILE ON)
# Following line creates compile commands for YouCompleteMe
SET(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# We want to be able to link this project to shared libraries
SET(CMAKE_POSITION_INDEPENDENT_CODE ON)

OPTION(ENABLE_PERF "Enable performance tests" OFF)
OPTION(DISABLE_SSE "Disable SSE in build" OFF)

# Set Debug build by default:
IF(NOT CMAKE_BUILD_TYPE)
	SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: Debug Release." FORCE)
ENDIF()

# Add all possible warnings if keys are supported:
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-Wall -W -Wextra" C_FLAG_WARN)
IF(C_FLAG_WARN)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wextra")
ENDIF()

IF(DISABLE_SSE)

    # user does not want SSE
    ADD_DEFINITIONS("-DNOSSE")

ELSE()
    # Check that compiler supports SSE
    # TODO: this trick wont work for Windows, but still I have no chance to build Win version.
    CHECK_CXX_COMPILER_FLAG("-msse4.1" C_SSE_FLAG)
    IF(C_SSE_FLAG)
        ADD_DEFINITIONS("-msse4.1")
    ELSE()
        ADD_DEFINITIONS("-DNOSSE")
    ENDIF()
ENDIF()

# We want to have tests
ENABLE_TESTING()

INCLUDE_DIRECTORIES(include)

ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(unittest)
ADD_SUBDIRECTORY(perftest)
