project(Eigen)

cmake_minimum_required(VERSION 2.8.2)

# guard against in-source builds

if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif()

# guard against bad build-type strings

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(    NOT cmake_build_type_tolower STREQUAL "debug"
   AND NOT cmake_build_type_tolower STREQUAL "release"
   AND NOT cmake_build_type_tolower STREQUAL "relwithdebinfo")
  message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values are Debug, Release, RelWithDebInfo (case-insensitive).")
endif()


#############################################################################
# retrieve version information                                               #
#############################################################################

# automatically parse the version number
file(READ "${PROJECT_SOURCE_DIR}/Eigen/src/Core/util/Macros.h" _eigen_version_header)
string(REGEX MATCH "define[ \t]+EIGEN_WORLD_VERSION[ \t]+([0-9]+)" _eigen_world_version_match "${_eigen_version_header}")
set(EIGEN_WORLD_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "define[ \t]+EIGEN_MAJOR_VERSION[ \t]+([0-9]+)" _eigen_major_version_match "${_eigen_version_header}")
set(EIGEN_MAJOR_VERSION "${CMAKE_MATCH_1}")
string(REGEX MATCH "define[ \t]+EIGEN_MINOR_VERSION[ \t]+([0-9]+)" _eigen_minor_version_match "${_eigen_version_header}")
set(EIGEN_MINOR_VERSION "${CMAKE_MATCH_1}")
set(EIGEN_VERSION_NUMBER ${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION})

# if the mercurial program is absent, this will leave the EIGEN_HG_CHANGESET string empty,
# but won't stop CMake.
execute_process(COMMAND hg tip -R ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE EIGEN_HGTIP_OUTPUT)
execute_process(COMMAND hg branch -R ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE EIGEN_BRANCH_OUTPUT)

# if this is the default (aka development) branch, extract the mercurial changeset number from the hg tip output...
if(EIGEN_BRANCH_OUTPUT MATCHES "default")
string(REGEX MATCH "^changeset: *[0-9]*:([0-9;a-f]+).*" EIGEN_HG_CHANGESET_MATCH "${EIGEN_HGTIP_OUTPUT}")
set(EIGEN_HG_CHANGESET "${CMAKE_MATCH_1}")
endif(EIGEN_BRANCH_OUTPUT MATCHES "default")
#...and show it next to the version number
if(EIGEN_HG_CHANGESET)
  set(EIGEN_VERSION "${EIGEN_VERSION_NUMBER} (mercurial changeset ${EIGEN_HG_CHANGESET})")
else(EIGEN_HG_CHANGESET)
  set(EIGEN_VERSION "${EIGEN_VERSION_NUMBER}")
endif(EIGEN_HG_CHANGESET)

# the user modifiable install path for header files
set(EIGEN_INCLUDE_INSTALL_DIR ${EIGEN_INCLUDE_INSTALL_DIR} CACHE PATH "The directory where we install the header files (optional)")

# set the internal install path for header files which depends on whether the user modifiable
# EIGEN_INCLUDE_INSTALL_DIR has been set by the user or not.
if(EIGEN_INCLUDE_INSTALL_DIR)
  set(INCLUDE_INSTALL_DIR
    ${EIGEN_INCLUDE_INSTALL_DIR}
    CACHE INTERNAL
    "The directory where we install the header files (internal)"
  )
else()
  set(INCLUDE_INSTALL_DIR
    "${CMAKE_INSTALL_PREFIX}/include/eigen3"
    CACHE INTERNAL
    "The directory where we install the header files (internal)"
  )
endif()

SET(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_subdirectory(Eigen)

add_subdirectory(unsupported)

message(STATUS "")
message(STATUS "Configured Eigen ${EIGEN_VERSION_NUMBER}")
message(STATUS "")

FIND_PATH(EIGEN_DIR "Eigen/Core"
    HINTS "${EIGEN_ROOT}" "$ENV{EIGEN_ROOT}" "${EIGEN_INCLUDE_DIR_HINTS}"
    PATHS "$ENV{PROGRAMFILES}/Eigen" "$ENV{PROGRAMW6432}/Eigen" "/usr" "/usr/local"
    PATH_SUFFIXES eigen3 include/eigen3 include
    DOC "Root directory of EIGEN library")
if(EXISTS "${EIGEN_DIR}" AND NOT "${EIGEN_DIR}" STREQUAL "")
	SET(EIGEN_FOUND TRUE)
	SET(EIGEN_INCLUDE_DIRS ${EIGEN_DIR})
	SET(EIGEN_DIR "${EIGEN_DIR}" CACHE PATH "" FORCE)
	MARK_AS_ADVANCED(EIGEN_DIR)

	# Extract Eigen version from Eigen/src/Core/util/Macros.h
	SET(EIGEN_VERSION_FILE ${EIGEN_INCLUDE_DIRS}/Eigen/src/Core/util/Macros.h)
	IF (NOT EXISTS ${EIGEN_VERSION_FILE})
		EIGEN_REPORT_NOT_FOUND(
		  "Could not find file: ${EIGEN_VERSION_FILE} "
		  "containing version information in Eigen install located at: "
		  "${EIGEN_INCLUDE_DIRS}.")
	ELSE (NOT EXISTS ${EIGEN_VERSION_FILE})
		FILE(READ ${EIGEN_VERSION_FILE} EIGEN_VERSION_FILE_CONTENTS)

		STRING(REGEX MATCH "#define EIGEN_WORLD_VERSION [0-9]+"
		  EIGEN_WORLD_VERSION "${EIGEN_VERSION_FILE_CONTENTS}")
		STRING(REGEX REPLACE "#define EIGEN_WORLD_VERSION ([0-9]+)" "\\1"
		  EIGEN_WORLD_VERSION "${EIGEN_WORLD_VERSION}")

		STRING(REGEX MATCH "#define EIGEN_MAJOR_VERSION [0-9]+"
		  EIGEN_MAJOR_VERSION "${EIGEN_VERSION_FILE_CONTENTS}")
		STRING(REGEX REPLACE "#define EIGEN_MAJOR_VERSION ([0-9]+)" "\\1"
		  EIGEN_MAJOR_VERSION "${EIGEN_MAJOR_VERSION}")

		STRING(REGEX MATCH "#define EIGEN_MINOR_VERSION [0-9]+"
		  EIGEN_MINOR_VERSION "${EIGEN_VERSION_FILE_CONTENTS}")
		STRING(REGEX REPLACE "#define EIGEN_MINOR_VERSION ([0-9]+)" "\\1"
		  EIGEN_MINOR_VERSION "${EIGEN_MINOR_VERSION}")

		# This is on a single line s/t CMake does not interpret it as a list of
		# elements and insert ';' separators which would result in 3.;2.;0 nonsense.
		SET(EIGEN_VERSION "${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION}")
	ENDIF (NOT EXISTS ${EIGEN_VERSION_FILE})
	SET(EIGEN_INCLUDE_DIR ${EIGEN_DIR})

	MESSAGE(STATUS "Eigen ${EIGEN_VERSION} found (include: ${EIGEN_INCLUDE_DIRS})")
else()
  MESSAGE(FATAL_ERROR "You are attempting to build without Eigen. "
          "Please use cmake variable -DEIGEN_INCLUDE_DIR_HINTS:STRING=\"PATH\" "
          "or EIGEN_INCLUDE_DIR_HINTS env. variable to a valid Eigen path. "
          "Or install last Eigen version.")
  package_report_not_found(EIGEN "Eigen cannot be found")
endif()
