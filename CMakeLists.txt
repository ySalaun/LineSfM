CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(LINE_SFM)

# guard against in-source builds
IF(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed.")
ENDIF()

#Configure openCV install
FIND_PACKAGE(OpenCV REQUIRED)

#Configure Eigen install
SET(EIGEN_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/third_party/eigen)
ADD_SUBDIRECTORY(third_party/eigen)
INCLUDE_DIRECTORIES(${EIGEN_DIR})

#Configure Ceres install
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

#Configure Lapack install
find_package(LAPACK REQUIRED )
include_directories(${LAPACK_INCLUDE_DIR})      

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}
		    ./
                    DETECTION/ 
                    MATCHING/
                    CALIBRATION/
                    third_party/
                    third_party/EDLines)
			
SET(EVAL_GT	third_party/openMVG/tools_precisionEvaluationToGt.hpp)
		
SET(GEOMETRY	third_party/openMVG/geometry/rigid_transformation3D_srt.hpp)

SET(HISTOGRAM	third_party/openMVG/histogram/histogram.hpp)

SET(HTMLDOC	third_party/openMVG/htmlDoc/htmlDoc.hpp)
		
SET(MULTIVIEW	third_party/openMVG/multiview/essential.cpp			third_party/openMVG/multiview/essential.hpp
		third_party/openMVG/multiview/projection.cpp			third_party/openMVG/multiview/projection.hpp
		third_party/openMVG/multiview/triangulation.cpp			third_party/openMVG/multiview/triangulation.hpp
		third_party/openMVG/multiview/solver_essential_kernel.cpp	third_party/openMVG/multiview/solver_essential_kernel.hpp
		third_party/openMVG/multiview/solver_essential_five_point.cpp	third_party/openMVG/multiview/solver_essential_five_point.hpp
		third_party/openMVG/multiview/solver_fundamental_kernel.cpp	third_party/openMVG/multiview/solver_fundamental_kernel.hpp)
		
SET(NUMERIC	third_party/openMVG/numeric/numeric.cpp				third_party/openMVG/numeric/numeric.h
		third_party/openMVG/numeric/poly.h
		third_party/openMVG/numeric/lm.hpp)

SET(OPENMVG ${EVAL_GT} ${GEOMETRY} ${HISTOGRAM} ${HTMLDOC} ${MULTIVIEW} ${NUMERIC} )

SET(ACVP	third_party/acvp/alignment.cpp	third_party/acvp/alignment.h
		third_party/acvp/binomial.cpp 	third_party/acvp/binomial.h
		third_party/acvp/LWImage.cpp	third_party/acvp/LWImage.h
		third_party/acvp/mdl.cpp	third_party/acvp/mdl.h
		third_party/acvp/segment.cpp	third_party/acvp/segment.h
		third_party/acvp/vpoint.cpp	third_party/acvp/vpoint.h)
		
SET(THIRD_PARTY ${ACVP} ${OPENMVG})

SET(EDLINES_DETECTION	third_party/EDLines/edlines_interface.cpp	third_party/EDLines/edlines_interface.hpp)

SET(LINE_DETECTION	DETECTION/lsd.cpp		DETECTION/lsd.hpp
			DETECTION/mlsd.cpp		DETECTION/mlsd.hpp
			DETECTION/detection.cpp 	DETECTION/detection.hpp)
			
SET(CALIBRATION_CPP	CALIBRATION/hybrid_essential.cpp	CALIBRATION/hybrid_essential.hpp
			CALIBRATION/refinement.cpp		CALIBRATION/refinement.hpp
			CALIBRATION/scale_uniformization.cpp	CALIBRATION/scale_uniformization.hpp
			CALIBRATION/miscellaneous.hpp)
			
SET(MATCHING_CPP	MATCHING/line_matching.cpp	MATCHING/line_matching.hpp)
			
SET(INTERFACE 		interface.cpp			interface.hpp)

SET(MATCHING_LIBS 	${LAPACK_LIBRARIES}
			${CMAKE_CURRENT_SOURCE_DIR}/third_party/arpack++/libarpack++.a 
			${CMAKE_CURRENT_SOURCE_DIR}/third_party/superlu/libsuperlu.a
			arpack)

SET(EDLINES		${CMAKE_CURRENT_SOURCE_DIR}/third_party/EDLines/EDLinesLib.a)
                    
# MULTISCALE LSD
ADD_EXECUTABLE(LSD_MULTISCALE ${LINE_DETECTION}
			      ${INTERFACE}
			      ${THIRD_PARTY}
			      main_detection.cpp)

TARGET_LINK_LIBRARIES(LSD_MULTISCALE ${OpenCV_LIBS})

# LINE MATCHING
ADD_EXECUTABLE(LINE_MATCHING ${MATCHING_CPP}
                             ${LINE_DETECTION}
                             ${INTERFACE}
                             ${THIRD_PARTY}
                             main_line_matching.cpp)

TARGET_LINK_LIBRARIES(LINE_MATCHING ${LIBS} ${OpenCV_LIBS} ${MATCHING_LIBS})

# ED-LINE MATCHING
ADD_EXECUTABLE(EDLINE_MATCHING  ${EDLINES_DETECTION}
				${MATCHING_CPP}
				${INTERFACE}
				${THIRD_PARTY}
				main_edline_matching.cpp)
              
TARGET_LINK_LIBRARIES(EDLINE_MATCHING ${LIBS} ${OpenCV_LIBS} ${MATCHING_LIBS} ${EDLINES})

# VANISHING POINT
ADD_EXECUTABLE(VANISHING_POINT ${INTERFACE}
                               ${THIRD_PARTY}
                               main_vanishing_point.cpp)

TARGET_LINK_LIBRARIES(VANISHING_POINT ${OpenCV_LIBS})


# CALIBRATION
ADD_EXECUTABLE(CALIBRATION ${CALIBRATION_CPP}
                           ${INTERFACE}
                           ${THIRD_PARTY}
                           main_calibration.cpp)
                           
TARGET_LINK_LIBRARIES(CALIBRATION ${OpenCV_LIBS} ${CERES_LIBRARIES})

# RECONSTRUCTION
ADD_EXECUTABLE(RECONSTRUCTION ${CALIBRATION_CPP}
			      ${MATCHING_CPP}
			      ${LINE_DETECTION}
			      ${INTERFACE}
			      ${THIRD_PARTY}
			      main_reconstruction.cpp)

TARGET_LINK_LIBRARIES(RECONSTRUCTION ${OpenCV_LIBS} ${CERES_LIBRARIES} ${MATCHING_LIBS})

SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
