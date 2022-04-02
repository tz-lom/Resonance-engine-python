# cmake_minimum_required(VERSION 3.16)
# project(
#        ResonancePythonEngineTests
#	LANGUAGES CXX)
	

# include(${CMAKE_BINARY_/DIR}/conanbuildinfo.cmake)
# conan_basic_setup(TARGETS)

add_executable(Test
	test.cpp
	)	
	
target_link_libraries(Test PRIVATE  pythonEngine Resonance CONAN_PKG::gtest)
target_compile_definitions(Test PRIVATE RESONANCE_EXPOSE_PROTOCOL RESONANCE_STANDALONE)

#add_executable(UnitTests
#	unittests.cpp
#)

#target_link_libraries(Test PRIVATE  pythonEngine Python::Python CONAN_PKG::pybind11 CONAN_PKG::gtest Resonance)
