if (CMAKE_SIZEOF_VOID_P EQUAL 8)
project(pythonEngine)

add_library(${PROJECT_NAME} SHARED
	pythonengine.cpp
	)
find_package(Python 3 COMPONENTS Development NumPy)
find_package(pybind11)

target_link_libraries(${PROJECT_NAME} PRIVATE  Resonance pybind11::pybind11 Python::Python)
target_compile_definitions(${PROJECT_NAME} PRIVATE RESONANCE_EXPOSE_PROTOCOL RESONANCE_STANDALONE)

install_target(${PROJECT_NAME} "bin/engines")

#add_subdirectory(test)
endif()
