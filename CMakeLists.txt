project(pythonEngine)

add_library(${PROJECT_NAME} SHARED
	pythonengine.cpp
	)
find_package(Python COMPONENTS Development NumPy)

target_link_libraries(${PROJECT_NAME} PRIVATE  Resonance CONAN_PKG::pybind11 Python::Python)
target_compile_definitions(${PROJECT_NAME} PRIVATE RESONANCE_EXPOSE_PROTOCOL RESONANCE_STANDALONE)

install_files_with_prefix(${PROJECT_NAME} "bin/engines")

#add_subdirectory(test)
