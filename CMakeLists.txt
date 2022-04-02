project(pythonEngine)

add_library(${PROJECT_NAME} SHARED
	pythonengine.cpp
	)
find_package(Python COMPONENTS Development NumPy)

target_link_libraries(${PROJECT_NAME} PRIVATE  Resonance CONAN_PKG::pybind11 Python::Python)
target_compile_definitions(${PROJECT_NAME} PRIVATE RESONANCE_EXPOSE_PROTOCOL RESONANCE_STANDALONE)
set_target_properties(${PROJECT_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/engines"
)

#install(TARGETS
#    pythonEngine
#    LIBRARY DESTINATION engines
#    ARCHIVE DESTINATION engines
#    RUNTIME DESTINATION engines
#)

#add_subdirectory(test)
