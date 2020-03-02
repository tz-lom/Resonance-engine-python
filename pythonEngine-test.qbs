import qbs 1.0

CppApplication {
    name: "pythonEngine-test"
    
    consoleApplication: true

    Depends { name: "pythonEngine" }
    Depends { name: "boost" }
    Depends { name: "gtest" }
    
    
    cpp.includePaths: [
        "Resonance/include",
        "thir"
    ]
    
    cpp.dynamicLibraries: ["pthread"]
    cpp.defines: ['RESONANCE_STANDALONE', 'RESONANCE_EXPOSE_PROTOCOL']
    cpp.cxxLanguageVersion: "c++11"

    
    files: [
        "test.cpp",
        "initCode.h",
        "Resonance/tests/test-script-engine-interface.hpp"
    ]
}
