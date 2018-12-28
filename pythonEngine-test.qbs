import qbs 1.0

CppApplication {
    name: "pythonEngine-test"
    
    consoleApplication: true

    Depends { name: "pythonEngine" }
    
    cpp.includePaths: [
        "Resonance/include",
        "googletest/googletest/include",
        "googletest/googletest",
        "thir",
        "boost/preprocessor/include",
        "boost/endian/include",
        "boost/config/include",
        "boost/predef/include",
        "boost/core/include",
        "boost/static_assert/include",
    ]
    
    cpp.dynamicLibraries: ["pthread"]
    cpp.defines: ['RESONANCE_STANDALONE']
    
    files: [
        "test.cpp",
        "googletest/googletest/src/gtest-all.cc",
    ]
}
