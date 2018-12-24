import qbs 1.0
import qbs.Probes as Probes

CppApplication {
    name: "Unit tests"
    type: ["application", "autotest"]
    
    
    consoleApplication: true
        
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
        '/usr/lib64/python2.7/site-packages/numpy/core/include'
    ]
    
    cpp.dynamicLibraries: ["pthread"]
    cpp.defines: ['RESONANCE_STANDALONE']
    
    Probes.PkgConfigProbe {
        id: python
        name: "python-2.7"
    }
    
    Properties {
        condition: qbs.targetOS.contains("linux")
        
        cpp.defines: outer.concat(python.defines)
        cpp.dynamicLibraries: outer.concat(python.libraries)
        cpp.libraryPaths: outer.concat(python.libraryPaths)
        cpp.includePaths: outer.concat(python.includePaths)
    }
    
    files: [
        "unittests.cpp",
        "googletest/googletest/src/gtest-all.cc",
    ]
}
