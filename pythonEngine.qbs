import qbs 1.0
import qbs.Probes as Probes

Product {
    Depends { name: "Resonance" }
    Depends { name: "cpp" }
    
    Properties {
        condition: Resonance != null
        Resonance.standalone: true
    }

    //TODO: need to fix
    //property string mobuleBaseName: "pythonEngine"

    name: "pythonEngine"
    type: "dynamiclibrary"
    files: [ "pythonengine.cpp" ]

    cpp.cxxLanguageVersion: "c++11"

//    cpp.includePaths: [
//        "thir",
//        "boost/preprocessor/include",
//        "boost/endian/include",
//        "boost/config/include",
//        "boost/predef/include",
//        "boost/core/include",
//        "boost/static_assert/include",
//        '/usr/lib64/python2.7/site-packages/numpy/core/include'
//    ]

    Properties {
        condition: qbs.targetOS.contains("windows") && qbs.architecture === "x86"
        cpp.defines: [
            'WIN32',
            '_hypot=hypot'
        ]

        //TODO: need to fix
        //name: mobuleBaseName + "_x86"

        property string pythonHome: "C:\\Python27"

        cpp.includePaths: [
            pythonHome + "/include",
            pythonHome + "/lib/site-packages/numpy/core/include",
            "thir",
            "boost/preprocessor/include",
            "boost/endian/include",
            "boost/config/include",
            "boost/predef/include",
            "boost/core/include",
            "boost/static_assert/include"
        ]
        cpp.libraryPaths: [
            pythonHome + "/libs"
        ]
        cpp.dynamicLibraries: 'python27'
    }

    Properties {
        condition: qbs.targetOS.contains("windows") && qbs.architecture === "x86_64"
        cpp.defines: [
            'MS_WIN64',
            '_hypot=hypot'
        ]

        //TODO: need to fix
        //name: mobuleBaseName + "_x86_64"

        property string pythonHome: 'C:/Python27_amd64'

        cpp.includePaths: [
            pythonHome + "/include",
            pythonHome + "/lib/site-packages/numpy/core/include",
            "thir",
            "boost/preprocessor/include",
            "boost/endian/include",
            "boost/config/include",
            "boost/predef/include",
            "boost/core/include",
            "boost/static_assert/include"
        ]
        cpp.libraryPaths: [
            pythonHome + "/libs"
        ]
        cpp.dynamicLibraries: 'python27'
    }

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
    
    Group {
        name: "Install module"
        fileTagsFilter: "dynamiclibrary"
        qbs.install: true
        qbs.installDir: "bin/engines"
    }
}
