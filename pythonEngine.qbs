import qbs 1.0

import qbs.Probes as Probes

Product {
   // condition: rbuild.ready

    Depends { name: "Resonance" }
    Depends { name: "cpp" }

    name: "pythonEngine"
    type: "dynamiclibrary"
    cpp.minimumOsxVersion: '10.9'
    cpp.cxxLanguageVersion: "c++11"
    Resonance.headersOnly: true

    files: [
        "pythonengine.cpp",
    ]

    Probes.PkgConfigProbe {
        id: python
        name: "python-2.7"
    }

    cpp.defines: python.defines
    cpp.dynamicLibraries: python.libraries
    cpp.libraryPaths: python.libraryPaths
    cpp.includePaths: python.includePaths

    Group {
        name: "Install module"
        fileTagsFilter: "dynamiclibrary"
        qbs.install: true
        qbs.installDir: "bin/engines"
    }
}
