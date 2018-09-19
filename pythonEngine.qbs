import qbs 1.0

import qbs.Probes as Probes

Product {
   // condition: rbuild.ready

    Depends { name: "Resonance" }
    Depends { name: "cpp" }
    
    Properties {
        condition: Resonance != null
        Resonance.standalone: true
    }

    name: "pythonEngine"
    type: "dynamiclibrary"

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
    cpp.includePaths: python.includePaths.concat("thir")

    Group {
        name: "Install module"
        fileTagsFilter: "dynamiclibrary"
        qbs.install: true
        qbs.installDir: "bin/engines"
    }
}
