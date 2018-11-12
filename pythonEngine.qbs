import qbs 1.0
import qbs.Environment // Подключил еще методов
import qbs.File        // Подключил еще методов
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
    files: [ "pythonengine.cpp" ]

    property string pythonHome: 'c:\\Python27'

    cpp.dynamicLibraries: 'python27'
    cpp.cxxLanguageVersion: "c++11"

    cpp.defines: [
        'WIN32',
        '_hypot=hypot'
    ]

    cpp.libraryPaths: pythonHome + "/libs" // Добавил в окружение доп. инструмент -PythonRun с путями, напрямую задается

    cpp.includePaths: [
            "thir",
            "boost/preprocessor/include",
            "boost/endian/include",
            "boost/config/include",
            "boost/predef/include",
            "boost/core/include",
            "boost/static_assert/include",
            pythonHome + "/include"
        ]

    Group {
        name: "Install module"
        fileTagsFilter: "dynamiclibrary"
        qbs.install: true
        qbs.installDir: "bin/engines"
    }
}
