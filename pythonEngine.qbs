import qbs 1.0
import qbs.Process
import qbs.Probes as Probes

 Product {
    condition: pythonAndNumpyRecognizer.found || pythonPkgConfig.found
    Depends { name: "Resonance" }
    Depends { name: "cpp" }
    Depends { name: "boost" }
    
    Properties {
        condition: Resonance != null
        Resonance.standalone: true
    }

    name: "pythonEngine"
    type: "dynamiclibrary"
    files: [ "pythonengine.cpp" ]

    cpp.cxxLanguageVersion: "c++11"
    cpp.defines: ['RESONANCE_EXPOSE_PROTOCOL', 'RESONANCE_STANDALONE']

    Probes.PkgConfigProbe {
        id: pythonPkgConfig
        name: "python-3.8-embed"
    }

    Probe {
        id: pythonAndNumpyRecognizer

        property string pythonRoot: ''
        property string pythonArch: ''
        property string numpyIncludeDir: ''
        property var includePaths: []
        property var libraries: []

        configure: {
            var app = ["python"];
            var getPythonRoot = ["-c", "import sys; print(sys.exec_prefix)"];
            var getPythonArch = ["-c", "import platform; print(platform.architecture()[0])"];
            const getNumpyIncludeDir = ["-c", "import numpy.distutils; print(numpy.distutils.misc_util.get_numpy_include_dirs()[0])"];
            const getPythonIncludeDir = ['-c', "from sysconfig import get_paths as gp; print(gp()['include'])"]
            const getPythonLibrariesDir = ['-c', "from sysconfig import get_paths as gp; print(gp()['data'])"]

            var pythonArchBuffer;

            found = false;
            var proc = new Process();

            if(proc.exec(app, getPythonRoot) === 0){
                pythonRoot = proc.readStdOut().trim();
                found = true;
            }else{
                found = false;
                return;
            }


            if(proc.exec(app, getPythonArch) === 0){
                pythonArchBuffer = proc.readStdOut().trim();
                found = true;
                if(pythonArchBuffer === "64bit"){
                    pythonArch = "x86_64"
                }else if(pythonArchBuffer === "32bit"){
                    pythonArch = "x86"
                }else{
                    found = false;
                }
            }

            if(proc.exec(app, getNumpyIncludeDir) === 0){
                numpyIncludeDir = proc.readStdOut().trim();
            }
            if(proc.exec(app, getPythonIncludeDir) === 0)
            {
                includePaths = [
                    proc.readStdOut().trim()
                ];
                if(numpyIncludeDir) includePaths.push(numpyIncludeDir);
            }
            if(proc.exec(app, getPythonLibrariesDir) === 0)
            {
                const basePath = proc.readStdOut().trim();
                libraries = [
                    basePath + '\\libs\\libpython38.dll.a',
                    basePath + '\\python38.dll'
                ];
            }
        }
    }

    
    cpp.includePaths: [
        "Resonance/include",
        "thir"
    ]


    Group {
        name: '!'+pythonPkgConfig.libraries//+pythonAndNumpyRecognizer.includePaths
    }

    Properties {
        condition: qbs.targetOS.contains("windows")

        cpp.dynamicLibraries: outer.concat(pythonAndNumpyRecognizer.libraries)
        cpp.includePaths: outer.concat(pythonAndNumpyRecognizer.includePaths)
    }

    Properties {
        condition: qbs.targetOS.contains("linux")

        cpp.defines: outer.concat(pythonPkgConfig.defines).concat(['LIBRARY_HACK='+pythonPkgConfig.libraries])
        cpp.dynamicLibraries: outer.concat(pythonPkgConfig.libraries)
        cpp.libraryPaths: outer.concat(pythonPkgConfig.libraryPaths)
        cpp.includePaths: outer.concat(pythonPkgConfig.includePaths)
        cpp.linkerFlags: ['-export-dynamic']
    }

    
    Group {
        name: "Install module"
        fileTagsFilter: "dynamiclibrary"
        qbs.install: true
        qbs.installDir: "bin/engines"
    }
}
