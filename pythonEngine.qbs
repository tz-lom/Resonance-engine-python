import qbs 1.0
import qbs.Process
import qbs.Probes as Probes

Product {
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
    cpp.defines: "RESONANCE_EXPOSE_PROTOCOL"

    Probe {
        id: pythonAndNumpyRecognizer

        property string pythonRoot: ''
        property string pythonArch: ''
        property string numpyIncludeDir: ''

        configure: {
            var app = ["python3.7"];
            var getPythonRoot = ["-c", "import sys; print(sys.exec_prefix)"];
            var getPythonArch = ["-c", "import platform; print(platform.architecture()[0])"];
            var getNumpyIncludeDir = ["-c", "import numpy.distutils; print(numpy.distutils.misc_util.get_numpy_include_dirs()[0])"];

            var pythonArchBuffer;

            found = false;
            var proc = new Process();

            if(proc.exec(app, getPythonRoot) === 0){
                pythonRoot = proc.readStdOut().trim();
                found = true;
            }else{
                found = false;
            }

            if(found){
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
                }else{
                    found = false;
                }
            }

            if(found){
                if(proc.exec(app, getNumpyIncludeDir) === 0){
                    numpyIncludeDir = proc.readStdOut().trim();
                }else{
                    found = false;
                }
            }
            proc.close();
        }
    }

    
    cpp.includePaths: [
        "Resonance/include",
        "thir"
    ]

    Properties {
        condition: qbs.targetOS.contains('windows')
        
        cpp.dynamicLibraries: 'python37'
    }
/*
    Properties {
        condition: qbs.targetOS.contains("windows") && qbs.architecture === "x86"
        cpp.defines: outer.concat(['WIN32', '_hypot=hypot'])
    }

    Properties {
        condition: qbs.targetOS.contains("windows") && qbs.architecture === "x86_64"
        cpp.defines: outer.concat(['MS_WIN64', '_hypot=hypot'])
    }
*/
    Probes.PkgConfigProbe {
        id: python
        name: "python-3.7"
    }

    Properties {
        condition: qbs.targetOS.contains("linux")
        
        cpp.defines: outer.concat(python.defines) //.concat(['LIBRARY_HACK='+python.libraries])
        cpp.dynamicLibraries: outer.concat(python.libraries)
        cpp.libraryPaths: outer.concat(python.libraryPaths)
        cpp.includePaths: outer.concat(python.includePaths)
        cpp.linkerFlags: ['-export-dynamic']
    }

    
    Group {
        name: "Install module"
        fileTagsFilter: "dynamiclibrary"
        qbs.install: true
        qbs.installDir: "bin/engines"
    }
}
