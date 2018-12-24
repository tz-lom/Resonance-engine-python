import qbs 1.0

Project {
    name: "Resonance-engine-python"
    
    qbsSearchPaths: "Resonance/qbs"
    
    references: [
        'pythonEngine.qbs',
        'pythonEngine-test.qbs',
        'pythonEngine-unittest.qbs'
    ]
}
