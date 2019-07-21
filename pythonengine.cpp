// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com


#include <Resonance/scriptengineinterface.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <Resonance/protocol.cpp>
#include <Resonance/rtc.cpp>
#include <iostream>
#include <map>
#include <vector>

#ifdef LIBRARY_HACK
#include <dlfcn.h>
#endif

#include <boost/preprocessor/stringize.hpp>

using namespace Resonance::R3;

using Resonance::RTC;

static InterfacePointers ip;
static const char* engineNameString = "python " BOOST_PP_STRINGIZE(PY_MAJOR_VERSION) "." BOOST_PP_STRINGIZE(PY_MINOR_VERSION);
static const char* engineInitString = "import resonance\n";
static const char* engineCodeString = R"(from resonance import *
createOutput(input(0), 'out')
)";

class SmartPyObject
{
public:
    enum CaptureMode {
        Move,
        Copy
    };

    SmartPyObject(PyObject* ptr, CaptureMode mode=Move)
    {
        if(mode == Copy)
        {
            Py_INCREF(ptr);
        }
        obj = ptr;
    }

    ~SmartPyObject()
    {
        Py_XDECREF(obj);
    }

    SmartPyObject(const SmartPyObject &right)
    {
        Py_INCREF(right.obj);
        obj = right.obj;
    }

    SmartPyObject& operator=(const SmartPyObject &right)
    {
        Py_INCREF(right.obj);
        obj = right.obj;
        return *this;
    }

    SmartPyObject(SmartPyObject &&right) = default;

    PyObject* get() const
    {
        return obj;
    }

    void reset(PyObject* ptr)
    {
        Py_XDECREF(obj);
        obj = ptr;
    }

    operator bool()
    {
        return obj != nullptr;
    }

private:
    PyObject *obj;
};

struct QueueMember {
    enum Type {
        sendBlockToStream,
        createOutputStream,
        startTimer,
        stopTimer
    };

    const Type type;
    const SmartPyObject args;
};

typedef struct {
    int id;
    Thir::SerializedData::rid type;
} OutputStreamDescription;

class InputStreamDescription {
public:
    InputStreamDescription()
        : id(0)
        , type(0)
        , si(nullptr)
    {
    }

    InputStreamDescription(int id, Thir::SerializedData::rid type, PyObject* si)
        : id(id)
        , type(type)
        , si(si)
    {
        Py_XINCREF(si);
    }

    ~InputStreamDescription()
    {
        Py_XDECREF(si);
    }

    InputStreamDescription(const InputStreamDescription& orig)
        : id(orig.id)
        , type(orig.type)
        , si(orig.si)
    {
        Py_XINCREF(si);
    }
    
    InputStreamDescription& operator=(const InputStreamDescription&) = default;

    int id;
    Thir::SerializedData::rid type;
    PyObject* si;
};

static std::map<int, OutputStreamDescription> outputs;
static std::map<int, InputStreamDescription> inputs;
static std::vector<QueueMember> queue;
static SmartPyObject callback_si_channels = nullptr;
static SmartPyObject callback_si_event = nullptr;
static SmartPyObject callback_db_event = nullptr;
static SmartPyObject callback_db_channels = nullptr;
static SmartPyObject callback_on_prepare = nullptr;
static SmartPyObject callback_on_data_block = nullptr;
static SmartPyObject callback_on_start = nullptr;
static SmartPyObject callback_on_stop = nullptr;
static SmartPyObject callback_trace = nullptr;

void trace(PyObject* obj)
{
    SmartPyObject arglist = Py_BuildValue("(O)", obj);
    SmartPyObject result = PyObject_CallObject(callback_trace.get(), arglist.get());
    if (!result)
        throw std::runtime_error("Failed to call trace");
}

bool pythonParceQueue();

const char* engineName()
{
    return engineNameString;
}

const char* engineInitDefault()
{
    return engineInitString;
}

const char* engineCodeDefault()
{
    return engineCodeString;
}

static PyObject* module;

PyObject* mod_addToQueue(PyObject*, PyObject* args)
{
    const char *name = nullptr;
    PyObject* data = nullptr;

    if (!PyArg_ParseTuple(args, "sO", &name, &data)) {
        Py_RETURN_FALSE;
    }

    if (std::string("sendBlockToStream") == name) {
        queue.emplace_back(QueueMember{QueueMember::sendBlockToStream, data});
        Py_RETURN_TRUE;
    } else if (std::string("createOutputStream") == name) {
        queue.emplace_back(QueueMember{QueueMember::createOutputStream, data});
        Py_RETURN_TRUE;
    } else if (std::string("startTimer") == name) {
        queue.emplace_back(QueueMember{QueueMember::startTimer, data});
        Py_RETURN_TRUE;
    } else if (std::string("stopTimer") == name) {
        queue.emplace_back(QueueMember{QueueMember::stopTimer, data});
        Py_RETURN_TRUE;
    } else {
        PyErr_BadInternalCall();
        Py_XDECREF(data);
        Py_RETURN_FALSE;
    }
}

PyObject* mod_do_nothing(PyObject*, PyObject*)
{
    Py_RETURN_NONE;
}

PyObject* mod_register_callbacks(PyObject*, PyObject* args)
{

    PyObject *new_callback_on_prepare = nullptr,
            *new_callback_on_data_block = nullptr,
            *new_callback_on_start = nullptr,
            *new_callback_on_stop = nullptr,
            *new_callback_si_channels = nullptr,
            *new_callback_si_event = nullptr,
            *new_callback_db_event = nullptr,
            *new_callback_db_channels = nullptr,
            *new_callback_trace = nullptr;
    
    if (!PyArg_UnpackTuple(args, "register_callbacks", 8, 9,
            &new_callback_on_prepare,
            &new_callback_on_data_block,
            &new_callback_on_start,
            &new_callback_on_stop,
            &new_callback_si_channels,
            &new_callback_si_event,
            &new_callback_db_event,
            &new_callback_db_channels,
            &new_callback_trace)) {
        Py_RETURN_FALSE;
    }
    
    callback_on_prepare    .reset(new_callback_on_prepare       );
    callback_on_data_block .reset(new_callback_on_data_block);
    callback_on_start      .reset(new_callback_on_start     );
    callback_on_stop       .reset(new_callback_on_stop      );
    callback_si_channels   .reset(new_callback_si_channels  );
    callback_si_event      .reset(new_callback_si_event     );
    callback_db_event      .reset(new_callback_db_event     );
    callback_db_channels   .reset(new_callback_db_channels  );
    callback_trace         .reset(new_callback_trace        );

    Py_RETURN_TRUE;
}

static PyMethodDef ModuleMethods[] = {
    { "register_callbacks", mod_register_callbacks, METH_VARARGS,
        "Register callbacks for Resonance runtime." },
    { "add_to_queue", mod_addToQueue, METH_VARARGS,
        "Adds data to resonance event queue" },
    { "do_nothing", mod_do_nothing, METH_VARARGS,
        "Used as a default value for callbacks" },
    { nullptr, nullptr, 0, nullptr }
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef ModuleDefinition = {
    PyModuleDef_HEAD_INIT,
    "resonate",
    nullptr,
    -1,
    ModuleMethods,
    nullptr,
    nullptr,
    nullptr,
    nullptr
};

PyMODINIT_FUNC
PyInit_resonate(void)
{
    return PyModule_Create(&ModuleDefinition);
}

#endif

#define VAL(str) #str
#define TOSTRING(str) VAL(str)

bool initializeEngine(InterfacePointers _ip, const char *code, size_t)
{
    ip = _ip;
    
#ifdef LIBRARY_HACK
    dlopen("lib" TOSTRING(LIBRARY_HACK) ".so", RTLD_LAZY | RTLD_GLOBAL);
#endif
    
#if PY_MAJOR_VERSION >= 3
    Py_SetProgramName(L"pythonEngine");
    PyImport_AppendInittab("resonate", PyInit_resonate);    
#else
    Py_SetProgramName("pythonEngine");
#endif
    Py_Initialize();
    
    import_array1(false)
    
#if PY_MAJOR_VERSION >= 3  
    module = PyImport_ImportModule("resonate");
#else
    module = Py_InitModule("resonate", ModuleMethods);
#endif

    callback_on_prepare = SmartPyObject(PyDict_GetItemString(PyModule_GetDict(module), "do_nothing"));

    callback_on_data_block = callback_on_prepare;
    callback_on_start = callback_on_prepare;
    callback_on_stop = callback_on_prepare;
    callback_si_channels = callback_on_prepare;
    callback_si_event = callback_on_prepare;
    callback_db_event = callback_on_prepare;
    callback_db_channels = callback_on_prepare;
    callback_trace = callback_on_prepare;

    PyRun_SimpleString(code);
    return true;
}

void freeEngine()
{
    Py_Finalize();
}

bool prepareEngine(const char* code, size_t codeLength, const SerializedDataContainer* const streams, size_t streamCount)
{
    outputs.clear();
    inputs.clear();
    queue.clear();

    SmartPyObject inputList(PyList_New(0));
    for (uint32_t i = 0; i < streamCount; ++i) {
        Thir::SerializedData data((const char*)streams[i].data, streams[i].size);
        //data.extractString<ConnectionHeaderContainer::name>()
        Thir::SerializedData type = data.field<ConnectionHeaderContainer::type>();

        switch (type.id()) {
        //case ConnectionHeader_Int32::ID:
        //case ConnectionHeader_Int64::ID:
        case ConnectionHeader_Float64::ID: {
            PyObject* si = PyObject_CallFunction(
                        callback_si_channels.get(), 
                        "ifis",
                        type.field<ConnectionHeader_Float64::channels>().value(),
                        type.field<ConnectionHeader_Float64::samplingRate>().value(),
                        i + 1,
                        data.field<ConnectionHeaderContainer::name>().value().c_str()
                    );
            PyObject_SetAttrString(si, "online", Py_True);
            
            if (si == nullptr)
                throw std::runtime_error("Failed to construct Float64 stream definition");

            if (PyList_Append(inputList.get(), si) != 0) {
                throw std::runtime_error("Failed to populate list ");
            }

            inputs[i] = InputStreamDescription(static_cast<int>(i + 1), Float64::ID, si);
        } break;

        case ConnectionHeader_Message::ID: {
            SmartPyObject arglist(Py_BuildValue(
                "(is)",
                i + 1,
                data.field<ConnectionHeaderContainer::name>().value().c_str()));

            PyObject* si = PyObject_CallObject(callback_si_event.get(), arglist.get());
            PyObject_SetAttrString(si, "online", Py_True);

            if (si == nullptr)
                throw std::runtime_error("Failed to construct Message stream definition");

            if (PyList_Append(inputList.get(), si) != 0) {
                throw std::runtime_error("Failed to populate list ");
            }

            inputs[i] = InputStreamDescription(static_cast<int>(i + 1), Message::ID, si);
        } break;
        }
    }

    SmartPyObject arglist(Py_BuildValue("(s,O)", code, inputList.get()));
    SmartPyObject result(PyObject_CallObject(callback_on_prepare.get(), arglist.get()));

    if (!result)
        throw std::runtime_error("Failed to call onPrepare");

    return pythonParceQueue();
}

void blockReceived(const int id, const SerializedDataContainer block)
{

    auto is = inputs[id];
    switch (is.type) {
    case Message::ID: {

        Thir::SerializedData data(block.data, block.size);

        SmartPyObject arglist(Py_BuildValue("(Ols)",
            is.si,
            data.field<Message::created>().value(),
            data.field<Message::message>().value().c_str()));
        SmartPyObject result(PyObject_CallObject(callback_db_event.get(), arglist.get()));
        if (!result)
            throw std::runtime_error("Failed to call dbEvent");

        SmartPyObject arglist2(Py_BuildValue("(O)", result.get()));

        SmartPyObject result2(PyObject_CallObject(callback_on_data_block.get(), arglist2.get()));
        if (!result2)
            throw std::runtime_error("Failed to call onDataBlock for Message");
    } break;
    case Float64::ID: {
        Thir::SerializedData data(block.data, block.size);

        auto vec = data.field<Float64::data>().toVector();

        int samples = data.field<Float64::samples>().value();
        npy_intp dims[2] = { static_cast<npy_intp>(samples), static_cast<npy_intp>(vec.size() / samples) };

        PyObject* pyDataBorrowed = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, reinterpret_cast<void*>(vec.data()));
        PyObject* pyData = PyArray_Copy(reinterpret_cast<PyArrayObject*>(pyDataBorrowed));

        SmartPyObject result(PyObject_CallFunction(
                                 callback_db_channels.get(),
                                 "OlO",
                                 is.si,
                                 data.field<Float64::created>().value(),
                                 pyData
                                 ));

        if (!result)
            throw std::runtime_error("Failed to call dbChannels");

        SmartPyObject arglist2(Py_BuildValue("(O)", result.get()));

        SmartPyObject result2(PyObject_CallObject(callback_on_data_block.get(), arglist2.get()));
        if (!result2)
            throw std::runtime_error("Failed to call onDataBlock for Float64");
    } break;
    }

    pythonParceQueue();
}

void startEngine()
{
    SmartPyObject arglist = PyTuple_New(0);
    SmartPyObject result = PyObject_CallObject(callback_on_start.get(), arglist.get());
    if (!result)
        throw std::runtime_error("Failed to call onStart");
}

void stopEngine()
{
    SmartPyObject arglist = PyTuple_New(0);
    SmartPyObject result = PyObject_CallObject(callback_on_stop.get(), arglist.get());
    if (!result)
        throw std::runtime_error("Failed to call onStop");
}

void onTimer(const int id, const uint64_t time)
{
}

bool pythonParceQueue()
{
    for (const QueueMember& event : queue) {
        switch (event.type) {
        case QueueMember::sendBlockToStream: {
            PyObject* streamInfo;
            PyObject* data;
            
            if (!PyArg_ParseTuple(const_cast<PyObject*>(event.args.get()), "OO", &streamInfo, &data)) {
                throw std::runtime_error("Can't unpack arguments for sendBlockToStream");
            }
            
            auto idObject = PyObject_GetAttrString(streamInfo, "id");

            auto id = PyLong_AsSize_t(idObject);
            
            auto os = outputs[id];

            switch (os.type) {
            case Message::ID: {
                SD block = Message::create()
                               .set(RTC::now())
                               .set(0)
                               .set(std::string(PyUnicode_AsUTF8(data)))
                               .finish();

                ip.sendBlock(os.id, SerializedDataContainer({ block->data(), block->size() }));
            } break;
            case Float64::ID: {
                if (!PyArray_Check(data)) {
                    throw std::runtime_error("Can't unpack arguments for sendBlockToStream [channels]");
                }
                PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(data);
                if (PyArray_NDIM(arr) != 2) {
                    throw std::runtime_error("Can't unpack arguments for sendBlockToStream [channels dim is wrong]");
                }
                if (PyArray_TYPE(arr) != NPY_FLOAT64) {
                    throw std::runtime_error("Can't unpack arguments for sendBlockToStream [ channels type is wrong]");
                }

                int rows = PyArray_DIM(arr, 0);

                double* first = (double*)(PyArray_DATA(arr));
                double* last = first + PyArray_SIZE(arr);

                SD block
                    = Float64::create()
                          .set(RTC::now())
                          .set(0)
                          .set(rows)
                          .add(first, last)
                          .finish()
                          .finish();

                ip.sendBlock(os.id, SerializedDataContainer({ block->data(), block->size() }));

            } break;
            }
        } break;
        case QueueMember::createOutputStream: {

            PyObject* streamObj = const_cast<PyObject*>(event.args.get());

            PyObject* id = PyObject_GetAttrString(streamObj, "id");
            PyObject* name = PyObject_GetAttrString(streamObj, "name");
            PyObject* type = PyObject_GetAttrString(streamObj, "_source");

            if (!PyLong_Check(id) ) { // @todo: check all args
                throw std::runtime_error("Failed check arguments for createOutputStream");
            }
            
            const char* nameStr = PyUnicode_AsUTF8(name);
            

            if ((PyObject*)type->ob_type == callback_si_event.get()) {
                SD type = ConnectionHeader_Message::create().next().finish();
                int sendId = ip.declareStream(nameStr, SerializedDataContainer({ type->data(), (uint32_t)type->size() }));
                if (sendId != -1) {
                    outputs[PyLong_AsLong(id)] = { sendId, Message::ID };
                }
            } else if ((PyObject*)type->ob_type == callback_si_channels.get()) {
                PyObject* samplingRate = PyObject_GetAttrString(type, "samplingRate");
                PyObject* channels = PyObject_GetAttrString(type, "channels");

                if (!PyLong_Check(channels) || !PyFloat_Check(samplingRate)) {
                    throw std::runtime_error("Failed check arguments for createOutputStream [channels]");
                }
                

                SD type = ConnectionHeader_Float64::create()
                              .set(PyLong_AsLong(channels))
                              .set(PyFloat_AsDouble(samplingRate))
                              .finish();
                int sendId = ip.declareStream(nameStr, SerializedDataContainer({ type->data(), (uint32_t)type->size() }));

                if (sendId != -1) {
                    outputs[PyLong_AsLong(id)] = { sendId, Float64::ID };
                }
            }
        } break;
        case QueueMember::startTimer: {
            /*long int id;
            long int timeout;
            PyObject* singleShot;
            if (!PyArg_ParseTuple(const_cast<PyObject*>(event.args), "llO", &id, &timeout, &singleShot)) {
                throw std::runtime_error("Can't unpack arguments for startTimer");
            }

            ip.startTimer(id, timeout, singleShot == Py_True);
*/
        } break;
        case QueueMember::stopTimer: {
            /*long int id;
            if (!PyArg_ParseTuple(const_cast<PyObject*>(event.args), "l", &id)) {
                throw std::runtime_error("Can't unpack arguments for stopTimer");
            }

            ip.stopTimer(id);
*/
        } break;
        }
    }
    queue.clear();
    return true;
}
