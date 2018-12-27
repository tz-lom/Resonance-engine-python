#include <Resonance/scriptengineinterface.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <Resonance/protocol.cpp>
#include <Resonance/rtc.cpp>
#include <map>
#include <vector>

using namespace Resonance::R3;

using Resonance::RTC;

static InterfacePointers ip;
static const char* engineNameString = "python";
static const char* engineInitString = "Init python";
static const char* engineCodeString = "Some python code";

class SmartPyObject : public std::unique_ptr<PyObject, void (*)(PyObject*)> {
public:
    SmartPyObject(PyObject* ptr)
        : std::unique_ptr<PyObject, void (*)(PyObject*)>(ptr, decref)
    {
    }

    static void decref(PyObject* obj)
    {
        Py_XDECREF(obj);
    }
};

class QueueMember {
public:
    enum Type {
        sendBlockToStream,
        createOutputStream,
        startTimer,
        stopTimer
    };

    QueueMember()
        : type(sendBlockToStream)
        , args(nullptr)
    {
    }

    QueueMember(Type type, PyObject* args)
        : type(type)
        , args(args)
    {
        Py_INCREF(args);
    }

    ~QueueMember()
    {
        Py_XDECREF(args);
    }

    QueueMember(const QueueMember& orig)
        : type(orig.type)
        , args(orig.args)
    {
        Py_XINCREF(args);
    }

    const Type type;
    const PyObject* args;
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

    int id;
    Thir::SerializedData::rid type;
    PyObject* si;
};

static std::map<int, OutputStreamDescription> outputs;
static std::map<int, InputStreamDescription> inputs;
static std::vector<QueueMember> queue;
static PyObject* callback_si_channels = nullptr;
static PyObject* callback_si_event = nullptr;
static PyObject* callback_db_event = nullptr;
static PyObject* callback_db_channels = nullptr;
static PyObject* callback_on_prepare = nullptr;
static PyObject* callback_on_data_block = nullptr;
static PyObject* callback_on_start = nullptr;
static PyObject* callback_on_stop = nullptr;

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
    PyStringObject* nameObj = nullptr;
    PyObject* data = nullptr;

    if (!PyArg_UnpackTuple(args, "add_to_queue", 2, 2, &nameObj, &data)) {
        Py_RETURN_FALSE;
    }

    const char* name = PyString_AS_STRING(nameObj);

    if (std::string("sendBlockToStream") == name) {
        queue.emplace_back(QueueMember::sendBlockToStream, data);
        Py_RETURN_TRUE;
    } else if (std::string("createOutputStream") == name) {
        queue.emplace_back(QueueMember::createOutputStream, data);
        Py_RETURN_TRUE;
    } else if (std::string("startTimer") == name) {
        queue.emplace_back(QueueMember::startTimer, data);
        Py_RETURN_TRUE;
    } else if (std::string("stopTimer") == name) {
        queue.emplace_back(QueueMember::stopTimer, data);
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

    if (!PyArg_UnpackTuple(args, "register_callbacks", 8, 8,
            &callback_on_prepare,
            &callback_on_data_block,
            &callback_on_start,
            &callback_on_stop,
            &callback_si_channels,
            &callback_si_event,
            &callback_db_event,
            &callback_db_channels)) {
        Py_RETURN_FALSE;
    }

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

bool initializeEngine(InterfacePointers _ip, const char* code, size_t)
{
    ip = _ip;
    Py_SetProgramName("pythonEngine");
    Py_Initialize();
    import_array1(false);
    module = Py_InitModule("resonate", ModuleMethods);

    callback_on_prepare = PyDict_GetItemString(PyModule_GetDict(module), "do_nothing");

    callback_on_data_block = callback_on_prepare;
    callback_on_start = callback_on_prepare;
    callback_on_stop = callback_on_prepare;
    callback_si_channels = callback_on_prepare;
    callback_si_event = callback_on_prepare;
    callback_db_event = callback_on_prepare;
    callback_db_channels = callback_on_prepare;

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

    (void)codeLength;
    PyRun_SimpleString(code);

    SmartPyObject inputList(PyList_New(0));
    for (uint32_t i = 0; i < streamCount; ++i) {
        Thir::SerializedData data((const char*)streams[i].data, streams[i].size);
        //data.extractString<ConnectionHeaderContainer::name>()
        Thir::SerializedData type = data.field<ConnectionHeaderContainer::type>();

        switch (type.id()) {
        //case ConnectionHeader_Int32::ID:
        //case ConnectionHeader_Int64::ID:
        case ConnectionHeader_Float64::ID: {

            SmartPyObject arglist(Py_BuildValue(
                "(ifis)",
                type.field<ConnectionHeader_Float64::channels>().value(),
                type.field<ConnectionHeader_Float64::samplingRate>().value(),
                i + 1,
                data.field<ConnectionHeaderContainer::name>().value().c_str()));
            PyObject* si = PyObject_CallObject(callback_si_channels, arglist.get());

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

            PyObject* si = PyObject_CallObject(callback_si_event, arglist.get());

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
    SmartPyObject result(PyObject_CallObject(callback_on_prepare, arglist.get()));

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
        SmartPyObject result(PyObject_CallObject(callback_db_event, arglist.get()));
        if (!result)
            throw std::runtime_error("Failed to call dbEvent");

        SmartPyObject arglist2(Py_BuildValue("(O)", result.get()));

        SmartPyObject result2(PyObject_CallObject(callback_on_data_block, arglist2.get()));
        if (!result2)
            throw std::runtime_error("Failed to call onDataBlock for Message");
    } break;
    case Float64::ID: {
        Thir::SerializedData data(block.data, block.size);

        auto vec = data.field<Float64::data>().toVector();

        int samples = data.field<Float64::samples>().value();
        npy_intp dims[2] = { static_cast<npy_intp>(samples), static_cast<npy_intp>(vec.size() / samples) };

        SmartPyObject pyData(PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, reinterpret_cast<void*>(vec.data())));

        SmartPyObject arglist(Py_BuildValue("(OlO)",
            is.si,
            data.field<Float64::created>().value(),
            pyData.get()));
        SmartPyObject result(PyObject_CallObject(callback_db_channels, arglist.get()));

        if (!result)
            throw std::runtime_error("Failed to call dbChannels");

        SmartPyObject arglist2(Py_BuildValue("(O)", result.get()));

        SmartPyObject result2(PyObject_CallObject(callback_on_data_block, arglist2.get()));
        if (!result2)
            throw std::runtime_error("Failed to call onDataBlock for Float64");
    } break;
    }

    pythonParceQueue();
}

void startEngine()
{
    SmartPyObject arglist = PyTuple_New(0);
    SmartPyObject result = PyObject_CallObject(callback_on_start, arglist.get());
    if (!result)
        throw std::runtime_error("Failed to call onStart");
}

void stopEngine()
{
    SmartPyObject arglist = PyTuple_New(0);
    SmartPyObject result = PyObject_CallObject(callback_on_stop, arglist.get());
    if (!result)
        throw std::runtime_error("Failed to call onStop");
}

bool pythonParceQueue()
{
    for (const QueueMember& event : queue) {
        switch (event.type) {
        case QueueMember::sendBlockToStream: {
            Py_ssize_t id;
            PyObject* data;
            if (!PyArg_ParseTuple(const_cast<PyObject*>(event.args), "nO", &id, &data)) {
                throw std::runtime_error("Can't unpack arguments for sendBlockToStream");
            }

            auto os = outputs[id];

            switch (os.type) {
            case Message::ID: {
                SD block = Message::create()
                               .set(RTC::now())
                               .set(0)
                               .set(PyString_AsString(data))
                               .finish();

                ip.sendBlock(os.id, SerializedDataContainer({ block->data(), block->size() }));
            } break;
            case Float64::ID: {

                SmartPyObject array = PyArray_FROM_OTF(data, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_ENSUREARRAY);

                if (!array) {
                    throw std::runtime_error("Can't unpack arguments for sendBlockToStream [channels]");
                }
                PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(array.get());
                if (PyArray_NDIM(arr) != 2) {
                    throw std::runtime_error("Can't unpack arguments for sendBlockToStream [channels dim is wrong]");
                }
                if (PyArray_TYPE(arr) != NPY_FLOAT64) {
                    throw std::runtime_error("Can't unpack arguments for sendBlockToStream [ channels type is wrong]");
                }

                int rows = PyArray_DIM(arr, 0);

                double* first = (double*)(PyArray_DATA(arr));
                double* last = first + PyArray_SIZE(arr);

                auto s1 = PyArray_STRIDE(arr, 0);
                auto s2 = PyArray_STRIDE(arr, 1);

                SD block = Float64::create()
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

            PyObject* dict = const_cast<PyObject*>(event.args);

            if (!PyDict_Check(dict)) {
                throw std::runtime_error("Can't unpack arguments for createOutputStream");
            }

            PyObject* id = PyDict_GetItemString(dict, "id");
            PyObject* name = PyDict_GetItemString(dict, "name");
            PyObject* type = PyDict_GetItemString(dict, "type");

            if (!PyInt_Check(id) || !PyString_Check(name) || !PyString_Check(type)) {
                throw std::runtime_error("Failed check arguments for createOutputStream");
            }

            if (std::string("event") == PyString_AS_STRING(type)) {
                SD type = ConnectionHeader_Message::create().next().finish();
                int sendId = ip.declareStream(PyString_AS_STRING(name), SerializedDataContainer({ type->data(), (uint32_t)type->size() }));
                if (sendId != -1) {
                    outputs[PyInt_AsLong(id)] = { sendId, Message::ID };
                }
            } else if (std::string("channels") == PyString_AS_STRING(type)) {
                PyObject* samplingRate = PyDict_GetItemString(dict, "samplingRate");
                PyObject* channels = PyDict_GetItemString(dict, "channels");

                if (!PyInt_Check(channels) || !PyFloat_Check(samplingRate)) {
                    throw std::runtime_error("Failed check arguments for createOutputStream [channels]");
                }

                SD type = ConnectionHeader_Float64::create()
                              .set(PyInt_AsLong(channels))
                              .set(PyFloat_AsDouble(samplingRate))
                              .finish();
                int sendId = ip.declareStream(PyString_AS_STRING(name), SerializedDataContainer({ type->data(), (uint32_t)type->size() }));

                if (sendId != -1) {
                    outputs[PyInt_AsLong(id)] = { sendId, Float64::ID };
                }
            }
        } break;
        case QueueMember::startTimer: {
            long int id;
            long int timeout;
            PyObject* singleShot;
            if (!PyArg_ParseTuple(const_cast<PyObject*>(event.args), "llO", &id, &timeout, &singleShot)) {
                throw std::runtime_error("Can't unpack arguments for startTimer");
            }

            ip.startTimer(id, timeout, singleShot == Py_True);

        } break;
        case QueueMember::stopTimer: {
            long int id;
            if (!PyArg_ParseTuple(const_cast<PyObject*>(event.args), "l", &id)) {
                throw std::runtime_error("Can't unpack arguments for stopTimer");
            }

            ip.stopTimer(id);

        } break;
        }
    }
    queue.clear();
    return true;
}
