#include <Resonance/scriptengineinterface.h>

#include <Python.h>

#include <Resonance/protocol.cpp>
#include <Resonance/rtc.cpp>

#include <map>


using namespace Resonance::R3;

using Resonance::RTC;

static InterfacePointers ip;
static const char* engineNameString = "python";
static const char* engineInitString = "Init python";
static const char* engineCodeString = "Some python code";

typedef struct {
    int id;
    Thir::SerializedData::rid type;
} StreamDescription;

static std::map<int, StreamDescription> outputs;
static std::map<int, StreamDescription> inputs;
bool pythonParceQueue();


const char * engineName()
{
    return engineNameString;
}

const char * engineInitDefault()
{
    return engineInitString;
}

const char * engineCodeDefault()
{
    return engineCodeString;
}

static PyObject *module;

PyObject* mod_createStream(PyObject *self, PyObject *args)
{
    (void)self;
    int id;
    char *c_name;
    char *c_type;
    PyObject *additional;

    if(!PyArg_UnpackTuple(args, "issO", 3, 5, &id, &c_name, &c_type, &additional))
    {
        return PyBool_FromLong(false);
    }

    std::string name = c_name;
    std::string type = c_type;

    if(type=="event")
    {
        SD type = ConnectionHeader_Message::create().next().finish();
        int sendId = ip.declareStream(name.data(), SerializedDataContainer({type->data(), (uint32_t)type->size()}));
        if(sendId!=-1)
        {
            outputs[id] = {sendId, Message::ID};
            return PyBool_FromLong(true);
        }
    }
    else if(type=="channels")
    {
        double samplingRate;
        int channels;

        if(!PyArg_UnpackTuple(additional, "dI", 2, 2, &samplingRate, &channels))
        {
            return PyBool_FromLong(false);
        }

        SD type = ConnectionHeader_Float64::create()
                .set(channels)
                .set(samplingRate)
                .finish();
        int sendId = ip.declareStream(name.data(), SerializedDataContainer({type->data(), (uint32_t)type->size()}));

        if(sendId!=-1)
        {
            outputs[id] = {sendId, Float64::ID};
            return PyBool_FromLong(true);
        }
    }
    return PyBool_FromLong(false);
}

PyObject* mod_sendData(PyObject *self, PyObject *args)
{
    (void)self;
    int id;

    if(!PyArg_UnpackTuple(args, "i", 1, 5, &id))
    {
        return PyBool_FromLong(false);
    }

    auto os = outputs[id];

    switch(os.type)
    {
    case Message::ID:
    {
       /* PyArg_VaParse()
        SerializedData *block = Message::create()
                .set(RTC::now())
                .set(0)
                .set(Rcpp::as<std::string>(args["data"]))
                .finish();

        ip.sendBlock(os.id, SerializedDataContainer({block->data(), block->size()}) );
        delete block;*/
    }
        break;
    case Float64::ID:
    {

        /*Rcpp::NumericMatrix data = args["data"];

        int rows = data.nrow();
        std::vector<double> idata = Rcpp::as<std::vector<double> >( transpose(data) );

        SerializedData *block = Float64::create()
                .set(RTC::now())
                .set(0)
                .set(rows)
                .set(idata)
                .finish();

        ip.sendBlock(os.id, SerializedDataContainer({block->data(), block->size()}) );
        delete block;*/
    }
        break;
    }
    return PyBool_FromLong(false);
}

static PyMethodDef ModuleMethods[] = {
    {"create_stream", mod_createStream, METH_VARARGS,
     "Creates new data stream."},
    {"send_data", mod_sendData, METH_VARARGS,
     "Sends data to stream."},
    {nullptr, nullptr, 0, nullptr}
};

bool initializeEngine(InterfacePointers _ip, const char *code, size_t codeLength)
{
    ip = _ip;
    Py_SetProgramName("pythonEngine");
    Py_Initialize();
    module = Py_InitModule("resonate", ModuleMethods);
    (void)codeLength;
    PyRun_SimpleString(code);
    return true;
}

void freeEngine()
{
    Py_Finalize();
}

bool prepareEngine(const char *code, size_t codeLength, const SerializedDataContainer * const streams, size_t streamCount)
{
    outputs.clear();
    inputs.clear();

    (void)codeLength;
    PyRun_SimpleString(code);

    PyObject *inputList = PyList_New(0);
    int registeredInputs = 0;
    for(uint32_t i=0; i<streamCount; ++i)
    {
        Thir::SerializedData data((const char*)streams[i].data, streams[i].size);
        //data.extractString<ConnectionHeaderContainer::name>()
        Thir::SerializedData type = data.field<ConnectionHeaderContainer::type>();

        switch(type.id())
        {
        //case ConnectionHeader_Int32::ID:
        //case ConnectionHeader_Int64::ID:
        case ConnectionHeader_Float64::ID:
        {
            PyList_Append(inputList, Py_BuildValue(
                              "{s:s,s:I,s:d}",
                              "type", "channels",
                              "channels", type.field<ConnectionHeader_Float64::channels>().value(),
                              "samplingRate", type.field<ConnectionHeader_Float64::samplingRate>().value()
                              ));

            inputs[i] = {++registeredInputs, Float64::ID};
        }
            break;

        case ConnectionHeader_Message::ID:
            PyList_Append(inputList, Py_BuildValue(
                              "{s:s}",
                              "type", "event"
                              ));
            inputs[i] = {++registeredInputs, Message::ID};
            break;
        }
    }


    PyObject *dict = PyModule_GetDict(module);
    PyObject *onPrepare = PyDict_GetItemString(dict, "on_prepare");
    if(onPrepare && PyCallable_Check(onPrepare))
    {
        Py_DECREF(PyObject_CallFunctionObjArgs(onPrepare, inputList));
    }
    Py_DECREF(inputList);
    Py_DECREF(onPrepare);
    Py_DECREF(dict);

    return true;
}

void blockReceived(const int id, const SerializedDataContainer block)
{
    PyObject *dict = PyModule_GetDict(module);
    auto is = inputs[id];
    switch(is.type)
    {
    case Message::ID:
    {
        PyObject *onDataBlock = PyDict_GetItemString(dict, "on_datablock_message");
        if(onDataBlock && PyCallable_Check(onDataBlock))
        {
            Thir::SerializedData data(block.data, block.size);
            Py_DECREF(PyObject_CallFunction(onDataBlock, "isK",
                                            is.id,
                                            data.field<Message::message>().value().c_str(),
                                            data.field<Message::created>().value()
                                            ));

        }
        Py_DECREF(onDataBlock);
    }
        break;
    case Float64::ID:
    {
        PyObject *onDataBlock = PyDict_GetItemString(dict, "on_datablock_double");
        if(onDataBlock && PyCallable_Check(onDataBlock))
        {
            Thir::SerializedData data(block.data, block.size);

            auto vec = data.field<Float64::data>().toVector();
            int samples = data.field<Float64::samples>();

            PyObject *ldata = PyList_New(vec.size());
            for(int i=vec.size()-1; i>=0; --i)
            {
                PyList_SET_ITEM(ldata, i, PyFloat_FromDouble(vec[i]));
            }


            Py_DECREF(PyObject_CallFunction(onDataBlock, "iOIK",
                                            is.id,
                                            samples,
                                            data.field<Float64::created>().value()/1E3
                                            ));
            Py_DECREF(ldata);
        }
        Py_DECREF(onDataBlock);
    }
        break;
    }

    Py_DECREF(dict);
}

void startEngine()
{
}

void stopEngine()
{
}


