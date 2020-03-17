// This is an open source non-commercial project. Dear PVS-Studio, please check it.

// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include <Resonance/scriptengineinterface.h>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <Resonance/protocol.cpp>
#include <Resonance/rtc.cpp>
#include <iostream>
#include <map>
#include <vector>

#ifdef LIBRARY_HACK
#include <dlfcn.h> 
#endif

namespace py = boost::python;
namespace np = boost::python::numpy;

using namespace Resonance::R3;

using Resonance::RTC;

static InterfacePointers ip;
static const char* engineNameString = "python " BOOST_PP_STRINGIZE(PY_MAJOR_VERSION) "." BOOST_PP_STRINGIZE(PY_MINOR_VERSION);
static const char* engineInitString = "import resonance\n";
static const char* engineCodeString = R"(from resonance import *
createOutput(input(0), 'out')
)";

struct QueueMember {
    enum Type {
        sendBlockToStream,
        createOutputStream,
        startTimer,
        stopTimer
    };
    
    QueueMember(const Type _type, const py::object _args):
        type(_type),
        args(_args){}

    const Type type;
    const py::object args;
};

typedef struct {
    int id;
    Thir::SerializedData::rid type;
} OutputStreamDescription;

struct InputStreamDescription {
    int id;
    Thir::SerializedData::rid type;
    py::object si;
};

static std::map<int, OutputStreamDescription> outputs;
static std::map<int, InputStreamDescription> inputs;
static std::vector<QueueMember> queue;

static py::object callback_si_channels;
static py::object callback_si_event;
static py::object callback_db_event;
static py::object callback_db_channels;
static py::object callback_on_prepare;
static py::object callback_on_data_block;
static py::object callback_on_start;
static py::object callback_on_stop;
static py::object callback_trace;

void trace(py::object obj)
{
    callback_trace(obj);
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

bool mod_addToQueue(py::str name, py::object data)
{

    if (std::string("sendBlockToStream") == name) {
        queue.emplace_back(QueueMember::sendBlockToStream, data);
        return true;
    } else if (std::string("createOutputStream") == name) {
        queue.emplace_back(QueueMember::createOutputStream, data);
        return true;
    } else if (std::string("startTimer") == name) {
        queue.emplace_back(QueueMember::startTimer, data);
        return true;
    } else if (std::string("stopTimer") == name) {
        queue.emplace_back(QueueMember::stopTimer, data);
        return true;
    } else {
        PyErr_BadInternalCall();
        return false;
    }
}

void mod_do_nothing(py::object )
{
}

void mod_register_callbacks(py::object new_callback_on_prepare,
                            py::object new_callback_on_data_block,
                            py::object new_callback_on_start,
                            py::object new_callback_on_stop,
                            py::object new_callback_si_channels,
                            py::object new_callback_si_event,
                            py::object new_callback_db_event,
                            py::object new_callback_db_channels)
{
    callback_on_prepare    = new_callback_on_prepare   ;
    callback_on_data_block = new_callback_on_data_block;
    callback_on_start      = new_callback_on_start     ;
    callback_on_stop       = new_callback_on_stop      ;
    callback_si_channels   = new_callback_si_channels  ;
    callback_si_event      = new_callback_si_event     ;
    callback_db_event      = new_callback_db_event     ;
    callback_db_channels   = new_callback_db_channels  ;
}

void mod_register_callbacks(py::object new_callback_on_prepare,
                            py::object new_callback_on_data_block,
                            py::object new_callback_on_start,
                            py::object new_callback_on_stop,
                            py::object new_callback_si_channels,
                            py::object new_callback_si_event,
                            py::object new_callback_db_event,
                            py::object new_callback_db_channels,
                            py::object new_callback_trace)
{
    mod_register_callbacks(
        new_callback_on_prepare   ,
        new_callback_on_data_block,
        new_callback_on_start     ,
        new_callback_on_stop      ,
        new_callback_si_channels  ,
        new_callback_si_event     ,
        new_callback_db_event     ,
        new_callback_db_channels
    );
    callback_trace = new_callback_trace;
}


#define RESONANCE_PACKAGE_RUNTIME_NAME resonate
constexpr auto do_nothing = "do_nothing";


void (*register_callbacks_9)(py::object,py::object,py::object,py::object,py::object,py::object,py::object,py::object,py::object) = &mod_register_callbacks;
void (*register_callbacks_8)(py::object,py::object,py::object,py::object,py::object,py::object,py::object,py::object) = &mod_register_callbacks;


BOOST_PYTHON_MODULE(RESONANCE_PACKAGE_RUNTIME_NAME)
{
    def("add_to_queue", &mod_addToQueue);
    def("register_callbacks", register_callbacks_8);
    def("register_callbacks", register_callbacks_9);
    def(do_nothing, &mod_do_nothing);
}

bool initializeEngine(InterfacePointers _ip, const char* code, size_t code_length)
{
    ip = _ip;
    
#ifdef LIBRARY_HACK
    dlopen("lib" BOOST_PP_STRINGIZE(LIBRARY_HACK) ".so", RTLD_LAZY | RTLD_GLOBAL);
#endif
    
    Py_SetProgramName(L"pythonEngine");
    PyImport_AppendInittab("resonate", &PyInit_resonate);
    Py_Initialize();
    np::initialize();
    
    //callback_on_prepare = py::import(BOOST_PP_STRINGIZE(RESONANCE_PACKAGE_RUNTIME_NAME)).attr(do_nothing);

    callback_on_data_block = callback_on_prepare;
    callback_on_start = callback_on_prepare;
    callback_on_stop = callback_on_prepare;
    callback_si_channels = callback_on_prepare;
    callback_si_event = callback_on_prepare;
    callback_db_event = callback_on_prepare;
    callback_db_channels = callback_on_prepare;
    callback_trace = callback_on_prepare;

    try{
        py::object main_module = py::import("__main__");
        py::object main_namespace = main_module.attr("__dict__");
        py::exec(py::str(code, code_length), main_namespace, main_namespace);
    }
    catch(py::error_already_set &e)
    {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
    
        py::handle<> hType(ptype);
        py::object extype(hType);
        py::handle<> hTraceback(ptraceback);
        py::object traceback(hTraceback);
    
        //Extract error message
        std::string strErrorMessage = py::extract<std::string>(pvalue);
        std::cout << "Error:" << strErrorMessage;
        
        return false;
    }
    
    return true;
}

void freeEngine()
{
    //Py_Finalize(); disabled because of bug in boost
}

bool prepareEngine(const char* code, size_t codeLength, const SerializedDataContainer* const streams, size_t streamCount)
{
    outputs.clear();
    inputs.clear();
    queue.clear();

    py::list inputList;
    for (uint32_t i = 0; i < streamCount; ++i) {
        Thir::SerializedData data(static_cast<const char*>(streams[i].data), streams[i].size);
        //data.extractString<ConnectionHeaderContainer::name>()
        Thir::SerializedData type = data.field<ConnectionHeaderContainer::type>();

        switch (type.id()) {
        //case ConnectionHeader_Int32::ID:
        //case ConnectionHeader_Int64::ID:
        case ConnectionHeader_Float64::ID: {
            
            const auto name = data.field<ConnectionHeaderContainer::name>().value();
            
            py::object si = callback_si_channels(
                        type.field<ConnectionHeader_Float64::channels>().value(),
                        type.field<ConnectionHeader_Float64::samplingRate>().value(),
                        i + 1,
                        py::str(name.c_str(), name.size())
                    );
            
            if (si.is_none())
                throw std::runtime_error("Failed to construct Float64 stream definition");
            
            si.attr("online")=true;
            
            inputList.append(si);

            inputs.emplace(i, InputStreamDescription{static_cast<int>(i + 1), Float64::ID, si});
        } break;

        case ConnectionHeader_Message::ID: {
            const auto name = data.field<ConnectionHeaderContainer::name>().value();
            py::object si = callback_si_event(i + 1, py::str(name.c_str(), name.size()));
            if (si.is_none())
                throw std::runtime_error("Failed to construct Message stream definition");
            
            si.attr("online") = true;
            inputList.append(si);

            inputs.emplace(i, InputStreamDescription{static_cast<int>(i + 1), Message::ID, si});
        } break;
        }
    }

    callback_on_prepare(py::str(code, codeLength), inputList);

    return pythonParceQueue();
}

void blockReceived(const int id, const SerializedDataContainer block)
{

    auto iterator = inputs.find(id);
    if(iterator == inputs.end())
    {
        throw std::runtime_error("Failed to retrieve stream information for block");
    }
    auto is = iterator->second;

    switch (is.type) {
    case Message::ID: {
        Thir::SerializedData data(block.data, block.size);

        auto block = callback_db_event(is.si,
                                 data.field<Message::created>().value(),
                                 data.field<Message::message>().value());

        callback_on_data_block(block);
    } break;
    case Float64::ID: {
        Thir::SerializedData data(block.data, block.size);

        auto vec = data.field<Float64::data>().toVector();

        int samples = data.field<Float64::samples>().value();
        const auto channels =  static_cast<Py_intptr_t>(vec.size() / samples);
        
        auto pyData = np::from_data(
                    reinterpret_cast<void*>(vec.data()), np::dtype::get_builtin<double>(), 
                    py::make_tuple(samples, channels),
                    py::make_tuple(sizeof(double)*channels, sizeof(double)),
                    py::object()).copy();

        auto block = callback_db_channels(is.si, data.field<Float64::created>().value(), pyData);
        callback_on_data_block(block);
    } break;
    }

    pythonParceQueue();
}

void startEngine()
{
    callback_on_start();
}

void stopEngine()
{
    callback_on_stop();
}

void onTimer(const int /*id*/, const uint64_t /*time*/)
{
}

bool pythonParceQueue()
{
    for (const QueueMember& event : queue) {
        switch (event.type) {
        case QueueMember::sendBlockToStream: {
            py::object streamInfo = event.args[0];
            auto data = event.args[1];
            
            int id = py::extract<int>(streamInfo.attr("id"));
            
            auto os = outputs[id];

            switch (os.type) {
            case Message::ID: {
                auto block = Message::create()
                               .set(RTC::now())
                               .set(0)
                               .set(py::extract<std::string>(data))
                               .finish();

                ip.sendBlock(os.id, SerializedDataContainer({ block->data(), block->size() }));
            } break;
            case Float64::ID: {
                np::ndarray arr = py::extract<np::ndarray>(data);
                if (arr.get_nd() != 2) {
                    throw std::runtime_error("Can't unpack arguments for sendBlockToStream [channels dim is wrong]");
                }
                if (arr.get_dtype() != np::dtype::get_builtin<double>()) {
                    throw std::runtime_error("Can't unpack arguments for sendBlockToStream [ channels type is wrong]");
                }

                int rows = arr.shape(0);

                double* first = reinterpret_cast<double*>(arr.get_data());
                double* last = first + arr.shape(0)*arr.shape(1);

                auto block
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
            int id = py::extract<int>(event.args.attr("id"));
            std::string name = py::extract<const std::string>(event.args.attr("name"));
            py::object type = event.args.attr("_source");

            if (reinterpret_cast<PyObject*>(type.ptr()->ob_type) == callback_si_event.ptr()) {
                auto type = ConnectionHeader_Message::create().next().finish();
                int sendId = ip.declareStream(name.c_str(), SerializedDataContainer({ type->data(), (uint32_t)type->size() }));
                if (sendId != -1) {
                    outputs[id] = { sendId, Message::ID };
                }
            } else if (reinterpret_cast<PyObject*>(type.ptr()->ob_type) == callback_si_channels.ptr()) {
                double samplingRate = py::extract<double>(type.attr("samplingRate"));
                int channels = py::extract<int>(type.attr("channels"));
                

                auto type = ConnectionHeader_Float64::create()
                              .set(channels)
                              .set(samplingRate)
                              .finish();
                int sendId = ip.declareStream(name.c_str(), SerializedDataContainer({ type->data(), (uint32_t)type->size() }));

                if (sendId != -1) {
                    outputs[id] = { sendId, Float64::ID };
                }
            }
        } break;
        case QueueMember::startTimer: {
            long int id = py::extract<long int>(event.args[0]);
            long int timeout = py::extract<long int>(event.args[1]);
            bool singleShot = py::extract<bool>(event.args[2]);

            ip.startTimer(id, timeout, singleShot);

        } break;
        case QueueMember::stopTimer: {
            long int id = py::extract<long int>(event.args[0]);
            ip.stopTimer(id);

        } break;
        }
    }
    queue.clear();
    return true;
}
