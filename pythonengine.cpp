// This is an open source non-commercial project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <QDebug>
#include <QLoggingCategory>
#include <Resonance/protocol.cpp>
#include <Resonance/rtc.cpp>
#include <Resonance/scriptengineinterface.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <map>
#include <vector>

#ifdef LIBRARY_HACK
#include <dlfcn.h>
#endif

namespace py = pybind11;

using namespace Resonance::R3;

using Resonance::RTC;

static InterfacePointers ip;
static const char* engineNameString = "python " BOOST_PP_STRINGIZE(PY_MAJOR_VERSION) "." BOOST_PP_STRINGIZE(PY_MINOR_VERSION);
static const char* engineInitString = "import resonance\n";
static const char* engineCodeString = R"(from resonance import *
createOutput(input(0), 'out')
)";
void logError(const std::string& msg)
{
    ip.logError(msg.data(), msg.size());
}

struct QueueMember
{
    enum Type { sendBlockToStream, createOutputStream, startTimer, stopTimer };

    QueueMember(const Type _type, const py::object _args)
        : type(_type)
        , args(_args)
    {}

    const Type type;
    const py::object args;
};

typedef struct
{
    int id;
    Thir::SerializedData::rid type;
} OutputStreamDescription;

struct InputStreamDescription
{
    int id;
    Thir::SerializedData::rid type;
    py::object si;
};

static std::map<int, OutputStreamDescription> outputs;
static std::map<int, InputStreamDescription> inputs;
static std::vector<QueueMember> queue;

static py::function callback_si_channels;
static py::function callback_si_event;
static py::function callback_db_event;
static py::function callback_db_channels;
static py::function callback_on_prepare;
static py::function callback_on_data_block;
static py::function callback_on_start;
static py::function callback_on_stop;
static py::function callback_trace;

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

bool mod_addToQueue(const std::string& name, py::object data)
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
        // PyErr_BadInternalCall(); @todo: ???
        return false;
    }
}

void mod_do_nothing(py::object) {}

void mod_register_callbacks(py::function new_callback_on_prepare,
                            py::function new_callback_on_data_block,
                            py::function new_callback_on_start,
                            py::function new_callback_on_stop,
                            py::function new_callback_si_channels,
                            py::function new_callback_si_event,
                            py::function new_callback_db_event,
                            py::function new_callback_db_channels)
{
    callback_on_prepare = new_callback_on_prepare;
    callback_on_data_block = new_callback_on_data_block;
    callback_on_start = new_callback_on_start;
    callback_on_stop = new_callback_on_stop;
    callback_si_channels = new_callback_si_channels;
    callback_si_event = new_callback_si_event;
    callback_db_event = new_callback_db_event;
    callback_db_channels = new_callback_db_channels;
}

void mod_register_callbacks(py::function new_callback_on_prepare,
                            py::function new_callback_on_data_block,
                            py::function new_callback_on_start,
                            py::function new_callback_on_stop,
                            py::function new_callback_si_channels,
                            py::function new_callback_si_event,
                            py::function new_callback_db_event,
                            py::function new_callback_db_channels,
                            py::function new_callback_trace)
{
    mod_register_callbacks(new_callback_on_prepare,
                           new_callback_on_data_block,
                           new_callback_on_start,
                           new_callback_on_stop,
                           new_callback_si_channels,
                           new_callback_si_event,
                           new_callback_db_event,
                           new_callback_db_channels);
    callback_trace = py::reinterpret_borrow<py::function>(new_callback_trace);
}

#define RESONANCE_PACKAGE_RUNTIME_NAME resonate
constexpr auto do_nothing = "do_nothing";

void (*register_callbacks_9)(py::function,
                             py::function,
                             py::function,
                             py::function,
                             py::function,
                             py::function,
                             py::function,
                             py::function,
                             py::function)
    = &mod_register_callbacks;
void (*register_callbacks_8)(py::function,
                             py::function,
                             py::function,
                             py::function,
                             py::function,
                             py::function,
                             py::function,
                             py::function)
    = &mod_register_callbacks;

Q_LOGGING_CATEGORY(pythonLogs, "PYTHON")

PYBIND11_EMBEDDED_MODULE(RESONANCE_PACKAGE_RUNTIME_NAME, m)
{
    m.def("add_to_queue", &mod_addToQueue);
    m.def("register_callbacks", register_callbacks_8);
    m.def("register_callbacks", register_callbacks_9);
    m.def(do_nothing, &mod_do_nothing);

    struct StdIOHook
    {};

    py::class_<StdIOHook> stdout(m, "stdout");
    stdout.def_static("write", [](const py::object& buffer) {
        qCDebug(pythonLogs) << buffer.cast<std::string>().c_str();
    });
    stdout.def_static("flush", []() {});

    // py::class_<StdIOHook> stderr(m, "stderr");
    // stderr.def_static("write", [](const py::object& buffer) {
    //     qCDebug(pythonLogs) << buffer.cast<std::string>().c_str();
    // });
    // stderr.def_static("flush", []() {});
}

void handle(const std::string& context)
try {
    throw;
} catch (const py::error_already_set& error) {
    // Extract error message
    const std::string strErrorMessage = py::str(error.value());
    logError(strErrorMessage + " during " + context);
} catch (const std::exception& exception) {
    logError(std::string(exception.what()) + " during " + context);
} catch (...) {
    logError("Unknown failure during " + context);
}

bool initializeEngine(InterfacePointers _ip, const char* code, size_t code_length)
{
    try {
        ip = _ip;

#ifdef LIBRARY_HACK
        dlopen("lib" BOOST_PP_STRINGIZE(LIBRARY_HACK) ".so", RTLD_LAZY | RTLD_GLOBAL);
#endif
        // Py_SetProgramName(L"pythonEngine");
        py::initialize_interpreter(false, 0, nullptr, false);
        auto pkg = py::module::import(BOOST_PP_STRINGIZE(RESONANCE_PACKAGE_RUNTIME_NAME));

        callback_on_prepare = pkg.attr(do_nothing);

        callback_on_data_block = py::reinterpret_borrow<py::function>(callback_on_prepare);
        callback_on_start = py::reinterpret_borrow<py::function>(callback_on_prepare);
        callback_on_stop = py::reinterpret_borrow<py::function>(callback_on_prepare);
        callback_si_channels = py::reinterpret_borrow<py::function>(callback_on_prepare);
        callback_si_event = py::reinterpret_borrow<py::function>(callback_on_prepare);
        callback_db_event = py::reinterpret_borrow<py::function>(callback_on_prepare);
        callback_db_channels = py::reinterpret_borrow<py::function>(callback_on_prepare);
        callback_trace = py::reinterpret_borrow<py::function>(callback_on_prepare);

        auto sys = py::module::import("sys");

        sys.attr("stdout") = pkg.attr("stdout");
        // sys.attr("stderr") = pkg.attr("stderr");

        py::exec(py::str(code, code_length));

        return true;
    } catch (...) {
        handle("initialize");
        return false;
    }
}

void freeEngine()
{
    try {
        // Py_Finalize(); disabled because of bug in boost
        py::finalize_interpreter();
    } catch (...) {
        handle("free");
    }
}

bool prepareEngine(const char* code,
                   size_t codeLength,
                   const SerializedDataContainer* const streams,
                   size_t streamCount)
{
    try {
        outputs.clear();
        inputs.clear();
        queue.clear();

        py::list inputList;
        for (uint32_t i = 0; i < streamCount; ++i) {
            Thir::SerializedData data(static_cast<const char*>(streams[i].data), streams[i].size);
            // data.extractString<ConnectionHeaderContainer::name>()
            Thir::SerializedData type = data.field<ConnectionHeaderContainer::type>();

            switch (type.id()) {
            // case ConnectionHeader_Int32::ID:
            // case ConnectionHeader_Int64::ID:
            case ConnectionHeader_Float64::ID: {
                const auto name = data.field<ConnectionHeaderContainer::name>().value();

                py::object si = callback_si_channels(
                    type.field<ConnectionHeader_Float64::channels>().value(),
                    type.field<ConnectionHeader_Float64::samplingRate>().value(),
                    i + 1,
                    py::str(name.c_str(), name.size()));

                if (si.is_none())
                    throw std::runtime_error("Failed to construct Float64 stream definition");

                si.attr("online") = true;

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
    } catch (...) {
        handle("prepare");
        return false;
    }
}

void blockReceived(const int id, const SerializedDataContainer block)
{
    try {
        auto iterator = inputs.find(id);
        if (iterator == inputs.end()) {
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

            const int samples = data.field<Float64::samples>().value();
            const int channels = vec.size() / samples;
            py::array_t<double> pyData({samples, channels},
                                       {sizeof(double) * channels, sizeof(double)},
                                       vec.data());

            auto block = callback_db_channels(is.si, data.field<Float64::created>().value(), pyData);
            callback_on_data_block(block);
        } break;
        }

        pythonParceQueue();
    } catch (...) {
        handle("blockReceived");
    }
}

void startEngine()
{
    try {
        callback_on_start();
    } catch (...) {
        handle("startEngine");
    }
}

void stopEngine()
{
    try {
        callback_on_stop();
    } catch (...) {
        handle("stopEngine");
    }
}

void onTimer(const int /*id*/, const uint64_t /*time*/)
{
    try {
    } catch (...) {
        handle("onTimer");
    }
}

bool pythonParceQueue()
{
    for (const QueueMember& event : queue) {
        switch (event.type) {
        case QueueMember::sendBlockToStream: {
            py::list args = event.args;
            py::object streamInfo = args[0];
            auto data = args[1];

            const int id = streamInfo.attr("id").cast<int>();

            auto os = outputs[id];

            switch (os.type) {
            case Message::ID: {
                const auto arr = data.cast<py::array>();
                if (py::isinstance<py::object>(arr.dtype())) {
                    throw std::runtime_error(
                        std::string("Can't unpack arguments for sendBlockToStream [event type "
                                    "is wrong = ")
                        + py::repr(data).cast<std::string>() + "]");
                }
                for (int i = 0; i < arr.shape(0); ++i) {
                    auto block = Message::create()
                                     .set(RTC::now())
                                     .set(0)
                                     .set(arr(i).cast<std::string>())
                                     .finish();

                    ip.sendBlock(os.id, SerializedDataContainer({block->data(), block->size()}));
                }
            } break;
            case Float64::ID: {
                const auto arr = data.cast<py::array_t<double>>();
                if (arr.ndim() != 2) {
                    throw std::runtime_error(
                        "Can't unpack arguments for sendBlockToStream [channels dim is wrong]");
                }

                int rows = arr.shape(0);
                if (rows > 0) {
                    const double* first = arr.data(0);
                    const double* last = first + arr.shape(0) * arr.shape(1);

                    auto block = Float64::create()
                                     .set(RTC::now())
                                     .set(0)
                                     .set(rows)
                                     .add(first, last)
                                     .finish()
                                     .finish();

                    ip.sendBlock(os.id, SerializedDataContainer({block->data(), block->size()}));
                }

            } break;
            }
        } break;
        case QueueMember::createOutputStream: {
            const int id = event.args.attr("id").cast<int>();
            const std::string name = event.args.attr("name").cast<std::string>();
            py::object type = event.args.attr("_source");

            if (reinterpret_cast<PyObject*>(type.ptr()->ob_type) == callback_si_event.ptr()) {
                auto type = ConnectionHeader_Message::create().next().finish();
                int sendId = ip.declareStream(name.c_str(),
                                              SerializedDataContainer(
                                                  {type->data(), (uint32_t) type->size()}));
                if (sendId != -1) {
                    outputs[id] = {sendId, Message::ID};
                }
            } else if (reinterpret_cast<PyObject*>(type.ptr()->ob_type)
                       == callback_si_channels.ptr()) {
                double samplingRate = type.attr("samplingRate").cast<double>();
                int channels = type.attr("channels").cast<int>();

                auto type
                    = ConnectionHeader_Float64::create().set(channels).set(samplingRate).finish();
                int sendId = ip.declareStream(name.c_str(),
                                              SerializedDataContainer(
                                                  {type->data(), (uint32_t) type->size()}));

                if (sendId != -1) {
                    outputs[id] = {sendId, Float64::ID};
                }
            }
        } break;
        case QueueMember::startTimer: {
            py::list args = event.args;
            long int id = args[0].cast<long int>();
            long int timeout = args[1].cast<long int>();
            bool singleShot = args[2].cast<bool>();

            ip.startTimer(id, timeout, singleShot);

        } break;
        case QueueMember::stopTimer: {
            py::list args = event.args;
            long int id = args[0].cast<long int>();
            ip.stopTimer(id);

        } break;
        }
    }
    queue.clear();
    return true;
}
