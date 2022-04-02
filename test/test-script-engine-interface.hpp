#include <Resonance/protocol.cpp>
#include <Resonance/scriptengineinterface.hpp>
#include <gtest/gtest.h>
#include <string>
#include <vector>


class Test_ScriptEngineInterface : public ::testing::Test {
public:
    const std::string uuid = "{33221100-5544-6677-8899-aabbccddeeff}";
    const std::string testCode = "";

    const int channels_streamId = 1;
    const std::string channels_name = "channels";
    const unsigned int channels_channes = 3;
    const unsigned int channels_samples = 63;
    const double channels_samplingRate = 21;
    const unsigned long channels_timestamp = 946729805250000000;

    const int event_streamId = 2;
    const std::string event_name = "event";
    const std::string event_data = "test";
    const std::string event_data_out = "test out";
    const unsigned long event_timestamp = 946729905250000000;

    const std::vector<double> inputChannels = {
        0.29475517441090415, 0.29475517441090415, 0.29475517441090415,
        0.56332005806362195, 0.56332005806362195, 0.56332005806362195,
        0.7818314824680298, 0.7818314824680298, 0.7818314824680298,
        0.93087374864420414,
        0.93087374864420414, 0.93087374864420414, 0.99720379718118013,
        0.99720379718118013, 0.99720379718118013, 0.97492791218182362,
        0.97492791218182362, 0.97492791218182362, 0.86602540378443871,
        0.86602540378443871, 0.86602540378443871, 0.68017273777091969,
        0.68017273777091969, 0.68017273777091969, 0.43388373911755823,
        0.43388373911755823, 0.43388373911755823, 0.14904226617617472,
        0.14904226617617472, 0.14904226617617472, -0.14904226617617447,
        -0.14904226617617447, -0.14904226617617447, -0.43388373911755801,
        -0.43388373911755801, -0.43388373911755801, -0.68017273777091947,
        -0.68017273777091947, -0.68017273777091947, -0.86602540378443837,
        -0.86602540378443837, -0.86602540378443837, -0.97492791218182362,
        -0.97492791218182362, -0.97492791218182362, -0.99720379718118024,
        -0.99720379718118024, -0.99720379718118024, -0.93087374864420447,
        -0.93087374864420447, -0.93087374864420447, -0.78183148246802991,
        -0.78183148246802991, -0.78183148246802991, -0.56332005806362195,
        -0.56332005806362195, -0.56332005806362195, -0.29475517441090471,
        -0.29475517441090471, -0.29475517441090471, -2.4492935982947064e-16,
        -2.4492935982947064e-16, -2.4492935982947064e-16, 0.29475517441090426,
        0.29475517441090426, 0.29475517441090426, 0.56332005806362229,
        0.56332005806362229, 0.56332005806362229, 0.78183148246802958,
        0.78183148246802958, 0.78183148246802958, 0.93087374864420425,
        0.93087374864420425, 0.93087374864420425, 0.99720379718118013,
        0.99720379718118013, 0.99720379718118013, 0.97492791218182373,
        0.97492791218182373, 0.97492791218182373, 0.86602540378443915,
        0.86602540378443915, 0.86602540378443915, 0.68017273777091924,
        0.68017273777091924, 0.68017273777091924, 0.43388373911755845,
        0.43388373911755845, 0.43388373911755845, 0.14904226617617364,
        0.14904226617617364, 0.14904226617617364, -0.14904226617617292,
        -0.14904226617617292, -0.14904226617617292, -0.43388373911755779,
        -0.43388373911755779, -0.43388373911755779, -0.68017273777091869,
        -0.68017273777091869, -0.68017273777091869, -0.86602540378443871,
        -0.86602540378443871, -0.86602540378443871, -0.97492791218182351,
        -0.97492791218182351, -0.97492791218182351, -0.99720379718118024,
        -0.99720379718118024, -0.99720379718118024, -0.93087374864420425,
        -0.93087374864420425, -0.93087374864420425, -0.78183148246803014,
        -0.78183148246803014, -0.78183148246803014, -0.56332005806362295,
        -0.56332005806362295, -0.56332005806362295, -0.29475517441090582,
        -0.29475517441090582, -0.29475517441090582, -4.8985871965894128e-16,
        -4.8985871965894128e-16, -4.8985871965894128e-16, 0.29475517441090315,
        0.29475517441090315, 0.29475517441090315, 0.56332005806362206,
        0.56332005806362206, 0.56332005806362206, 0.78183148246802947,
        0.78183148246802947, 0.78183148246802947, 0.93087374864420447,
        0.93087374864420447, 0.93087374864420447, 0.99720379718118013,
        0.99720379718118013, 0.99720379718118013, 0.97492791218182373,
        0.97492791218182373, 0.97492791218182373, 0.86602540378443837,
        0.86602540378443837, 0.86602540378443837, 0.68017273777091936,
        0.68017273777091936, 0.68017273777091936, 0.43388373911756023,
        0.43388373911756023, 0.43388373911756023, 0.14904226617617389,
        0.14904226617617389, 0.14904226617617389, -0.14904226617617267,
        -0.14904226617617267, -0.14904226617617267, -0.43388373911755757,
        -0.43388373911755757, -0.43388373911755757, -0.6801727377709198,
        -0.6801727377709198, -0.6801727377709198, -0.86602540378443771,
        -0.86602540378443771, -0.86602540378443771, -0.97492791218182351,
        -0.97492791218182351, -0.97492791218182351, -0.99720379718118013,
        -0.99720379718118013, -0.99720379718118013, -0.93087374864420491,
        -0.93087374864420491, -0.93087374864420491, -0.78183148246803025,
        -0.78183148246803025, -0.78183148246803025, -0.56332005806362462,
        -0.56332005806362462, -0.56332005806362462, -0.29475517441090265,
        -0.29475517441090265, -0.29475517441090265, -7.3478807948841188e-16,
        -7.3478807948841188e-16, -7.3478807948841188e-16
    };

    const std::vector<double> outputChannels = {
        0.29475517441090415, -0.29475517441090415, 0.29475517441090415,
        0.56332005806362195, -0.56332005806362195, 0.56332005806362195,
        0.7818314824680298, -0.7818314824680298, 0.7818314824680298,
        0.93087374864420414, -0.93087374864420414, 0.93087374864420414,
        0.99720379718118013, -0.99720379718118013, 0.99720379718118013,
        0.97492791218182362, -0.97492791218182362, 0.97492791218182362,
        0.86602540378443871, -0.86602540378443871, 0.86602540378443871,
        0.68017273777091969, -0.68017273777091969, 0.68017273777091969,
        0.43388373911755823, -0.43388373911755823, 0.43388373911755823,
        0.14904226617617472, -0.14904226617617472, 0.14904226617617472,
        -0.14904226617617447, 0.14904226617617447, -0.14904226617617447,
        -0.43388373911755801, 0.43388373911755801, -0.43388373911755801,
        -0.68017273777091947, 0.68017273777091947, -0.68017273777091947,
        -0.86602540378443837, 0.86602540378443837, -0.86602540378443837,
        -0.97492791218182362, 0.97492791218182362, -0.97492791218182362,
        -0.99720379718118024, 0.99720379718118024, -0.99720379718118024,
        -0.93087374864420447, 0.93087374864420447, -0.93087374864420447,
        -0.78183148246802991, 0.78183148246802991, -0.78183148246802991,
        -0.56332005806362195, 0.56332005806362195, -0.56332005806362195,
        -0.29475517441090471, 0.29475517441090471, -0.29475517441090471,
        -2.4492935982947064e-16, 2.4492935982947064e-16, -2.4492935982947064e-16,
        0.29475517441090426, -0.29475517441090426, 0.29475517441090426,
        0.56332005806362229, -0.56332005806362229, 0.56332005806362229,
        0.78183148246802958, -0.78183148246802958, 0.78183148246802958,
        0.93087374864420425, -0.93087374864420425, 0.93087374864420425,
        0.99720379718118013, -0.99720379718118013, 0.99720379718118013,
        0.97492791218182373, -0.97492791218182373, 0.97492791218182373,
        0.86602540378443915, -0.86602540378443915, 0.86602540378443915,
        0.68017273777091924, -0.68017273777091924, 0.68017273777091924,
        0.43388373911755845, -0.43388373911755845, 0.43388373911755845,
        0.14904226617617364, -0.14904226617617364, 0.14904226617617364,
        -0.14904226617617292, 0.14904226617617292, -0.14904226617617292,
        -0.43388373911755779, 0.43388373911755779, -0.43388373911755779,
        -0.68017273777091869, 0.68017273777091869, -0.68017273777091869,
        -0.86602540378443871, 0.86602540378443871, -0.86602540378443871,
        -0.97492791218182351, 0.97492791218182351, -0.97492791218182351,
        -0.99720379718118024, 0.99720379718118024, -0.99720379718118024,
        -0.93087374864420425, 0.93087374864420425, -0.93087374864420425,
        -0.78183148246803014, 0.78183148246803014, -0.78183148246803014,
        -0.56332005806362295, 0.56332005806362295, -0.56332005806362295,
        -0.29475517441090582, 0.29475517441090582, -0.29475517441090582,
        -4.8985871965894128e-16, 4.8985871965894128e-16, -4.8985871965894128e-16,
        0.29475517441090315, -0.29475517441090315, 0.29475517441090315,
        0.56332005806362206, -0.56332005806362206, 0.56332005806362206,
        0.78183148246802947, -0.78183148246802947, 0.78183148246802947,
        0.93087374864420447, -0.93087374864420447, 0.93087374864420447,
        0.99720379718118013, -0.99720379718118013, 0.99720379718118013,
        0.97492791218182373, -0.97492791218182373, 0.97492791218182373,
        0.86602540378443837, -0.86602540378443837, 0.86602540378443837,
        0.68017273777091936, -0.68017273777091936, 0.68017273777091936,
        0.43388373911756023, -0.43388373911756023, 0.43388373911756023,
        0.14904226617617389, -0.14904226617617389, 0.14904226617617389,
        -0.14904226617617267, 0.14904226617617267, -0.14904226617617267,
        -0.43388373911755757, 0.43388373911755757, -0.43388373911755757,
        -0.6801727377709198, 0.6801727377709198, -0.6801727377709198,
        -0.86602540378443771, 0.86602540378443771, -0.86602540378443771,
        -0.97492791218182351, 0.97492791218182351, -0.97492791218182351,
        -0.99720379718118013, 0.99720379718118013, -0.99720379718118013,
        -0.93087374864420491, 0.93087374864420491, -0.93087374864420491,
        -0.78183148246803025, 0.78183148246803025, -0.78183148246803025,
        -0.56332005806362462, 0.56332005806362462, -0.56332005806362462,
        -0.29475517441090265, 0.29475517441090265, -0.29475517441090265,
        -7.3478807948841188e-16, 7.3478807948841188e-16, -7.3478807948841188e-16
    };

    void SetUp() override
    {
        // Set interfacePointers
        interfacePointers.declareStream = &declareStream;
        interfacePointers.sendBlock = &sendBlock;

        // Reset function callbacks

        declareStreamCall = &expectChannelsStream;
        channels_stream_data = std::make_shared<Resonance::R3::Thir::SerializedData>(static_cast<size_t>(0));
        event_stream_data = std::make_shared<Resonance::R3::Thir::SerializedData>(static_cast<size_t>(0));

        // Create channels_header_block
        auto channelHeaderContainer = Resonance::R3::ConnectionHeaderContainer::create()
                                          .set(channels_name)
                                          .set(uuid);
        channelHeaderContainer
            .beginRecursive<Resonance::R3::ConnectionHeader_Float64>()
            .set(channels_channes)
            .set(channels_samplingRate)
            .finish();
        channels_header_block = channelHeaderContainer.next().finish();

        // Create event_input_block

        auto eventHeaderContainer = Resonance::R3::ConnectionHeaderContainer::create()
                                        .set(event_name)
                                        .set(uuid);
        eventHeaderContainer
            .beginRecursive<Resonance::R3::ConnectionHeader_Message>()
            .next()
            .finish();

        event_header_block = eventHeaderContainer.next().finish();

        // Create channels_input_block

        channels_input_block = Resonance::R3::Float64::create().set(channels_timestamp).set(channels_timestamp).set(63).addVector(inputChannels).finish().finish();
        event_input_block = Resonance::R3::Message::create().set(event_timestamp).set(event_timestamp).set(event_data).finish();
    }

    Resonance::SD channels_header_block;
    Resonance::SD channels_input_block;

    Resonance::SD event_header_block;
    Resonance::SD event_input_block;

    InterfacePointers interfacePointers;

    static Resonance::SD channels_stream_data;
    static Resonance::SD event_stream_data;

    static Resonance::SD channels_block_data;
    static Resonance::SD event_block_data;

    typedef int (*declareStreamCall_t)(const char*, const SerializedDataContainer);

    static declareStreamCall_t declareStreamCall;

    static int block_id;
    static Resonance::SD block_data;

    void resetBlockExpectations()
    {
        block_id = -1;
        block_data.reset();
    }

    static int declareStream(const char* name, const SerializedDataContainer type)
    {
        return declareStreamCall(name, type);
    }

    static bool sendBlock(const int id, const SerializedDataContainer block)
    {
        block_id = id;
        block_data = std::make_shared<Resonance::R3::Thir::SerializedData>(block.data, block.size, true);
        return true;
    }

    static int expectEventStream(const char* name,
        const SerializedDataContainer type)
    {
        EXPECT_EQ(name, std::string("event-out"));
        event_stream_data = std::make_shared<Resonance::R3::Thir::SerializedData>(type.data, type.size, true);

        declareStreamCall = &lastDeclareStreamExpectation;
        return 2;
    }

    static int expectChannelsStream(const char* name,
        const SerializedDataContainer type)
    {
        EXPECT_EQ(name, std::string("channels-out"));
        channels_stream_data = std::make_shared<Resonance::R3::Thir::SerializedData>(type.data, type.size, true);

        declareStreamCall = &expectEventStream;
        return 1;
    }

    static int lastDeclareStreamExpectation(const char*,
        const SerializedDataContainer)
    {
        return 0;
    }
};

Resonance::SD Test_ScriptEngineInterface::channels_stream_data;
Resonance::SD Test_ScriptEngineInterface::event_stream_data;
Resonance::SD Test_ScriptEngineInterface::channels_block_data;
Resonance::SD Test_ScriptEngineInterface::event_block_data;
Test_ScriptEngineInterface::declareStreamCall_t Test_ScriptEngineInterface::declareStreamCall;
int Test_ScriptEngineInterface::block_id;
Resonance::SD Test_ScriptEngineInterface::block_data;

TEST_F(Test_ScriptEngineInterface, CheckName)
{
    EXPECT_EQ(engineName(), std::string(engineNameReference));
}

TEST_F(Test_ScriptEngineInterface, CheckSequence)
{

    ASSERT_TRUE(initializeEngine(interfacePointers, reinterpret_cast<char*>(initCode), sizeof(initCode)));

    SerializedDataContainer streams[] = {
        { channels_header_block->data(), channels_header_block->size() },
        { event_header_block->data(), event_header_block->size() }
    };

    ASSERT_TRUE(prepareEngine(testCode.c_str(), testCode.size(), streams,
        sizeof(streams) / sizeof(streams[0])));

    EXPECT_EQ(declareStreamCall, &Test_ScriptEngineInterface::lastDeclareStreamExpectation);

    ASSERT_EQ(channels_stream_data->id(), Resonance::R3::ConnectionHeader_Float64::ID);
    EXPECT_EQ(channels_stream_data->field<Resonance::R3::ConnectionHeader_Float64::channels>(), channels_channes);
    EXPECT_EQ(channels_stream_data->field<Resonance::R3::ConnectionHeader_Float64::samplingRate>(), channels_samplingRate);

    ASSERT_EQ(event_stream_data->id(), Resonance::R3::ConnectionHeader_Message::ID);

    startEngine();

    // send channel data

    resetBlockExpectations();
    blockReceived(channels_streamId - 1, { channels_input_block->data(), channels_input_block->size() });

    EXPECT_EQ(block_id, channels_streamId);
    ASSERT_TRUE(block_data);
    EXPECT_EQ(block_data->id(), Resonance::R3::Float64::ID);
    EXPECT_EQ(block_data->field<Resonance::R3::Float64::samples>(), channels_samples);
    EXPECT_EQ(block_data->field<Resonance::R3::Float64::data>().toVector(), outputChannels);

    // send message

    resetBlockExpectations();
    blockReceived(event_streamId - 1, { event_input_block->data(), event_input_block->size() });

    EXPECT_EQ(block_id, event_streamId);
    ASSERT_TRUE(block_data);
    ASSERT_EQ(block_data->id(), Resonance::R3::Message::ID);
    EXPECT_EQ(block_data->field<Resonance::R3::Message::message>().value(), event_data_out);

    stopEngine();

    freeEngine();
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
