#include "pythonengine.cpp"
#include <gtest/gtest.h>

char initCode[] = " ";

static int declareStream(const char* name, const SerializedDataContainer type)
{
    return 0; //declareStreamCall(name, type);
}

static bool sendBlock(const int id, const SerializedDataContainer block)
{
    //block_id = id;
    //block_data = std::make_shared<Resonance::R3::Thir::SerializedData>(block.data, block.size, true);
    return true;
}

class TestPythonEngine : public testing::Test {
public:
    static void SetUpTestCase()
    {
        InterfacePointers interfacePointers;

        initializeEngine(interfacePointers, reinterpret_cast<char*>(initCode), sizeof(initCode));
    }

    void SetUp()
    {
        inputs.clear();
        outputs.clear();
        queue.clear();
    }
};

TEST_F(TestPythonEngine, MustStoreData)
{
    PyRun_SimpleString("import resonate\nresonate.add_to_queue('sendBlockToStream', 3)");

    ASSERT_EQ(queue.size(), 1);
    ASSERT_TRUE(PyInt_Check(queue[0].args));
    ASSERT_EQ(PyInt_AS_LONG(queue[0].args), 3);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
