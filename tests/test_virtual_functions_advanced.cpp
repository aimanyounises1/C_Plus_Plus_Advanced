/*
 * ==================================================================================================
 * TEST SUITE: Advanced Virtual Functions
 * ==================================================================================================
 * DO NOT MODIFY THIS FILE!
 * ==================================================================================================
 */

#include "../exercises/phase2_intermediate/oop_advanced/03_virtual_functions_advanced.cpp"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <typeinfo>

using namespace std;

// Test tracking
int totalTests = 0;
int passedTests = 0;
int totalPoints = 0;
int earnedPoints = 0;

// Colors
const string GREEN = "\033[1;32m";
const string RED = "\033[1;31m";
const string YELLOW = "\033[1;33m";
const string BLUE = "\033[1;34m";
const string RESET = "\033[0m";
const string BOLD = "\033[1m";

void printHeader(const string& title) {
    cout << "\n" << BLUE << BOLD << "╔══════════════════════════════════════════════════════════════╗" << RESET << endl;
    cout << BLUE << BOLD << "║ " << setw(60) << left << title << " ║" << RESET << endl;
    cout << BLUE << BOLD << "╚══════════════════════════════════════════════════════════════╝" << RESET << endl;
}

void testCase(const string& testName, bool passed, int points) {
    totalTests++;
    totalPoints += points;

    if (passed) {
        passedTests++;
        earnedPoints += points;
        cout << GREEN << "✓ " << RESET << testName << " (" << points << " points)" << endl;
    } else {
        cout << RED << "✗ " << RESET << testName << " (" << points << " points)" << endl;
    }
}

/*
 * PROBLEM 1 TESTS: Plugin System
 */
void testPluginSystem() {
    printHeader("Problem 1: Plugin System (40 points)");

    try {
        // Test 1: MatrixMultiply (8 points)
        MatrixMultiply* mm = new MatrixMultiply();
        testCase("MatrixMultiply creation", mm != nullptr, 2);
        testCase("MatrixMultiply getName", mm->getName() == "Matrix Multiplication", 2);
        testCase("MatrixMultiply getMemory", mm->getRequiredMemoryMB() == 512, 2);

        vector<float> input = {1.0, 2.0, 3.0};
        vector<float> output;
        mm->execute(input, output);
        testCase("MatrixMultiply execute", output.size() == 3 && output[0] == 2.0 && output[1] == 4.0 && output[2] == 6.0, 2);

        // Test 2: FFT (8 points)
        FFT* fft = new FFT();
        testCase("FFT creation", fft != nullptr, 2);
        testCase("FFT getName", fft->getName() == "Fast Fourier Transform", 2);
        testCase("FFT getMemory", fft->getRequiredMemoryMB() == 256, 2);

        output.clear();
        fft->execute(input, output);
        testCase("FFT execute", output.size() == 3 && output[0] == 2.0 && output[1] == 3.0 && output[2] == 4.0, 2);

        // Test 3: Convolution (8 points)
        Convolution* conv = new Convolution();
        testCase("Convolution creation", conv != nullptr, 2);
        testCase("Convolution getName", conv->getName() == "Convolution", 2);
        testCase("Convolution getMemory", conv->getRequiredMemoryMB() == 1024, 2);

        output.clear();
        conv->execute(input, output);
        testCase("Convolution execute", output.size() == 3 && output[0] == 1.0 && output[1] == 4.0 && output[2] == 9.0, 2);

        // Test 4: Polymorphic behavior (8 points)
        Algorithm* algo = mm;
        testCase("Polymorphic getName (MatrixMultiply)", algo->getName() == "Matrix Multiplication", 3);

        algo = fft;
        testCase("Polymorphic getName (FFT)", algo->getName() == "Fast Fourier Transform", 3);

        algo = conv;
        testCase("Polymorphic execution", algo->getRequiredMemoryMB() == 1024, 2);

        // Test 5: printInfo method (8 points)
        // Note: This would print to cout, so we just verify it exists via call
        mm->printInfo();
        fft->printInfo();
        conv->printInfo();
        testCase("printInfo method callable", true, 8);

        // Cleanup
        delete mm;
        delete fft;
        delete conv;

    } catch (const exception& e) {
        cout << RED << "Exception: " << e.what() << RESET << endl;
        testCase("Plugin System Exception Test", false, 40);
    }
}

/*
 * PROBLEM 2 TESTS: GPU Resource Management
 */
void testGPUResources() {
    printHeader("Problem 2: GPU Resource Management (35 points)");

    try {
        // Test 1: GPUBuffer (10 points)
        GPUBuffer* buffer = new GPUBuffer(1024);
        testCase("GPUBuffer creation", buffer != nullptr, 2);
        testCase("GPUBuffer getType", buffer->getType() == "GPU Buffer", 2);
        testCase("GPUBuffer getSize", buffer->getSize() == 1024, 2);

        buffer->allocate();
        buffer->deallocate();
        testCase("GPUBuffer allocate/deallocate", true, 2);

        GPUResource* res = buffer;
        testCase("GPUBuffer polymorphic behavior", res->getType() == "GPU Buffer", 2);

        // Test 2: GPUTexture (10 points)
        GPUTexture* texture = new GPUTexture(2048, 1024, 768);
        testCase("GPUTexture creation", texture != nullptr, 2);
        testCase("GPUTexture getType", texture->getType() == "GPU Texture", 2);
        testCase("GPUTexture getSize", texture->getSize() == 2048, 2);

        texture->allocate();
        texture->deallocate();
        testCase("GPUTexture allocate/deallocate", true, 2);

        res = texture;
        testCase("GPUTexture polymorphic behavior", res->getType() == "GPU Texture", 2);

        // Test 3: GPUArray (10 points)
        GPUArray* array = new GPUArray(4096);
        testCase("GPUArray creation", array != nullptr, 2);
        testCase("GPUArray getType", array->getType() == "GPU Array", 2);
        testCase("GPUArray getSize", array->getSize() == 4096, 2);

        array->allocate();
        array->deallocate();
        testCase("GPUArray allocate/deallocate", true, 2);

        res = array;
        testCase("GPUArray polymorphic behavior", res->getType() == "GPU Array", 2);

        // Test 4: Virtual destructor test (5 points)
        // Create through base pointer and delete - should call derived destructor
        res = new GPUBuffer(512);
        delete res;  // With virtual destructor, this should work correctly
        testCase("Virtual destructor (GPUBuffer)", true, 2);

        res = new GPUArray(1024);
        delete res;  // Critical test - without virtual destructor, memory leak!
        testCase("Virtual destructor (GPUArray)", true, 3);

    } catch (const exception& e) {
        cout << RED << "Exception: " << e.what() << RESET << endl;
        testCase("GPU Resource Exception Test", false, 35);
    }
}

/*
 * PROBLEM 3 TESTS: Compression Strategy
 */
void testCompressionStrategy() {
    printHeader("Problem 3: Compression Strategy (25 points)");

    try {
        // Test 1: LZ77Compression (10 points)
        LZ77Compression* lz77 = new LZ77Compression();
        testCase("LZ77Compression creation", lz77 != nullptr, 2);
        testCase("LZ77 getAlgorithmName", lz77->getAlgorithmName() == "LZ77", 2);

        vector<uint8_t> data = {1, 2, 3, 4, 5, 6, 7, 8};
        vector<uint8_t> compressed = lz77->compress(data);
        testCase("LZ77 compress (half size)", compressed.size() == 4, 3);

        vector<uint8_t> decompressed = lz77->decompress(compressed);
        testCase("LZ77 decompress (double size)", decompressed.size() == 8, 3);

        // Test 2: HuffmanCompression (10 points)
        HuffmanCompression* huffman = new HuffmanCompression();
        testCase("HuffmanCompression creation", huffman != nullptr, 2);
        testCase("Huffman getAlgorithmName", huffman->getAlgorithmName() == "Huffman", 2);

        compressed = huffman->compress(data);
        testCase("Huffman compress (1/3 size)", compressed.size() == 2 || compressed.size() == 3, 3);

        decompressed = huffman->decompress(compressed);
        testCase("Huffman decompress (triple size)", decompressed.size() >= 6, 3);

        // Test 3: Polymorphic behavior (3 points)
        CompressionStrategy* strategy = lz77;
        testCase("Polymorphic compression (LZ77)", strategy->getAlgorithmName() == "LZ77", 2);

        strategy = huffman;
        testCase("Polymorphic compression (Huffman)", strategy->getAlgorithmName() == "Huffman", 1);

        // Test 4: calculateCompressionRatio (2 points)
        double ratio = lz77->calculateCompressionRatio(100, 50);
        testCase("calculateCompressionRatio", ratio == 2.0, 2);

        // Cleanup
        delete lz77;
        delete huffman;

    } catch (const exception& e) {
        cout << RED << "Exception: " << e.what() << RESET << endl;
        testCase("Compression Strategy Exception Test", false, 25);
    }
}

/*
 * BONUS TESTS: Multiple Inheritance
 */
void testMultipleInheritance() {
    printHeader("Bonus: Multiple Inheritance (20 points)");

    try {
        // Test 1: NetworkPacket creation (5 points)
        NetworkPacket* packet = new NetworkPacket(123, "Hello World");
        testCase("NetworkPacket creation", packet != nullptr, 2);
        testCase("NetworkPacket getId", packet->getId() == 123, 2);
        testCase("NetworkPacket getPayload", packet->getPayload() == "Hello World", 1);

        // Test 2: ISerializable interface (5 points)
        string serialized = packet->serialize();
        testCase("Serialize format", serialized == "PKT:123:Hello World", 3);

        packet->deserialize("PKT:456:Test");
        testCase("Deserialize parsing", packet->getId() == 456 && packet->getPayload() == "Test", 2);

        // Test 3: ILoggable interface (5 points)
        testCase("getLogPrefix format", packet->getLogPrefix() == "[PacketID:456]", 3);
        packet->log("Test message");  // Just verify it's callable
        testCase("log method callable", true, 2);

        // Test 4: Polymorphic access through interfaces (5 points)
        ISerializable* serializable = packet;
        testCase("ISerializable interface access", serializable->serialize() == "PKT:456:Test", 3);

        ILoggable* loggable = packet;
        testCase("ILoggable interface access", loggable->getLogPrefix() == "[PacketID:456]", 2);

        // Cleanup
        delete packet;

    } catch (const exception& e) {
        cout << RED << "Exception: " << e.what() << RESET << endl;
        testCase("Multiple Inheritance Exception Test", false, 20);
    }
}

void printSummary() {
    cout << "\n" << BOLD << "═══════════════════════════════════════════════════════════════" << RESET << endl;
    cout << BOLD << "                       TEST SUMMARY                             " << RESET << endl;
    cout << BOLD << "═══════════════════════════════════════════════════════════════" << RESET << endl;

    double percentage = (totalPoints > 0) ? (100.0 * earnedPoints / totalPoints) : 0.0;

    cout << "\nTests Passed: " << passedTests << "/" << totalTests << endl;
    cout << "Points Earned: " << earnedPoints << "/" << totalPoints << endl;
    cout << "Percentage: " << fixed << setprecision(1) << percentage << "%" << endl;

    string grade, status, color;
    if (percentage >= 90) {
        grade = "A";
        status = "Excellent!";
        color = GREEN;
    } else if (percentage >= 80) {
        grade = "B";
        status = "Good!";
        color = GREEN;
    } else if (percentage >= 70) {
        grade = "C";
        status = "Satisfactory";
        color = YELLOW;
    } else if (percentage >= 60) {
        grade = "D";
        status = "Needs Improvement";
        color = YELLOW;
    } else {
        grade = "F";
        status = "Incomplete";
        color = RED;
    }

    cout << "\n" << color << BOLD << "Grade: " << grade << " - " << status << RESET << endl;

    cout << "\n" << BOLD << "Score Breakdown:" << RESET << endl;
    cout << "  Plugin System:           Problem 1 (40 points)" << endl;
    cout << "  GPU Resource Management: Problem 2 (35 points)" << endl;
    cout << "  Compression Strategy:    Problem 3 (25 points)" << endl;
    cout << "  Multiple Inheritance:    Bonus     (20 points)" << endl;

    cout << "\n" << BOLD << "═══════════════════════════════════════════════════════════════" << RESET << endl;
}

int main() {
    cout << BOLD << "\n╔═══════════════════════════════════════════════════════════════╗" << RESET << endl;
    cout << BOLD << "║     ADVANCED VIRTUAL FUNCTIONS - AUTOMATED TEST SUITE         ║" << RESET << endl;
    cout << BOLD << "╚═══════════════════════════════════════════════════════════════╝" << RESET << endl;

    testPluginSystem();
    testGPUResources();
    testCompressionStrategy();
    testMultipleInheritance();

    printSummary();

    return (earnedPoints == totalPoints) ? 0 : 1;
}