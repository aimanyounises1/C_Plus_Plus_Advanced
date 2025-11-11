/*
 * ==================================================================================================
 * ADVANCED EXERCISE: Virtual Functions & Memory Management
 * ==================================================================================================
 * Difficulty: Advanced | Estimated Time: 60-75 minutes
 * Points: 120 total (100 required + 20 bonus)
 *
 * CHALLENGE MODE: Implement from scratch!
 * - NO code scaffolding provided
 * - YOU design the class hierarchy
 * - YOU write all function signatures
 * - YOU figure out constructor syntax
 * - Tests will verify your implementation
 *
 * This exercise tests your understanding of:
 * ✓ Virtual destructors and memory management
 * ✓ Pure virtual vs regular virtual functions
 * ✓ Override and final keywords
 * ✓ Polymorphic behavior
 * ✓ C++ syntax (constructors, destructors, const correctness)
 * ==================================================================================================
 */

/*
 * ==================================================================================================
 * PROBLEM 1: Plugin System for GPU Algorithms (40 points)
 * ==================================================================================================
 *
 * Design a plugin architecture for GPU computation algorithms. Different algorithms can be loaded
 * at runtime and executed polymorphically.
 *
 * REQUIREMENTS:
 *
 * 1. Create an ABSTRACT class called "Algorithm" with:
 *    - Pure virtual method: string getName() const
 *    - Pure virtual method: void execute(const vector<float>& input, vector<float>& output) const
 *    - Pure virtual method: int getRequiredMemoryMB() const
 *    - Regular virtual method: void printInfo() const
 *      This should print: "Algorithm: <name>, Memory: <memory>MB"
 *    - Proper virtual destructor
 *
 * 2. Implement THREE concrete algorithm classes:
 *
 *    a) MatrixMultiply:
 *       - getName() returns "Matrix Multiplication"
 *       - getRequiredMemoryMB() returns 512
 *       - execute() multiplies each input element by 2.0 (simplified simulation)
 *
 *    b) FFT (Fast Fourier Transform):
 *       - getName() returns "Fast Fourier Transform"
 *       - getRequiredMemoryMB() returns 256
 *       - execute() adds 1.0 to each input element (simplified simulation)
 *
 *    c) Convolution:
 *       - getName() returns "Convolution"
 *       - getRequiredMemoryMB() returns 1024
 *       - execute() squares each input element (simplified simulation)
 *       - Mark this class as FINAL (cannot be inherited)
 *
 * 3. Key C++ Requirements:
 *    - Use 'const' correctly on all getter methods
 *    - Use 'override' keyword on all overridden methods
 *    - Use 'final' on the Convolution class
 *    - Proper virtual destructor in base class
 *    - Include <vector> and <string> headers
 *
 * TESTING:
 * - Tests will create Algorithm* pointers to your concrete classes
 * - Tests will verify polymorphic behavior
 * - Tests will check memory is properly cleaned up (no leaks)
 * ==================================================================================================
 */

// YOUR CODE HERE for Problem 1
// Hint: Start with the Algorithm abstract class, then implement the three concrete classes


/*
 * ==================================================================================================
 * PROBLEM 2: Resource Management with Virtual Destructors (35 points)
 * ==================================================================================================
 *
 * Implement a hierarchy of GPU resource classes that properly manage memory.
 * This tests your understanding of VIRTUAL DESTRUCTORS and proper cleanup.
 *
 * REQUIREMENTS:
 *
 * 1. Create a BASE class called "GPUResource" with:
 *    - Protected member: size_t sizeInBytes
 *    - Constructor that takes size_t size parameter
 *    - Pure virtual method: string getType() const
 *    - Pure virtual method: void allocate()
 *    - Pure virtual method: void deallocate()
 *    - Regular virtual method: size_t getSize() const (returns sizeInBytes)
 *    - CRITICAL: Virtual destructor that calls deallocate()
 *
 * 2. Implement THREE derived classes:
 *
 *    a) GPUBuffer:
 *       - Private member: void* devicePtr (initialize to nullptr)
 *       - Constructor: takes size parameter, passes to base
 *       - getType() returns "GPU Buffer"
 *       - allocate() simulates allocation by setting devicePtr = (void*)0x1000
 *       - deallocate() simulates cleanup by setting devicePtr = nullptr
 *       - Destructor should print "GPUBuffer destroyed"
 *
 *    b) GPUTexture:
 *       - Private members: int width, int height (initialize to 0)
 *       - Constructor: takes size, width, height parameters
 *       - getType() returns "GPU Texture"
 *       - allocate() simulates allocation by setting width and height
 *       - deallocate() simulates cleanup by setting width=0, height=0
 *       - Destructor should print "GPUTexture destroyed"
 *
 *    c) GPUArray:
 *       - Private member: float* data (initialize to nullptr)
 *       - Constructor: takes size parameter, allocates actual array: data = new float[size/sizeof(float)]
 *       - getType() returns "GPU Array"
 *       - allocate() does nothing (already allocated in constructor)
 *       - deallocate() frees the array: delete[] data; data = nullptr;
 *       - Destructor should print "GPUArray destroyed"
 *
 * 3. Key Points:
 *    - WITHOUT virtual destructor, derived destructors won't be called
 *    - This causes MEMORY LEAKS (especially for GPUArray)
 *    - Tests will verify proper cleanup when deleting through base pointer
 *
 * CRITICAL C++ CONCEPT:
 * ```cpp
 * GPUResource* res = new GPUArray(1024);
 * delete res;  // Without virtual destructor: only ~GPUResource called (LEAK!)
 *              // With virtual destructor: ~GPUArray then ~GPUResource called (CORRECT!)
 * ```
 * ==================================================================================================
 */

// YOUR CODE HERE for Problem 2


/*
 * ==================================================================================================
 * PROBLEM 3: Strategy Pattern with Final Methods (25 points)
 * ==================================================================================================
 *
 * Implement a compression strategy system where certain methods cannot be overridden.
 * This tests your understanding of the FINAL keyword.
 *
 * REQUIREMENTS:
 *
 * 1. Create an ABSTRACT class "CompressionStrategy" with:
 *    - Pure virtual method: string getAlgorithmName() const
 *    - Pure virtual method: vector<uint8_t> compress(const vector<uint8_t>& data) const
 *    - Pure virtual method: vector<uint8_t> decompress(const vector<uint8_t>& data) const
 *    - FINAL virtual method: double calculateCompressionRatio(size_t original, size_t compressed) const
 *      Implementation: return (double)original / compressed;
 *    - Virtual destructor
 *
 * 2. Implement TWO compression algorithms:
 *
 *    a) LZ77Compression:
 *       - getAlgorithmName() returns "LZ77"
 *       - compress() returns a vector with half the size (simulate 2x compression)
 *       - decompress() returns a vector with double the size
 *
 *    b) HuffmanCompression (mark class as FINAL):
 *       - getAlgorithmName() returns "Huffman"
 *       - compress() returns a vector with 1/3 the size (simulate 3x compression)
 *       - decompress() returns a vector with triple the size
 *       - This class CANNOT be inherited (use final keyword)
 *
 * 3. Key Requirements:
 *    - calculateCompressionRatio must be marked FINAL (cannot be overridden)
 *    - HuffmanCompression must be marked FINAL (cannot be inherited)
 *    - Use <cstdint> header for uint8_t
 *
 * WHY FINAL?
 * - Some methods implement critical logic that shouldn't be changed
 * - Prevents accidental override that could break functionality
 * - Documents design intent
 * ==================================================================================================
 */

// YOUR CODE HERE for Problem 3


/*
 * ==================================================================================================
 * BONUS PROBLEM: Multiple Inheritance with Interfaces (20 points)
 * ==================================================================================================
 *
 * Implement a class that inherits from multiple abstract interfaces.
 * This is similar to Java's interface implementation but using C++ pure virtual functions.
 *
 * REQUIREMENTS:
 *
 * 1. Create interface "ISerializable" with:
 *    - Pure virtual: string serialize() const
 *    - Pure virtual: void deserialize(const string& data)
 *    - Virtual destructor
 *
 * 2. Create interface "ILoggable" with:
 *    - Pure virtual: void log(const string& message) const
 *    - Pure virtual: string getLogPrefix() const
 *    - Virtual destructor
 *
 * 3. Create class "NetworkPacket" that implements BOTH interfaces:
 *    - Private members: int packetId, string payload
 *    - Constructor: takes int id and string payload
 *    - serialize() returns format: "PKT:<id>:<payload>"
 *    - deserialize() parses "PKT:<id>:<payload>" and sets members
 *    - log() prints: "<prefix> <message>"
 *    - getLogPrefix() returns "[PacketID:<id>]"
 *    - Getter: string getPayload() const
 *    - Getter: int getId() const
 *
 * SYNTAX NOTE:
 * In C++, multiple inheritance syntax:
 * ```cpp
 * class NetworkPacket : public ISerializable, public ILoggable {
 *     // Must implement ALL pure virtual methods from both interfaces
 * };
 * ```
 *
 * This is different from Java where you use "implements" keyword!
 * ==================================================================================================
 */

// YOUR CODE HERE for Bonus Problem


/*
 * ==================================================================================================
 * TESTING INSTRUCTIONS
 * ==================================================================================================
 *
 * After implementing all classes above, compile and test:
 *
 * ```bash
 * ./run_test.sh virtual_functions_advanced
 * ```
 *
 * Or manually:
 * ```bash
 * g++ -std=c++17 03_virtual_functions_advanced.cpp \
 *     ../../tests/test_virtual_functions_advanced.cpp \
 *     -o test_virtual_advanced
 * ./test_virtual_advanced
 * ```
 *
 * GRADING:
 * - Problem 1 (Plugin System): 40 points
 * - Problem 2 (Virtual Destructors): 35 points
 * - Problem 3 (Final Keywords): 25 points
 * - Bonus (Multiple Inheritance): 20 points
 * - Total: 120 points
 *
 * Grade Scale:
 * - 90-100+: A (Excellent)
 * - 80-89: B (Good)
 * - 70-79: C (Satisfactory)
 * - 60-69: D (Needs Work)
 * - <60: F (Incomplete)
 * ==================================================================================================
 */

/*
 * ==================================================================================================
 * IMPORTANT C++ SYNTAX REMINDERS (if coming from Java/Python)
 * ==================================================================================================
 *
 * 1. CONSTRUCTOR SYNTAX:
 *    Java:   public MyClass(int x) { this.x = x; }
 *    C++:    MyClass(int x) : memberX(x) {}  // Initialization list preferred!
 *
 * 2. INHERITANCE:
 *    Java:   class Child extends Parent
 *    C++:    class Child : public Parent
 *
 * 3. INTERFACE IMPLEMENTATION:
 *    Java:   class Foo implements IBar, IBaz
 *    C++:    class Foo : public IBar, public IBaz  // Same syntax as inheritance!
 *
 * 4. PURE VIRTUAL (ABSTRACT METHOD):
 *    Java:   abstract void method();
 *    C++:    virtual void method() = 0;
 *
 * 5. OVERRIDE:
 *    Java:   @Override void method() { }
 *    C++:    void method() override { }  // 'override' is keyword, not annotation
 *
 * 6. FINAL:
 *    Java:   final class Foo / final void method()
 *    C++:    class Foo final / void method() final
 *
 * 7. VIRTUAL DESTRUCTOR (NO JAVA EQUIVALENT):
 *    C++:    virtual ~MyClass() { }  // CRITICAL for polymorphic classes!
 *
 * 8. CONST CORRECTNESS (LIMITED IN JAVA):
 *    C++:    int getValue() const { }  // Method doesn't modify object
 * ==================================================================================================
 */