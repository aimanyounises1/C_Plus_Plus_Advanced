/*
 * Exercise: File I/O
 * Difficulty: Beginner
 * Time: 35-45 minutes
 * Topics: ifstream, ofstream, fstream, file modes, error handling, parsing
 *
 * LEARNING OBJECTIVES:
 * - Master file input (reading)
 * - Master file output (writing)
 * - Understand file modes (append, binary, etc.)
 * - Learn error handling for files
 * - Practice parsing different file formats
 * - Understand file position and seeking
 *
 * INTERVIEW RELEVANCE:
 * - File I/O is fundamental for data processing
 * - Error handling is critical in production code
 * - Parsing is common in technical interviews
 * - Configuration files are ubiquitous
 * - GPU programs often read data from files
 */

#include <iostream>
#include <fstream>   // File streams
#include <string>
#include <vector>
#include <sstream>   // String streams for parsing
#include <iomanip>   // For formatting output

int main() {
    std::cout << "=== File I/O Exercises ===\n\n";

    // ========================================================================
    // EXERCISE 1: Writing to Files (ofstream) (5 min)
    // ========================================================================
    std::cout << "Exercise 1: Writing to Files\n";
    std::cout << "-----------------------------\n";

    // TODO 1.1: Write text to a file
    // std::ofstream outFile("output.txt");
    // if (outFile.is_open()) {
    //     outFile << "Hello, File I/O!\n";
    //     outFile << "This is line 2\n";
    //     outFile << "Number: " << 42 << "\n";
    //     outFile.close();
    //     std::cout << "File written successfully\n";
    // } else {
    //     std::cerr << "Error opening file for writing\n";
    // }


    // TODO 1.2: Write using constructor (auto-close via RAII)
    // {
    //     std::ofstream file("auto_close.txt");
    //     file << "RAII automatically closes the file\n";
    // }  // File closed here automatically


    // TODO 1.3: Check if file opened successfully
    // if (!outFile) {  // Equivalent to !outFile.is_open()
    //     std::cerr << "Failed to open file\n";
    // }


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 2: Reading from Files (ifstream) (10 min)
    // ========================================================================
    std::cout << "Exercise 2: Reading from Files\n";
    std::cout << "-------------------------------\n";

    // First, create a test file to read
    {
        std::ofstream testFile("test_input.txt");
        testFile << "Line 1\n";
        testFile << "Line 2\n";
        testFile << "Number: 42\n";
        testFile << "Float: 3.14\n";
    }

    // TODO 2.1: Read file line by line
    // std::ifstream inFile("test_input.txt");
    // if (inFile.is_open()) {
    //     std::string line;
    //     while (std::getline(inFile, line)) {
    //         std::cout << line << "\n";
    //     }
    //     inFile.close();
    // }


    // TODO 2.2: Read file word by word
    // std::ifstream wordFile("test_input.txt");
    // std::string word;
    // while (wordFile >> word) {
    //     std::cout << "Word: " << word << "\n";
    // }


    // TODO 2.3: Read entire file into a string
    // std::ifstream file("test_input.txt");
    // std::string content((std::istreambuf_iterator<char>(file)),
    //                      std::istreambuf_iterator<char>());
    // std::cout << "File content:\n" << content;


    // TODO 2.4: Read and parse structured data
    // File format: "Number: 42"
    // std::ifstream dataFile("test_input.txt");
    // std::string label;
    // int number;
    // dataFile >> label >> number;  // Reads "Number:" and 42


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 3: File Modes (5 min)
    // ========================================================================
    std::cout << "Exercise 3: File Modes\n";
    std::cout << "----------------------\n";

    // TODO 3.1: Append mode (add to end of file)
    // std::ofstream appendFile("append_test.txt", std::ios::app);
    // appendFile << "This is appended\n";


    // TODO 3.2: Truncate mode (default for ofstream - clears file)
    // std::ofstream truncFile("trunc_test.txt", std::ios::trunc);


    // TODO 3.3: Binary mode
    // std::ofstream binFile("binary.dat", std::ios::binary);
    // int value = 42;
    // binFile.write(reinterpret_cast<char*>(&value), sizeof(value));


    // TODO 3.4: Multiple flags combined
    // std::ofstream file("test.txt", std::ios::out | std::ios::app | std::ios::binary);


    std::cout << "Different file modes demonstrated\n";
    std::cout << "\n";

    // ========================================================================
    // EXERCISE 4: Error Handling (10 min)
    // ========================================================================
    std::cout << "Exercise 4: Error Handling\n";
    std::cout << "--------------------------\n";

    // TODO 4.1: Check if file exists before reading
    std::ifstream checkFile("nonexistent.txt");
    if (!checkFile) {
        std::cout << "File does not exist or cannot be opened\n";
    }

    // TODO 4.2: Check stream state
    // std::ifstream file("test.txt");
    // if (file.fail()) {
    //     std::cerr << "File operation failed\n";
    // }
    // if (file.eof()) {
    //     std::cout << "Reached end of file\n";
    // }
    // if (file.bad()) {
    //     std::cerr << "Fatal error occurred\n";
    // }


    // TODO 4.3: Clear error state
    // file.clear();  // Clear error flags
    // file.seekg(0); // Go back to beginning


    // TODO 4.4: Proper error handling pattern
    // std::ifstream file("data.txt");
    // if (!file) {
    //     std::cerr << "Error: Cannot open data.txt\n";
    //     return 1;
    // }
    //
    // std::string line;
    // while (std::getline(file, line)) {
    //     // Process line
    // }
    //
    // if (file.bad()) {
    //     std::cerr << "Error: Fatal error while reading\n";
    // } else if (!file.eof()) {
    //     std::cerr << "Error: File read incomplete\n";
    // }


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 5: Parsing Data from Files (10 min)
    // ========================================================================
    std::cout << "Exercise 5: Parsing Data\n";
    std::cout << "------------------------\n";

    // Create a CSV file for testing
    {
        std::ofstream csv("data.csv");
        csv << "Name,Age,GPA\n";
        csv << "Alice,20,3.8\n";
        csv << "Bob,22,3.5\n";
        csv << "Charlie,21,3.9\n";
    }

    // TODO 5.1: Parse CSV file
    std::ifstream csvFile("data.csv");
    if (csvFile.is_open()) {
        std::string line;

        // Read header
        std::getline(csvFile, line);
        std::cout << "CSV Header: " << line << "\n\n";

        // Read data lines
        while (std::getline(csvFile, line)) {
            std::stringstream ss(line);
            std::string name, age, gpa;

            // Parse using getline with delimiter
            std::getline(ss, name, ',');
            std::getline(ss, age, ',');
            std::getline(ss, gpa, ',');

            std::cout << "Name: " << name
                      << ", Age: " << age
                      << ", GPA: " << gpa << "\n";
        }
    }

    // TODO 5.2: Parse numbers from a file
    // File: numbers.txt contains "1 2 3 4 5"
    {
        std::ofstream numFile("numbers.txt");
        numFile << "1 2 3 4 5\n";
        numFile << "10 20 30 40 50\n";
    }

    // Read numbers into a vector
    // std::ifstream numFile("numbers.txt");
    // std::vector<int> numbers;
    // int num;
    // while (numFile >> num) {
    //     numbers.push_back(num);
    // }


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 6: File Position and Seeking (5 min)
    // ========================================================================
    std::cout << "Exercise 6: File Position\n";
    std::cout << "-------------------------\n";

    {
        std::ofstream posFile("position_test.txt");
        posFile << "0123456789ABCDEFGHIJ";
    }

    // TODO 6.1: Get current position
    // std::ifstream file("position_test.txt");
    // std::streampos pos = file.tellg();
    // std::cout << "Current position: " << pos << "\n";


    // TODO 6.2: Seek to specific position
    // file.seekg(10);  // Move to position 10
    // char c;
    // file >> c;
    // std::cout << "Character at position 10: " << c << "\n";


    // TODO 6.3: Seek relative to current position
    // file.seekg(5, std::ios::cur);   // Move 5 forward from current
    // file.seekg(-3, std::ios::cur);  // Move 3 backward from current


    // TODO 6.4: Seek from end
    // file.seekg(-5, std::ios::end);  // 5 characters from end


    // TODO 6.5: Get file size
    std::ifstream sizeFile("position_test.txt", std::ios::ate);  // Open at end
    std::streampos fileSize = sizeFile.tellg();
    std::cout << "File size: " << fileSize << " bytes\n";

    std::cout << "\n";

    // ========================================================================
    // EXERCISE 7: Binary File I/O (5 min)
    // ========================================================================
    std::cout << "Exercise 7: Binary Files\n";
    std::cout << "------------------------\n";

    // TODO 7.1: Write binary data
    {
        std::ofstream binOut("binary_data.bin", std::ios::binary);
        int values[] = {10, 20, 30, 40, 50};
        binOut.write(reinterpret_cast<char*>(values), sizeof(values));
    }

    // TODO 7.2: Read binary data
    {
        std::ifstream binIn("binary_data.bin", std::ios::binary);
        int values[5];
        binIn.read(reinterpret_cast<char*>(values), sizeof(values));

        std::cout << "Binary data read: ";
        for (int v : values) {
            std::cout << v << " ";
        }
        std::cout << "\n";
    }

    // TODO 7.3: Write a structure to binary file
    // struct Point {
    //     int x;
    //     int y;
    // };
    //
    // Point p = {10, 20};
    // std::ofstream out("point.bin", std::ios::binary);
    // out.write(reinterpret_cast<char*>(&p), sizeof(p));


    std::cout << "\n";

    // ========================================================================
    // EXERCISE 8: Practical Applications (5 min)
    // ========================================================================
    std::cout << "Exercise 8: Practical Applications\n";
    std::cout << "-----------------------------------\n";

    // TODO 8.1: Configuration file reader
    {
        std::ofstream config("config.txt");
        config << "width=1920\n";
        config << "height=1080\n";
        config << "fullscreen=true\n";
    }

    // Parse configuration
    // std::ifstream configFile("config.txt");
    // std::string line;
    // while (std::getline(configFile, line)) {
    //     size_t pos = line.find('=');
    //     if (pos != std::string::npos) {
    //         std::string key = line.substr(0, pos);
    //         std::string value = line.substr(pos + 1);
    //         std::cout << key << " -> " << value << "\n";
    //     }
    // }


    // TODO 8.2: Log file writer
    // void writeLog(const std::string& message) {
    //     std::ofstream log("app.log", std::ios::app);
    //     log << "[" << getCurrentTime() << "] " << message << "\n";
    // }


    // TODO 8.3: Save game state
    // struct GameState {
    //     int level;
    //     int score;
    //     float health;
    // };
    //
    // void saveGame(const GameState& state) {
    //     std::ofstream save("savegame.dat", std::ios::binary);
    //     save.write(reinterpret_cast<const char*>(&state), sizeof(state));
    // }


    std::cout << "\n";

    // ========================================================================
    // CHALLENGE EXERCISES (Optional - 10 min)
    // ========================================================================
    std::cout << "Challenge Exercises\n";
    std::cout << "-------------------\n";

    // CHALLENGE 1: Read a file and count word frequency
    // Create a map of word -> count


    // CHALLENGE 2: Implement a simple database (CSV)
    // - Add record
    // - Search record
    // - Delete record
    // - Update record


    // CHALLENGE 3: Parse JSON-like format
    // Read: { "name": "Alice", "age": 25 }


    // CHALLENGE 4: Implement file encryption
    // Simple XOR cipher: encrypt and decrypt files


    // CHALLENGE 5: Binary serialization of complex data
    // Serialize a vector of structures


    std::cout << "\n";

    // Clean up test files
    std::cout << "Cleaning up test files...\n";
    // In a real program, you might want to delete these:
    // std::remove("output.txt");
    // std::remove("test_input.txt");
    // etc.

    return 0;
}

// ========================================================================
// COMMON INTERVIEW QUESTIONS
// ========================================================================
/*
 * Q1: What's the difference between ifstream, ofstream, and fstream?
 * A: ifstream: Input file stream (reading)
 *    - Opens file for reading
 *    - Example: std::ifstream file("input.txt");
 *
 *    ofstream: Output file stream (writing)
 *    - Opens file for writing
 *    - Truncates by default
 *    - Example: std::ofstream file("output.txt");
 *
 *    fstream: File stream (reading and writing)
 *    - Can do both input and output
 *    - Must specify mode explicitly
 *    - Example: std::fstream file("data.txt", std::ios::in | std::ios::out);
 *
 * Q2: What are file open modes?
 * A: std::ios::in      - Open for reading
 *    std::ios::out     - Open for writing (truncates by default)
 *    std::ios::app     - Append (write to end)
 *    std::ios::trunc   - Truncate (clear file)
 *    std::ios::binary  - Binary mode (no text conversion)
 *    std::ios::ate     - Open and seek to end
 *
 *    Combine with |:
 *    std::ofstream file("data.txt", std::ios::out | std::ios::app);
 *
 * Q3: How do you check if a file opened successfully?
 * A: Several ways:
 *
 *    1. is_open():
 *       if (file.is_open()) { ... }
 *
 *    2. Boolean conversion:
 *       if (file) { ... }
 *       if (!file) { std::cerr << "Failed\n"; }
 *
 *    3. fail():
 *       if (file.fail()) { std::cerr << "Failed\n"; }
 *
 *    Best practice: Check immediately after opening!
 *
 * Q4: What's the difference between text and binary mode?
 * A: Text mode (default):
 *    - Platform-specific line ending conversion
 *    - Windows: \r\n ↔ \n
 *    - Unix/Mac: \n stays \n
 *    - Use for human-readable text files
 *
 *    Binary mode (std::ios::binary):
 *    - No conversion
 *    - Bytes written = bytes on disk
 *    - Use for:
 *      - Binary data (images, executables)
 *      - Exact byte representation
 *      - Cross-platform data files
 *
 * Q5: How do you read an entire file into a string?
 * A: Several methods:
 *
 *    1. Using iterators (efficient):
 *       std::ifstream file("data.txt");
 *       std::string content((std::istreambuf_iterator<char>(file)),
 *                            std::istreambuf_iterator<char>());
 *
 *    2. Using stringstream:
 *       std::ifstream file("data.txt");
 *       std::stringstream buffer;
 *       buffer << file.rdbuf();
 *       std::string content = buffer.str();
 *
 *    3. Using getline in a loop:
 *       std::string line, content;
 *       while (std::getline(file, line)) {
 *           content += line + "\n";
 *       }
 *
 * Q6: What's the difference between >> and getline?
 * A: operator>> (extraction):
 *    - Reads up to whitespace (space, tab, newline)
 *    - Skips leading whitespace
 *    - Example: file >> word; // Reads one word
 *
 *    getline():
 *    - Reads entire line (including spaces)
 *    - Stops at newline (or specified delimiter)
 *    - Consumes but doesn't include delimiter
 *    - Example: std::getline(file, line);
 *
 *    For CSV: std::getline(ss, field, ',');
 *
 * Q7: How do you handle file errors?
 * A: Check stream state flags:
 *
 *    - good(): No errors, can continue
 *    - eof(): End of file reached
 *    - fail(): Logical error (format mismatch, file not found)
 *    - bad(): Fatal error (hardware failure, disk full)
 *
 *    Pattern:
 *    std::ifstream file("data.txt");
 *    if (!file) {
 *        std::cerr << "Cannot open file\n";
 *        return;
 *    }
 *
 *    while (file >> data) {
 *        // Process data
 *    }
 *
 *    if (file.bad()) {
 *        std::cerr << "Fatal error\n";
 *    } else if (!file.eof()) {
 *        std::cerr << "Format error\n";
 *    }
 *
 * Q8: How do you seek in a file?
 * A: Use seekg() for input, seekp() for output:
 *
 *    Absolute:
 *    file.seekg(100);  // Go to byte 100
 *
 *    Relative to current:
 *    file.seekg(10, std::ios::cur);  // Move 10 bytes forward
 *    file.seekg(-5, std::ios::cur);  // Move 5 bytes backward
 *
 *    Relative to end:
 *    file.seekg(-10, std::ios::end); // 10 bytes before end
 *
 *    Get position:
 *    std::streampos pos = file.tellg();
 *
 *    Get file size:
 *    file.seekg(0, std::ios::end);
 *    size_t size = file.tellg();
 */

/*
 * FILE I/O IN GPU PROGRAMMING:
 * =============================
 *
 * 1. Reading Data for GPU Processing:
 *    // Read large dataset from file
 *    std::ifstream file("large_data.bin", std::ios::binary);
 *    std::vector<float> hostData(size);
 *    file.read(reinterpret_cast<char*>(hostData.data()), size * sizeof(float));
 *
 *    // Transfer to GPU
 *    float* deviceData;
 *    cudaMalloc(&deviceData, size * sizeof(float));
 *    cudaMemcpy(deviceData, hostData.data(), size * sizeof(float), cudaMemcpyHostToDevice);
 *
 * 2. Configuration Files for GPU Kernels:
 *    // Read kernel configuration from file
 *    std::ifstream config("kernel_config.txt");
 *    int blockSize, gridSize;
 *    config >> blockSize >> gridSize;
 *
 *    kernel<<<gridSize, blockSize>>>(...);
 *
 * 3. Saving GPU Results:
 *    // Copy results from GPU
 *    cudaMemcpy(hostResults.data(), deviceResults, size * sizeof(float), cudaMemcpyDeviceToHost);
 *
 *    // Write to file
 *    std::ofstream out("results.bin", std::ios::binary);
 *    out.write(reinterpret_cast<char*>(hostResults.data()), size * sizeof(float));
 *
 * 4. Matrix Data Files:
 *    // Read matrix dimensions and data
 *    std::ifstream matrixFile("matrix.txt");
 *    int rows, cols;
 *    matrixFile >> rows >> cols;
 *
 *    std::vector<float> matrix(rows * cols);
 *    for (int i = 0; i < rows * cols; i++) {
 *        matrixFile >> matrix[i];
 *    }
 *
 * 5. Performance: Memory-Mapped Files
 *    For very large files, consider memory-mapped I/O:
 *    - Faster than standard file I/O
 *    - OS handles caching
 *    - Platform-specific (mmap on Unix, MapViewOfFile on Windows)
 *
 * COMPILATION:
 * ============
 * g++ -std=c++17 -Wall -Wextra 03_file_io.cpp -o fileio
 * ./fileio
 *
 * LEARNING CHECKLIST:
 * ===================
 * ☐ Can open files for reading and writing
 * ☐ Understand different file modes
 * ☐ Know how to check for errors
 * ☐ Can parse different file formats
 * ☐ Understand text vs binary mode
 * ☐ Can seek to specific positions in files
 * ☐ Know how to read/write binary data
 * ☐ Understand RAII for automatic file closing
 * ☐ Can handle common file I/O errors
 *
 * NEXT STEPS:
 * ===========
 * - Move to phase1_fundamentals/pointers_memory/
 * - Practice parsing CSV, JSON, XML files
 * - Learn about memory-mapped files
 * - Study std::filesystem (C++17)
 * - Understand buffering and performance
 */
