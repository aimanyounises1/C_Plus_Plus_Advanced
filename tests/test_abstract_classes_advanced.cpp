/*
 * ==================================================================================================
 * TEST SUITE: Advanced Abstract Classes
 * ==================================================================================================
 * DO NOT MODIFY THIS FILE!
 * ==================================================================================================
 */

#include "../exercises/phase2_intermediate/oop_advanced/04_abstract_classes_advanced.cpp"
#include <iostream>
#include <iomanip>
#include <cassert>

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
 * PROBLEM 1 TESTS: Database Driver Architecture
 */
void testDatabaseDrivers() {
    printHeader("Problem 1: Database Driver Architecture (35 points)");

    try {
        // Test 1: MySQLDriver (10 points)
        MySQLDriver* mysql = new MySQLDriver();
        testCase("MySQLDriver creation", mysql != nullptr, 2);
        testCase("MySQL getDriverName", mysql->getDriverName() == "MySQL", 2);

        bool connected = mysql->connect("mysql://localhost");
        testCase("MySQL connect", connected, 2);

        bool executed = mysql->execute("SELECT * FROM users");
        testCase("MySQL execute (connected)", executed, 2);

        vector<string> results = mysql->fetchResults();
        testCase("MySQL fetchResults", results.size() == 1 && results[0] == "MySQL data", 2);

        // Test 2: PostgreSQLDriver (10 points)
        PostgreSQLDriver* postgres = new PostgreSQLDriver();
        testCase("PostgreSQLDriver creation", postgres != nullptr, 2);
        testCase("PostgreSQL getDriverName", postgres->getDriverName() == "PostgreSQL", 2);

        postgres->connect("postgresql://localhost");
        postgres->execute("SELECT * FROM users");
        results = postgres->fetchResults();
        testCase("PostgreSQL fetchResults", results.size() == 2, 3);
        testCase("PostgreSQL row data", results[0] == "PostgreSQL row 1" && results[1] == "PostgreSQL row 2", 3);

        // Test 3: MongoDBDriver (10 points)
        MongoDBDriver* mongo = new MongoDBDriver();
        testCase("MongoDBDriver creation", mongo != nullptr, 2);
        testCase("MongoDB getDriverName", mongo->getDriverName() == "MongoDB", 2);

        mongo->connect("mongodb://localhost");
        mongo->execute("db.users.find()");
        results = mongo->fetchResults();
        testCase("MongoDB fetchResults", results.size() == 1 && results[0] == "MongoDB document", 3);

        // Test 4: Polymorphic behavior (5 points)
        DatabaseDriver* driver = mysql;
        testCase("Polymorphic driver (MySQL)", driver->getDriverName() == "MySQL", 2);

        driver = postgres;
        testCase("Polymorphic driver (PostgreSQL)", driver->getDriverName() == "PostgreSQL", 2);

        driver = mongo;
        testCase("Polymorphic driver (MongoDB)", driver->getDriverName() == "MongoDB", 1);

        // Cleanup
        mysql->disconnect();
        postgres->disconnect();
        mongo->disconnect();
        delete mysql;
        delete postgres;
        delete mongo;

    } catch (const exception& e) {
        cout << RED << "Exception: " << e.what() << RESET << endl;
        testCase("Database Driver Exception Test", false, 35);
    }
}

/*
 * PROBLEM 2 TESTS: Graphics Renderer
 */
void testRenderers() {
    printHeader("Problem 2: Graphics Renderer (35 points)");

    try {
        // Test 1: OpenGLRenderer (10 points)
        OpenGLRenderer* opengl = new OpenGLRenderer();
        testCase("OpenGLRenderer creation", opengl != nullptr, 2);
        testCase("OpenGL getRendererName", opengl->getRendererName() == "OpenGL", 2);

        opengl->render();
        testCase("OpenGL render (frame 1)", opengl->getFrameCount() == 1, 3);

        opengl->render();
        testCase("OpenGL render (frame 2)", opengl->getFrameCount() == 2, 3);

        // Test 2: DirectXRenderer (10 points)
        DirectXRenderer* directx = new DirectXRenderer();
        testCase("DirectXRenderer creation", directx != nullptr, 2);
        testCase("DirectX getRendererName", directx->getRendererName() == "DirectX", 2);

        directx->render();
        testCase("DirectX render (frame 1)", directx->getFrameCount() == 1, 3);

        directx->render();
        directx->render();
        testCase("DirectX render (frame 3)", directx->getFrameCount() == 3, 3);

        // Test 3: VulkanRenderer (10 points)
        VulkanRenderer* vulkan = new VulkanRenderer();
        testCase("VulkanRenderer creation", vulkan != nullptr, 2);
        testCase("Vulkan getRendererName", vulkan->getRendererName() == "Vulkan", 2);

        vulkan->render();
        testCase("Vulkan render (frame 1)", vulkan->getFrameCount() == 1, 3);

        for (int i = 0; i < 9; i++) vulkan->render();
        testCase("Vulkan render (frame 10)", vulkan->getFrameCount() == 10, 3);

        // Test 4: Polymorphic behavior (5 points)
        Renderer* renderer = opengl;
        renderer->render();
        testCase("Polymorphic render (OpenGL)", renderer->getFrameCount() == 3, 2);

        renderer = directx;
        testCase("Polymorphic getRendererName", renderer->getRendererName() == "DirectX", 2);

        renderer = vulkan;
        testCase("Polymorphic getFrameCount", renderer->getFrameCount() == 10, 1);

        // Cleanup
        delete opengl;
        delete directx;
        delete vulkan;

    } catch (const exception& e) {
        cout << RED << "Exception: " << e.what() << RESET << endl;
        testCase("Renderer Exception Test", false, 35);
    }
}

/*
 * PROBLEM 3 TESTS: Stream Processing
 */
void testStreamProcessors() {
    printHeader("Problem 3: Stream Processing (30 points)");

    try {
        vector<int> input = {1, 2, 3, 4, 5};

        // Test 1: MapProcessor (8 points)
        MapProcessor* mapper = new MapProcessor();
        testCase("MapProcessor creation", mapper != nullptr, 2);

        vector<int> result = mapper->transform(input);
        testCase("MapProcessor transform", result.size() == 5 && result[0] == 2 && result[4] == 10, 6);

        // Test 2: FilterProcessor (8 points)
        FilterProcessor* filter = new FilterProcessor();
        testCase("FilterProcessor creation", filter != nullptr, 2);

        result = filter->filter(input, 3);
        testCase("FilterProcessor filter (>3)", result.size() == 2 && result[0] == 4 && result[1] == 5, 6);

        // Test 3: ReduceProcessor transform (5 points)
        ReduceProcessor* reducer = new ReduceProcessor();
        testCase("ReduceProcessor creation", reducer != nullptr, 2);

        result = reducer->transform(input);
        testCase("ReduceProcessor transform", result.size() == 5 && result[0] == 11 && result[4] == 15, 3);

        // Test 4: ReduceProcessor filter (4 points)
        result = reducer->filter(input, 3);
        testCase("ReduceProcessor filter (<3)", result.size() == 2 && result[0] == 1 && result[1] == 2, 4);

        // Test 5: ReduceProcessor aggregate (5 points)
        int sum = reducer->aggregate(input);
        testCase("ReduceProcessor aggregate", sum == 15, 5);

        // Cleanup
        delete mapper;
        delete filter;
        delete reducer;

    } catch (const exception& e) {
        cout << RED << "Exception: " << e.what() << RESET << endl;
        testCase("Stream Processor Exception Test", false, 30);
    }
}

/*
 * BONUS TESTS: Factory Pattern
 */
void testFactoryPattern() {
    printHeader("Bonus: Factory Pattern (20 points)");

    try {
        // Test 1: LightThemeFactory (8 points)
        LightThemeFactory* lightFactory = new LightThemeFactory();
        testCase("LightThemeFactory creation", lightFactory != nullptr, 2);
        testCase("Light getThemeName", lightFactory->getThemeName() == "Light", 2);

        Widget* lightButton = lightFactory->createButton();
        testCase("Light createButton", lightButton != nullptr && lightButton->getType() == "Button", 2);

        Widget* lightField = lightFactory->createTextField();
        testCase("Light createTextField", lightField != nullptr && lightField->getType() == "TextField", 2);

        // Test 2: DarkThemeFactory (8 points)
        DarkThemeFactory* darkFactory = new DarkThemeFactory();
        testCase("DarkThemeFactory creation", darkFactory != nullptr, 2);
        testCase("Dark getThemeName", darkFactory->getThemeName() == "Dark", 2);

        Widget* darkButton = darkFactory->createButton();
        testCase("Dark createButton", darkButton != nullptr && darkButton->getType() == "Button", 2);

        Widget* darkField = darkFactory->createTextField();
        testCase("Dark createTextField", darkField != nullptr && darkField->getType() == "TextField", 2);

        // Test 3: Polymorphic factory (4 points)
        ThemeFactory* factory = lightFactory;
        Widget* widget = factory->createButton();
        testCase("Polymorphic factory (Light)", widget != nullptr, 2);

        factory = darkFactory;
        widget = factory->createTextField();
        testCase("Polymorphic factory (Dark)", widget != nullptr && widget->getType() == "TextField", 2);

        // Cleanup
        delete lightButton;
        delete lightField;
        delete darkButton;
        delete darkField;
        delete lightFactory;
        delete darkFactory;

    } catch (const exception& e) {
        cout << RED << "Exception: " << e.what() << RESET << endl;
        testCase("Factory Pattern Exception Test", false, 20);
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
    cout << "  Database Drivers:      Problem 1 (35 points)" << endl;
    cout << "  Graphics Renderer:     Problem 2 (35 points)" << endl;
    cout << "  Stream Processing:     Problem 3 (30 points)" << endl;
    cout << "  Factory Pattern:       Bonus     (20 points)" << endl;

    cout << "\n" << BOLD << "═══════════════════════════════════════════════════════════════" << RESET << endl;
}

int main() {
    cout << BOLD << "\n╔═══════════════════════════════════════════════════════════════╗" << RESET << endl;
    cout << BOLD << "║     ADVANCED ABSTRACT CLASSES - AUTOMATED TEST SUITE          ║" << RESET << endl;
    cout << BOLD << "╚═══════════════════════════════════════════════════════════════╝" << RESET << endl;

    testDatabaseDrivers();
    testRenderers();
    testStreamProcessors();
    testFactoryPattern();

    printSummary();

    return (earnedPoints == totalPoints) ? 0 : 1;
}