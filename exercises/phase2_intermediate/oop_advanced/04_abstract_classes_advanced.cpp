/*
 * ==================================================================================================
 * ADVANCED EXERCISE: Abstract Classes & Interface Design
 * ==================================================================================================
 * Difficulty: Advanced | Estimated Time: 60-75 minutes
 * Points: 120 total (100 required + 20 bonus)
 *
 * CHALLENGE MODE: Implement from scratch!
 * - Design complex class hierarchies
 * - Implement multiple interfaces
 * - Solve real-world architectural problems
 * - NO code scaffolding - you write everything!
 *
 * This exercise tests:
 * ✓ Abstract class design
 * ✓ Pure virtual functions
 * ✓ Interface segregation principle
 * ✓ Multiple inheritance
 * ✓ Real-world design patterns
 * ==================================================================================================
 */

/*
 * ==================================================================================================
 * PROBLEM 1: Database Driver Architecture (35 points)
 * ==================================================================================================
 *
 * Design an abstract database driver system that can work with different databases (MySQL,
 * PostgreSQL, MongoDB) polymorphically.
 *
 * SCENARIO:
 * You're building a data processing application that needs to support multiple database backends.
 * Users can switch between databases at runtime without changing application code.
 *
 * REQUIREMENTS:
 *
 * 1. Create ABSTRACT class "DatabaseDriver" with:
 *    - Pure virtual: bool connect(const string& connectionString)
 *    - Pure virtual: void disconnect()
 *    - Pure virtual: bool execute(const string& query)
 *    - Pure virtual: vector<string> fetchResults()
 *    - Pure virtual: string getDriverName() const
 *    - Protected member: bool isConnected (default false)
 *    - Regular virtual: string getStatus() const
 *      Returns: "<DriverName>: <Connected/Disconnected>"
 *    - Virtual destructor (should disconnect if still connected)
 *
 * 2. Implement THREE database drivers:
 *
 *    a) MySQLDriver:
 *       - getDriverName() returns "MySQL"
 *       - connect() sets isConnected = true, returns true
 *       - disconnect() sets isConnected = false
 *       - execute() returns true if isConnected, false otherwise
 *       - fetchResults() returns vector with one element: "MySQL data"
 *
 *    b) PostgreSQLDriver:
 *       - getDriverName() returns "PostgreSQL"
 *       - connect() sets isConnected = true, returns true
 *       - disconnect() sets isConnected = false
 *       - execute() returns true if isConnected, false otherwise
 *       - fetchResults() returns vector with two elements: "PostgreSQL row 1", "PostgreSQL row 2"
 *
 *    c) MongoDBDriver:
 *       - getDriverName() returns "MongoDB"
 *       - connect() sets isConnected = true, returns true
 *       - disconnect() sets isConnected = false
 *       - execute() returns true if isConnected, false otherwise
 *       - fetchResults() returns vector with one element: "MongoDB document"
 *
 * 3. Key Requirements:
 *    - Destructor must call disconnect() if isConnected is true
 *    - Use proper const correctness
 *    - Protected members accessible in derived classes
 *
 * WHY ABSTRACT CLASSES HERE?
 * - Defines common interface for all database operations
 * - Application code works with DatabaseDriver*, not specific implementation
 * - Easy to add new database drivers without changing existing code
 * ==================================================================================================
 */

// YOUR CODE HERE for Problem 1


/*
 * ==================================================================================================
 * PROBLEM 2: Graphics Renderer with Template Method Pattern (35 points)
 * ==================================================================================================
 *
 * Implement a graphics rendering system using the Template Method pattern. The base class defines
 * the rendering algorithm, but derived classes customize specific steps.
 *
 * SCENARIO:
 * You're building a cross-platform graphics engine. Different platforms (OpenGL, DirectX, Vulkan)
 * have different implementation details, but follow the same rendering pipeline.
 *
 * REQUIREMENTS:
 *
 * 1. Create ABSTRACT class "Renderer" with:
 *    - Pure virtual: void initializeContext()
 *    - Pure virtual: void clearScreen()
 *    - Pure virtual: void renderGeometry()
 *    - Pure virtual: void swapBuffers()
 *    - Pure virtual: string getRendererName() const
 *    - Protected member: int frameCount (initialize to 0)
 *    - FINAL virtual method: void render()
 *      Implementation:
 *        - Call initializeContext()
 *        - Call clearScreen()
 *        - Call renderGeometry()
 *        - Call swapBuffers()
 *        - Increment frameCount
 *      Mark as FINAL so derived classes can't change the algorithm!
 *    - Regular virtual: int getFrameCount() const (returns frameCount)
 *    - Virtual destructor
 *
 * 2. Implement THREE renderers:
 *
 *    a) OpenGLRenderer:
 *       - getRendererName() returns "OpenGL"
 *       - initializeContext() does nothing (empty body)
 *       - clearScreen() does nothing
 *       - renderGeometry() does nothing
 *       - swapBuffers() does nothing
 *
 *    b) DirectXRenderer:
 *       - getRendererName() returns "DirectX"
 *       - initializeContext() does nothing
 *       - clearScreen() does nothing
 *       - renderGeometry() does nothing
 *       - swapBuffers() does nothing
 *
 *    c) VulkanRenderer:
 *       - getRendererName() returns "Vulkan"
 *       - initializeContext() does nothing
 *       - clearScreen() does nothing
 *       - renderGeometry() does nothing
 *       - swapBuffers() does nothing
 *
 * 3. Key Concept - Template Method Pattern:
 *    - render() defines the ALGORITHM (sequence of steps)
 *    - Derived classes implement each STEP
 *    - Algorithm stays consistent across all implementations
 *    - render() is FINAL to prevent changing the algorithm
 *
 * IMPORTANT:
 * The render() method must be marked FINAL. This ensures all renderers follow the same
 * rendering pipeline, with only implementation details varying.
 * ==================================================================================================
 */

// YOUR CODE HERE for Problem 2


/*
 * ==================================================================================================
 * PROBLEM 3: Stream Processing with Multiple Interfaces (30 points)
 * ==================================================================================================
 *
 * Design a data stream processing system where processors implement multiple capabilities through
 * separate interfaces. This demonstrates interface segregation principle.
 *
 * SCENARIO:
 * You're building a data pipeline. Different processors have different capabilities:
 * - Some can transform data
 * - Some can filter data
 * - Some can aggregate data
 * Not all processors need all capabilities!
 *
 * REQUIREMENTS:
 *
 * 1. Create interface "ITransformable" with:
 *    - Pure virtual: vector<int> transform(const vector<int>& input) const
 *    - Virtual destructor
 *
 * 2. Create interface "IFilterable" with:
 *    - Pure virtual: vector<int> filter(const vector<int>& input, int threshold) const
 *    - Virtual destructor
 *
 * 3. Create interface "IAggregatable" with:
 *    - Pure virtual: int aggregate(const vector<int>& input) const
 *    - Virtual destructor
 *
 * 4. Implement THREE processor classes:
 *
 *    a) MapProcessor (implements ONLY ITransformable):
 *       - transform() multiplies each element by 2
 *
 *    b) FilterProcessor (implements ONLY IFilterable):
 *       - filter() returns only elements greater than threshold
 *
 *    c) ReduceProcessor (implements ALL THREE interfaces):
 *       - transform() adds 10 to each element
 *       - filter() returns elements less than threshold
 *       - aggregate() returns sum of all elements
 *
 * 5. Key Requirements:
 *    - Use proper multiple inheritance syntax
 *    - MapProcessor inherits from ONLY ITransformable
 *    - FilterProcessor inherits from ONLY IFilterable
 *    - ReduceProcessor inherits from ALL THREE interfaces
 *
 * WHY SEPARATE INTERFACES?
 * Interface Segregation Principle: Clients shouldn't depend on interfaces they don't use.
 * MapProcessor doesn't need filtering capability, so it doesn't inherit IFilterable.
 * ==================================================================================================
 */

// YOUR CODE HERE for Problem 3


/*
 * ==================================================================================================
 * BONUS PROBLEM: Factory Pattern with Abstract Products (20 points)
 * ==================================================================================================
 *
 * Implement the Abstract Factory pattern for creating families of related objects.
 *
 * SCENARIO:
 * You're building a UI library that supports multiple themes (Light, Dark). Each theme has
 * different implementations of buttons, text fields, and checkboxes.
 *
 * REQUIREMENTS:
 *
 * 1. Create abstract product "Widget" with:
 *    - Pure virtual: void render() const
 *    - Pure virtual: string getType() const
 *    - Virtual destructor
 *
 * 2. Create concrete widgets:
 *    a) LightButton (inherits Widget):
 *       - getType() returns "Button"
 *       - render() does nothing
 *
 *    b) DarkButton (inherits Widget):
 *       - getType() returns "Button"
 *       - render() does nothing
 *
 *    c) LightTextField (inherits Widget):
 *       - getType() returns "TextField"
 *       - render() does nothing
 *
 *    d) DarkTextField (inherits Widget):
 *       - getType() returns "TextField"
 *       - render() does nothing
 *
 * 3. Create abstract factory "ThemeFactory" with:
 *    - Pure virtual: Widget* createButton() const
 *    - Pure virtual: Widget* createTextField() const
 *    - Pure virtual: string getThemeName() const
 *    - Virtual destructor
 *
 * 4. Create concrete factories:
 *    a) LightThemeFactory:
 *       - createButton() returns new LightButton()
 *       - createTextField() returns new LightTextField()
 *       - getThemeName() returns "Light"
 *
 *    b) DarkThemeFactory:
 *       - createButton() returns new DarkButton()
 *       - createTextField() returns new DarkTextField()
 *       - getThemeName() returns "Dark"
 *
 * 5. Key Concept:
 *    Factory pattern separates object creation from usage.
 *    Abstract factory creates families of related objects.
 *
 * MEMORY NOTE:
 * Factory methods return raw pointers. In production, use smart pointers!
 * Tests will handle cleanup.
 * ==================================================================================================
 */

// YOUR CODE HERE for Bonus Problem


/*
 * ==================================================================================================
 * TESTING INSTRUCTIONS
 * ==================================================================================================
 *
 * After implementing all classes, test with:
 *
 * ```bash
 * ./run_test.sh abstract_classes_advanced
 * ```
 *
 * Or manually:
 * ```bash
 * g++ -std=c++17 04_abstract_classes_advanced.cpp \
 *     ../../tests/test_abstract_classes_advanced.cpp \
 *     -o test_abstract_advanced
 * ./test_abstract_advanced
 * ```
 *
 * GRADING:
 * - Problem 1 (Database Drivers): 35 points
 * - Problem 2 (Renderer Template): 35 points
 * - Problem 3 (Stream Processors): 30 points
 * - Bonus (Factory Pattern): 20 points
 * - Total: 120 points
 *
 * Grade Scale:
 * - 90-100+: A (Excellent - Ready for production!)
 * - 80-89: B (Good - Minor improvements needed)
 * - 70-79: C (Satisfactory - Review concepts)
 * - 60-69: D (Needs Work - Practice more)
 * - <60: F (Incomplete - Study theory again)
 * ==================================================================================================
 */

/*
 * ==================================================================================================
 * DESIGN PATTERN CHEAT SHEET
 * ==================================================================================================
 *
 * 1. TEMPLATE METHOD PATTERN:
 *    - Base class defines algorithm skeleton (FINAL method)
 *    - Derived classes implement individual steps (pure virtual)
 *    - Ensures consistent algorithm across implementations
 *    Example: Renderer::render() is FINAL, steps are pure virtual
 *
 * 2. ABSTRACT FACTORY PATTERN:
 *    - Interface for creating families of related objects
 *    - Concrete factories create concrete products
 *    - Client works with abstract factory and products
 *    Example: ThemeFactory creates theme-specific widgets
 *
 * 3. INTERFACE SEGREGATION:
 *    - Many small interfaces better than one large interface
 *    - Classes implement only interfaces they need
 *    - Reduces coupling and unnecessary dependencies
 *    Example: MapProcessor only implements ITransformable
 *
 * 4. STRATEGY PATTERN:
 *    - Abstract class defines interface for algorithm
 *    - Concrete classes implement different algorithms
 *    - Can swap algorithms at runtime
 *    Example: DatabaseDriver can be MySQL, PostgreSQL, or MongoDB
 * ==================================================================================================
 */

/*
 * ==================================================================================================
 * C++ VS JAVA - ABSTRACT CLASSES
 * ==================================================================================================
 *
 * PURE VIRTUAL FUNCTIONS:
 * Java:    abstract void method();
 * C++:     virtual void method() = 0;
 *
 * ABSTRACT CLASS:
 * Java:    abstract class Foo { }
 * C++:     class Foo { virtual void method() = 0; };  // Any pure virtual makes it abstract
 *
 * CANNOT INSTANTIATE:
 * Java:    Foo f = new Foo();  // Compile error
 * C++:     Foo f;               // Compile error
 *
 * CAN HAVE POINTERS:
 * Java:    Foo f = new Derived();  // OK
 * C++:     Foo* f = new Derived(); // OK
 *
 * MUST IMPLEMENT ALL PURE VIRTUALS:
 * Java:    class Derived extends Foo { void method() { } }  // Must implement
 * C++:     class Derived : public Foo { void method() override { } };  // Must implement
 *
 * CAN HAVE CONSTRUCTOR:
 * Java:    Yes (called by derived class constructor)
 * C++:     Yes (called by derived class constructor)
 *
 * CAN HAVE DATA MEMBERS:
 * Java:    Yes
 * C++:     Yes
 *
 * KEY DIFFERENCE:
 * Java has separate "interface" keyword for pure interfaces.
 * C++ uses abstract classes for both abstract classes and interfaces (convention based).
 * ==================================================================================================
 */