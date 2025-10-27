/*
 * ==================================================================================================
 * Exercise: Observer Pattern
 * ==================================================================================================
 * Difficulty: Intermediate/Advanced | Time: 40-50 minutes
 *
 * LEARNING OBJECTIVES:
 * 1. Master Observer (Publish-Subscribe) pattern
 * 2. Understand Subject-Observer relationship
 * 3. Learn event-driven programming
 * 4. Practice decoupled notification systems
 *
 * INTERVIEW RELEVANCE (NVIDIA):
 * - CUDA event notification
 * - Stream completion callbacks
 * - GPU state monitoring
 * - Asynchronous operation handling
 * ==================================================================================================
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <memory>
using namespace std;

/*
 * EXERCISE 1: Basic Observer Pattern (15 min)
 */

// Observer interface
class Observer {
public:
    virtual void update(float temperature) = 0;
    virtual ~Observer() = default;
};

// Subject (Observable)
class WeatherStation {
private:
    vector<Observer*> observers;
    float temperature;

public:
    void attach(Observer* obs) {
        observers.push_back(obs);
    }

    void detach(Observer* obs) {
        observers.erase(
            remove(observers.begin(), observers.end(), obs),
            observers.end()
        );
    }

    void setTemperature(float temp) {
        temperature = temp;
        notify();
    }

    void notify() {
        for (auto obs : observers) {
            obs->update(temperature);
        }
    }
};

// Concrete Observers
class PhoneDisplay : public Observer {
public:
    void update(float temperature) override {
        cout << "Phone Display: Temperature is " << temperature << "°C" << endl;
    }
};

class WebDisplay : public Observer {
public:
    void update(float temperature) override {
        cout << "Web Display: Current temp: " << temperature << "°C" << endl;
    }
};

/*
 * EXERCISE 2: Observer with Smart Pointers (15 min)
 */

class IObserver {
public:
    virtual void onNotify(const string& event, int data) = 0;
    virtual ~IObserver() = default;
};

class Subject {
private:
    vector<weak_ptr<IObserver>> observers;

public:
    void subscribe(shared_ptr<IObserver> obs) {
        observers.push_back(obs);
    }

    void notify(const string& event, int data) {
        // Clean up expired observers
        observers.erase(
            remove_if(observers.begin(), observers.end(),
                [](const weak_ptr<IObserver>& wp) { return wp.expired(); }),
            observers.end()
        );

        // Notify active observers
        for (auto& weakObs : observers) {
            if (auto obs = weakObs.lock()) {
                obs->onNotify(event, data);
            }
        }
    }
};

class Logger : public IObserver {
public:
    void onNotify(const string& event, int data) override {
        cout << "[Logger] Event: " << event << ", Data: " << data << endl;
    }
};

class Counter : public IObserver {
private:
    int count = 0;
public:
    void onNotify(const string& event, int data) override {
        count++;
        cout << "[Counter] Total events: " << count << endl;
    }
};

/*
 * EXERCISE 3: Push vs Pull Model (10 min)
 */

// Push model: Subject sends all data
class PushSubject {
private:
    vector<Observer*> observers;
    float temperature;
    float humidity;

public:
    void attach(Observer* obs) { observers.push_back(obs); }

    void setState(float temp, float hum) {
        temperature = temp;
        humidity = hum;
        for (auto obs : observers) {
            obs->update(temperature);  // Push all data
        }
    }
};

// Pull model: Observer queries Subject
class PullSubject {
private:
    vector<class PullObserver*> observers;
    float temperature;
    float humidity;

public:
    void attach(class PullObserver* obs);
    void setState(float temp, float hum);
    float getTemperature() const { return temperature; }
    float getHumidity() const { return humidity; }
};

class PullObserver {
public:
    virtual void update(PullSubject* subject) = 0;
    virtual ~PullObserver() = default;
};

void PullSubject::attach(PullObserver* obs) {
    observers.push_back(obs);
}

void PullSubject::setState(float temp, float hum) {
    temperature = temp;
    humidity = hum;
    for (auto obs : observers) {
        obs->update(this);  // Observer pulls data
    }
}

class DetailedDisplay : public PullObserver {
public:
    void update(PullSubject* subject) override {
        cout << "Detailed Display: Temp=" << subject->getTemperature()
             << "°C, Humidity=" << subject->getHumidity() << "%" << endl;
    }
};

/*
 * COMMON INTERVIEW QUESTIONS:
 *
 * Q1: What is the Observer pattern?
 * A: Defines one-to-many dependency: when subject changes state,
 *    all observers are notified automatically
 *
 * Q2: Components of Observer pattern?
 * A: - Subject: Maintains list of observers, notifies on state change
 *    - Observer: Interface for objects that should be notified
 *    - ConcreteSubject: Stores state, sends notifications
 *    - ConcreteObserver: Implements update interface
 *
 * Q3: Push vs Pull model?
 * A: Push: Subject sends all data to observers
 *    Pull: Subject notifies, observers query for needed data
 *    Pull is more flexible, Push is simpler
 *
 * Q4: Advantages of Observer?
 * A: - Loose coupling between subject and observers
 *    - Dynamic relationships (add/remove at runtime)
 *    - Broadcast communication
 *    - Open/Closed Principle
 *
 * Q5: Disadvantages?
 * A: - Can cause memory leaks if not detached properly
 *    - Unexpected updates (hard to track notification chain)
 *    - Performance cost with many observers
 *    - No guarantee of notification order
 *
 * Q6: How to prevent memory leaks?
 * A: - Use weak_ptr to store observers
 *    - Explicit detach in observer destructor
 *    - Smart pointer management
 *    - RAII pattern
 *
 * Q7: Observer vs Mediator pattern?
 * A: Observer: One-to-many, subjects notify observers
 *    Mediator: Many-to-many, centralizes communication
 *
 * Q8: Thread safety considerations?
 * A: - Protect observer list with mutex
 *    - Be careful of deadlocks in nested notifications
 *    - Consider async notification queue
 *
 * ==================================================================================================
 * GPU/CUDA RELEVANCE:
 * - CUDA event callbacks: Notify when kernel completes
 * - Stream synchronization: Observe stream state changes
 * - Memory transfer monitoring: Track data movement completion
 * - Performance monitoring: Observe GPU metrics (temperature, utilization)
 *
 * Example use case:
 * class CudaStreamObserver {
 * public:
 *     virtual void onStreamComplete(cudaStream_t stream) = 0;
 * };
 *
 * class CudaStreamManager {
 *     vector<CudaStreamObserver*> observers;
 * public:
 *     void notifyCompletion(cudaStream_t stream) {
 *         for (auto obs : observers) {
 *             obs->onStreamComplete(stream);
 *         }
 *     }
 * };
 *
 * // Usage:
 * class Logger : public CudaStreamObserver {
 *     void onStreamComplete(cudaStream_t stream) override {
 *         log("Stream completed");
 *     }
 * };
 *
 * COMPILATION: g++ -std=c++11 03_observer.cpp -o observer
 * ==================================================================================================
 */

int main() {
    cout << "=== Observer Pattern Practice ===" << endl;

    // Basic Observer
    cout << "\n1. Basic Observer Pattern:" << endl;
    WeatherStation station;
    PhoneDisplay phone;
    WebDisplay web;

    station.attach(&phone);
    station.attach(&web);

    station.setTemperature(25.5);
    station.setTemperature(30.0);

    station.detach(&phone);
    cout << "\nAfter detaching phone:" << endl;
    station.setTemperature(28.0);

    // Observer with Smart Pointers
    cout << "\n2. Observer with Smart Pointers:" << endl;
    Subject subject;
    auto logger = make_shared<Logger>();
    auto counter = make_shared<Counter>();

    subject.subscribe(logger);
    subject.subscribe(counter);

    subject.notify("TemperatureChanged", 25);
    subject.notify("HumidityChanged", 60);

    // Pull Model
    cout << "\n3. Pull Model:" << endl;
    PullSubject pullSubject;
    DetailedDisplay detailed;

    pullSubject.attach(&detailed);
    pullSubject.setState(26.5, 65);
    pullSubject.setState(28.0, 70);

    return 0;
}
