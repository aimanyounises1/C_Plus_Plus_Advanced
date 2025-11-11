//
// Created by Aiman Younis on 10/11/2025.
//

#include <iostream>
#include <fstream>
using namespace std;

int main() {
    ofstream fout;
    fout.open("06_cpp_files_lecture.txt");
    fout << "Hello World!\n";
    fout.close();
    return 0;

}