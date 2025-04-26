#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// Function to replace all occurrences of a substring in a string
std::string replaceAll(const std::string& str, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return str;
    }
    
    std::string result = str;
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
        result.replace(pos, from.length(), to);
        pos += to.length();
    }
    
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <substring_to_replace> <replacement_string> [input_file] [output_file]" << std::endl;
        std::cerr << "If input_file is not provided, reads from stdin" << std::endl;
        std::cerr << "If output_file is not provided, writes to stdout" << std::endl;
        return 1;
    }
    
    std::string fromStr = argv[1];
    std::string toStr = argv[2];
    
    std::string inputText;
    
    if (argc >= 4) {
        // Read from file
        std::ifstream inFile(argv[3]);
        if (!inFile) {
            std::cerr << "Cannot open input file: " << argv[3] << std::endl;
            return 1;
        }
        
        std::string line;
        while (std::getline(inFile, line)) {
            inputText += line + "\n";
        }
        inFile.close();
    } else {
        // Read from stdin
        std::string line;
        while (std::getline(std::cin, line)) {
            inputText += line + "\n";
        }
    }
    
    // Perform replacement
    std::string outputText = replaceAll(inputText, fromStr, toStr);
    
    if (argc >= 5) {
        // Write to file
        std::ofstream outFile(argv[4]);
        if (!outFile) {
            std::cerr << "Cannot open output file: " << argv[4] << std::endl;
            return 1;
        }
        outFile << outputText;
        outFile.close();
    } else {
        // Write to stdout
        std::cout << outputText;
    }
    
    return 0;
}