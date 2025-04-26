#include <iostream>
#include <fstream>
#include <string>
#include <regex>

// Function to read the entire file content
std::string readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    return std::string(std::istreambuf_iterator<char>(file), 
                      std::istreambuf_iterator<char>());
}

// Function to merge two MLIR modules into one
std::string mergeModules(const std::string& module1, const std::string& module2) {
    // Find the first module declaration
    size_t moduleStart1 = module1.find("module {");
    if (moduleStart1 == std::string::npos) {
        // No module declaration found, treat the entire string as module content
        moduleStart1 = 0;
    } else {
        moduleStart1 += 8; // Skip past "module {"
    }
    
    // Find the last closing brace in the first module
    int braceCount = 1;
    size_t moduleEnd1 = moduleStart1;
    
    // Only search for the matching brace if we found a module declaration
    if (moduleStart1 != 0) {
        for (size_t i = moduleStart1; i < module1.length(); i++) {
            if (module1[i] == '{') braceCount++;
            else if (module1[i] == '}') braceCount--;
            
            if (braceCount == 0) {
                moduleEnd1 = i;
                break;
            }
        }
    } else {
        moduleEnd1 = module1.length();
    }
    
    // Extract module1 content (without the outer braces if they exist)
    std::string module1Content;
    if (moduleStart1 == 0) {
        module1Content = module1;
    } else {
        module1Content = module1.substr(moduleStart1, moduleEnd1 - moduleStart1);
    }
    
    // Find the second module declaration
    size_t moduleStart2 = module2.find("module {");
    if (moduleStart2 == std::string::npos) {
        // No module declaration found, treat the entire string as module content
        moduleStart2 = 0;
    } else {
        moduleStart2 += 8; // Skip past "module {"
    }
    
    // Find the last closing brace in the second module
    braceCount = 1;
    size_t moduleEnd2 = moduleStart2;
    
    // Only search for the matching brace if we found a module declaration
    if (moduleStart2 != 0) {
        for (size_t i = moduleStart2; i < module2.length(); i++) {
            if (module2[i] == '{') braceCount++;
            else if (module2[i] == '}') braceCount--;
            
            if (braceCount == 0) {
                moduleEnd2 = i;
                break;
            }
        }
    } else {
        moduleEnd2 = module2.length();
    }
    
    // Extract module2 content (without the outer braces if they exist)
    std::string module2Content;
    if (moduleStart2 == 0) {
        module2Content = module2;
    } else {
        module2Content = module2.substr(moduleStart2, moduleEnd2 - moduleStart2);
    }
    
    // Combine the modules
    return "module {\n" + module1Content + "\n" + module2Content + "\n}";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <module1_file> <module2_file> [output_file]" << std::endl;
        std::cerr << "If output_file is not provided, writes to stdout" << std::endl;
        return 1;
    }
    
    try {
        // Read module files
        std::string module1Content = readFile(argv[1]);
        std::string module2Content = readFile(argv[2]);
        
        // Merge modules
        std::string mergedModule = mergeModules(module1Content, module2Content);
        
        // Output the result
        if (argc >= 4) {
            std::ofstream outFile(argv[3], std::ios::binary);
            if (!outFile) {
                std::cerr << "Cannot open output file: " << argv[3] << std::endl;
                return 1;
            }
            outFile << mergedModule;
            outFile.close();
        } else {
            std::cout << mergedModule;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}