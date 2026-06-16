#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <iomanip>

// Set to 4 for this basic math test so we don't have to type 16 numbers
static const int DIMS = 4; 

// Represents a single stored document and its numerical embedding
struct VectorItem {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> emb;
};

// Blueprint for distance metric functions
using DistFn = std::function<float(const std::vector<float>&, const std::vector<float>&)>;

// Straight-line distance (Pythagorean)
float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    for (int i = 0; i < (int)a.size(); i++) { 
        float d = a[i] - b[i]; 
        s += d * d; 
    }
    return std::sqrt(s);
}

// Angular distance (Standard for AI text embeddings)
float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < (int)a.size(); i++) {
        dot += a[i] * b[i]; 
        na += a[i] * a[i]; 
        nb += b[i] * b[i];
    }
    if (na < 1e-9f || nb < 1e-9f) return 1.0f;
    return 1.0f - dot / (std::sqrt(na) * std::sqrt(nb));
}

// Grid-based distance
float manhattan(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    for (int i = 0; i < (int)a.size(); i++) {
        s += std::abs(a[i] - b[i]);
    }
    return s;
}

// Factory function to select the math metric at runtime
DistFn getDistFn(const std::string& m) {
    if (m == "cosine")    return cosine;
    if (m == "manhattan") return manhattan;
    return euclidean;
}

int main() {
    std::cout << "- Vector Database: Core Math Test -" << std::endl;

    // Create two test vectors
    VectorItem itemA = {1, "Machine Learning", "Tech", {0.9f, 0.8f, 0.1f, 0.2f}};
    VectorItem itemB = {2, "Artificial Intelligence", "Tech", {0.8f, 0.9f, 0.2f, 0.1f}};
    
    // Create a vector that points in a completely different direction
    VectorItem itemC = {3, "Cooking Pasta", "Food", {0.1f, 0.0f, 0.9f, 0.8f}};

    std::cout << std::fixed << std::setprecision(4);

    std::cout << "\nComparing: [" << itemA.metadata << "] vs [" << itemB.metadata << "]\n";
    std::cout << "Euclidean Distance: " << euclidean(itemA.emb, itemB.emb) << std::endl;
    std::cout << "Manhattan Distance: " << manhattan(itemA.emb, itemB.emb) << std::endl;
    std::cout << "Cosine Distance:    " << cosine(itemA.emb, itemB.emb) << " (how close to 0 this is)\n";

    std::cout << "\nComparing: [" << itemA.metadata << "] vs [" << itemC.metadata << "]\n";
    std::cout << "Euclidean Distance: " << euclidean(itemA.emb, itemC.emb) << std::endl;
    std::cout << "Manhattan Distance: " << manhattan(itemA.emb, itemC.emb) << std::endl;
    std::cout << "Cosine Distance:    " << cosine(itemA.emb, itemC.emb) << " ( how far from 0 this is)\n";

    return 0;
}