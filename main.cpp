#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <functional>
#include <iomanip>
#include <algorithm>
#include <queue>

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

class BruteForce {
public:
    std::vector<VectorItem> items;

    void insert(const VectorItem& v) { items.push_back(v); }

    // O(N) linear search through every item
    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::vector<std::pair<float,int>> r;
        r.reserve(items.size());
        
        // Calculate distance to every single item
        for (auto& v : items) {
            r.push_back({dist(q, v.emb), v.id});
        }
        
        // Sort from closest to furthest
        std::sort(r.begin(), r.end());
        
        // Return only the top 'k' closest
        if ((int)r.size() > k) r.resize(k);
        return r;
    }
};


//  KD-TREE SEARCH (Multi-dimensional Binary Search)


struct KDNode {
    VectorItem item;
    KDNode* left  = nullptr;
    KDNode* right = nullptr;
    explicit KDNode(const VectorItem& v) : item(v) {}
};

class KDTree {
    KDNode* root = nullptr;
    int dims;

    // Recursively delete nodes to prevent memory leaks
    void destroy(KDNode* n) {
        if (!n) return; 
        destroy(n->left); 
        destroy(n->right); 
        delete n;
    }

    // Insert by branching left/right based on current dimension axis
    KDNode* ins(KDNode* n, const VectorItem& v, int d) {
        if (!n) return new KDNode(v);
        int ax = d % dims;
        if (v.emb[ax] < n->item.emb[ax]) 
            n->left  = ins(n->left,  v, d+1);
        else                              
            n->right = ins(n->right, v, d+1);
        return n;
    }

    // Recursive search maintaining a Priority Queue (Max-Heap) for top-k
    void knn(KDNode* n, const std::vector<float>& q, int k, int d, DistFn dist,
             std::priority_queue<std::pair<float,int>>& heap)
    {
        if (!n) return;
        
        float dn = dist(q, n->item.emb);
        
        // If heap isn't full, or this node is closer than the furthest node in heap
        if ((int)heap.size() < k || dn < heap.top().first) {
            heap.push({dn, n->item.id});
            if ((int)heap.size() > k) heap.pop(); // Kick out the worst one
        }
        
        int ax = d % dims;
        float diff = q[ax] - n->item.emb[ax];
        
        // Determine which side of the tree is likely closer
        KDNode* closer  = diff < 0 ? n->left  : n->right;
        KDNode* farther = diff < 0 ? n->right : n->left;
        
        knn(closer, q, k, d+1, dist, heap);
        
        // Only check the further side if it's mathematically possible a closer node exists there
        if ((int)heap.size() < k || std::abs(diff) < heap.top().first)
            knn(farther, q, k, d+1, dist, heap);
    }

public:
    explicit KDTree(int d) : dims(d) {}
    ~KDTree() { destroy(root); }

    void insert(const VectorItem& v) { root = ins(root, v, 0); }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, DistFn dist)
    {
        std::priority_queue<std::pair<float,int>> heap;
        knn(root, q, k, 0, dist, heap);
        
        std::vector<std::pair<float,int>> r;
        while (!heap.empty()) { 
            r.push_back(heap.top()); 
            heap.pop(); 
        }
        std::sort(r.begin(), r.end());
        return r;
    }
};

int main() {
    std::cout << "=== Vector Database: Search Test ===" << std::endl;

    BruteForce bf;
    KDTree kdt(DIMS);
    DistFn dist = getDistFn("cosine");

    // Load some mock data
    VectorItem itemA = {1, "Machine Learning", "Tech", {0.9f, 0.8f, 0.1f, 0.2f}};
    VectorItem itemB = {2, "Artificial Intelligence", "Tech", {0.8f, 0.9f, 0.2f, 0.1f}};
    VectorItem itemC = {3, "Cooking Pasta", "Food", {0.1f, 0.0f, 0.9f, 0.8f}};
    VectorItem itemD = {4, "Baking Bread", "Food", {0.0f, 0.1f, 0.8f, 0.9f}};

    bf.insert(itemA); kdt.insert(itemA);
    bf.insert(itemB); kdt.insert(itemB);
    bf.insert(itemC); kdt.insert(itemC);
    bf.insert(itemD); kdt.insert(itemD);

    // Create a query asking about food
    std::vector<float> query = {0.0f, 0.0f, 1.0f, 1.0f}; 

    std::cout << "\nSearching for 'Food' related concepts (K=2):\n";
    
    auto bf_results = bf.knn(query, 2, dist);
    std::cout << "\nBrute Force Results:\n";
    for(auto& hit : bf_results) {
        std::cout << "Found ID: " << hit.second << " (Distance: " << hit.first << ")\n";
    }

    auto kd_results = kdt.knn(query, 2, dist);
    std::cout << "\nKD-Tree Results:\n";
    for(auto& hit : kd_results) {
         std::cout << "Found ID: " << hit.second << " (Distance: " << hit.first << ")\n";
    }

    return 0;
}