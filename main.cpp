#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <queue>
#include <set>
#include <sstream>
#include <iomanip>
#include <functional>
#include <fstream>
#include <climits>


static const int DIMS = 4; 

struct VectorItem {
    int id;
    std::string metadata;
    std::string category;
    std::vector<float> emb;
};

// Blueprint for distance metric functions
using DistFn = std::function<float(const std::vector<float>&, const std::vector<float>&)>;


float euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    float s = 0;
    for (int i = 0; i < (int)a.size(); i++) { 
        float d = a[i] - b[i]; 
        s += d * d; 
    }
    return std::sqrt(s);
}


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
    void rebuild(const std::vector<VectorItem>& items) {
        destroy(root); root = nullptr;
        for (auto& v : items) insert(v);
    }
};

class HNSW {
    struct Node {
        VectorItem item;
        int maxLyr;
        std::vector<std::vector<int>> nbrs;
    };

    std::unordered_map<int, Node> G;
    int    M, M0, ef_build;
    float  mL;
    int    topLayer = -1;
    int    entryPt  = -1;
    std::mt19937 rng;

    int randLevel() {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        return (int)std::floor(-std::log(u(rng)) * mL);
    }

    std::vector<std::pair<float,int>> searchLayer(
        const std::vector<float>& q, int ep, int ef, int lyr, DistFn dist)
    {
        std::unordered_map<int,bool> vis;
        std::priority_queue<std::pair<float,int>,
            std::vector<std::pair<float,int>>, std::greater<>> cands;
        std::priority_queue<std::pair<float,int>> found;

        float d0 = dist(q, G[ep].item.emb);
        vis[ep] = true;
        cands.push({d0, ep});
        found.push({d0, ep});

        while (!cands.empty()) {
            auto [cd, cid] = cands.top(); cands.pop();
            if ((int)found.size() >= ef && cd > found.top().first) break;
            if (lyr >= (int)G[cid].nbrs.size()) continue;
            for (int nid : G[cid].nbrs[lyr]) {
                if (vis[nid] || !G.count(nid)) continue;
                vis[nid] = true;
                float nd = dist(q, G[nid].item.emb);
                if ((int)found.size() < ef || nd < found.top().first) {
                    cands.push({nd, nid});
                    found.push({nd, nid});
                    if ((int)found.size() > ef) found.pop();
                }
            }
        }

        std::vector<std::pair<float,int>> res;
        while (!found.empty()) { res.push_back(found.top()); found.pop(); }
        std::sort(res.begin(), res.end());
        return res;
    }

    std::vector<int> selectNbrs(std::vector<std::pair<float,int>>& cands, int maxM) {
        std::vector<int> r;
        for (int i = 0; i < std::min((int)cands.size(), maxM); i++)
            r.push_back(cands[i].second);
        return r;
    }

public:
    HNSW(int m = 16, int efBuild = 200)
        : M(m), M0(2*m), ef_build(efBuild),
          mL(1.0f / std::log((float)m)), rng(42) {}

    void insert(const VectorItem& item, DistFn dist) {
        int id  = item.id;
        int lvl = randLevel();
        G[id]   = {item, lvl, std::vector<std::vector<int>>(lvl + 1)};

        if (entryPt == -1) { entryPt = id; topLayer = lvl; return; }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        for (int lc = std::min(topLayer, lvl); lc >= 0; lc--) {
            auto W   = searchLayer(item.emb, ep, ef_build, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            auto sel = selectNbrs(W, maxM);
            G[id].nbrs[lc] = sel;

            for (int nid : sel) {
                if (!G.count(nid)) continue;
                if ((int)G[nid].nbrs.size() <= lc) G[nid].nbrs.resize(lc + 1);
                auto& conn = G[nid].nbrs[lc];
                conn.push_back(id);
                if ((int)conn.size() > maxM) {
                    std::vector<std::pair<float,int>> ds;
                    for (int c : conn) if (G.count(c))
                        ds.push_back({dist(G[nid].item.emb, G[c].item.emb), c});
                    std::sort(ds.begin(), ds.end());
                    conn.clear();
                    for (int i = 0; i < maxM && i < (int)ds.size(); i++)
                        conn.push_back(ds[i].second);
                }
            }
            if (!W.empty()) ep = W[0].second;
        }
        if (lvl > topLayer) { topLayer = lvl; entryPt = id; }
    }

    std::vector<std::pair<float,int>> knn(
        const std::vector<float>& q, int k, int ef, DistFn dist)
    {
        if (entryPt == -1) return {};
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            if (lc < (int)G[ep].nbrs.size()) {
                auto W = searchLayer(q, ep, 1, lc, dist);
                if (!W.empty()) ep = W[0].second;
            }
        }
        auto W = searchLayer(q, ep, std::max(ef, k), 0, dist);
        if ((int)W.size() > k) W.resize(k);
        return W;
    }

    void remove(int id) {
        if (!G.count(id)) return;
        for (auto& [nid, nd] : G)
            for (auto& layer : nd.nbrs)
                layer.erase(std::remove(layer.begin(), layer.end(), id), layer.end());
        if (entryPt == id) {
            entryPt = -1;
            for (auto& [nid, nd] : G) if (nid != id) { entryPt = nid; break; }
        }
        G.erase(id);
    }

    struct GraphInfo {
        int topLayer, nodeCount;
        std::vector<int> nodesPerLayer, edgesPerLayer;
        struct NV { int id; std::string metadata, category; int maxLyr; };
        struct EV { int src, dst, lyr; };
        std::vector<NV> nodes;
        std::vector<EV> edges;
    };

    GraphInfo getInfo() {
        GraphInfo gi;
        gi.topLayer  = topLayer;
        gi.nodeCount = (int)G.size();
        int maxL = std::max(topLayer + 1, 1);
        gi.nodesPerLayer.assign(maxL, 0);
        gi.edgesPerLayer.assign(maxL, 0);
        for (auto& [id, nd] : G) {
            gi.nodes.push_back({id, nd.item.metadata, nd.item.category, nd.maxLyr});
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer[lc]++;
                if (lc < (int)nd.nbrs.size())
                    for (int nid : nd.nbrs[lc])
                        if (id < nid) {
                            gi.edgesPerLayer[lc]++;
                            gi.edges.push_back({id, nid, lc});
                        }
            }
        }
        return gi;
    }

    size_t size() const { return G.size(); }
};

int main() {
    std::cout << "Booting up VectorDB engine tests...\n\n";

    
    BruteForce bf;
    KDTree kdt(DIMS);
    HNSW hnsw(16, 200); // M=16, ef_build=200
    
    auto cosineDist = getDistFn("cosine");

    std::vector<VectorItem> docs = {
        {1, "Machine Learning", "Tech", {0.9f, 0.8f, 0.1f, 0.2f}},
        {2, "Artificial Intelligence", "Tech", {0.8f, 0.9f, 0.2f, 0.1f}},
        {3, "Data Science", "Tech", {0.85f, 0.85f, 0.15f, 0.15f}},
        {4, "Cooking Pasta", "Food", {0.1f, 0.0f, 0.9f, 0.8f}},
        {5, "Baking Bread", "Food", {0.0f, 0.1f, 0.8f, 0.9f}},
        {6, "Grilling Steak", "Food", {0.2f, 0.1f, 0.85f, 0.85f}}
    };

    std::cout << "Indexing " << docs.size() << " documents...\n";
    for (const auto& doc : docs) {
        bf.insert(doc);
        kdt.insert(doc);
        hnsw.insert(doc, cosineDist);
    }

    
    auto info = hnsw.getInfo();
    std::cout << "HNSW built successfully. Top layer reached: " << info.topLayer << "\n\n";

    
    std::vector<float> query = {0.0f, 0.0f, 1.0f, 1.0f}; 
    int k = 2; 

    std::cout << "Searching for 'Food' (K=" << k << ")...\n\n";
    
    
    std::cout << "[Brute Force Baseline]\n";
    for(auto& hit : bf.knn(query, k, cosineDist)) {
        std::cout << "  -> Doc ID: " << hit.second << " (dist: " << hit.first << ")\n";
    }

    
    std::cout << "\n[KD-Tree]\n";
    for(auto& hit : kdt.knn(query, k, cosineDist)) {
         std::cout << "  -> Doc ID: " << hit.second << " (dist: " << hit.first << ")\n";
    }

    
    std::cout << "\n[HNSW (ef=50)]\n";
    for(auto& hit : hnsw.knn(query, k, 50, cosineDist)) {
         std::cout << "  -> Doc ID: " << hit.second << " (dist: " << hit.first << ")\n";
    }

    std::cout << "\nDone. If HNSW matches Brute Force, the graph routing works perfectly!\n";

    return 0;
}