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
#include "httplib.h"
 

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
    void remove(int id) {
        items.erase(std::remove_if(items.begin(), items.end(),
            [id](const VectorItem& v){ return v.id == id; }), items.end());
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

// vectorDB

class VectorDB {
    std::unordered_map<int, VectorItem> store;
    BruteForce bf;
    KDTree     kdt;
    HNSW       hnsw;
    std::mutex mu;
    int nextId = 1;

public:
    const int dims;
    explicit VectorDB(int d) : kdt(d), hnsw(16, 200), dims(d) {}

    int insert(const std::string& meta, const std::string& cat,
               const std::vector<float>& emb, DistFn dist)
    {
        std::lock_guard<std::mutex> lk(mu);
        VectorItem v{nextId++, meta, cat, emb};
        store[v.id] = v;
        bf.insert(v); kdt.insert(v); hnsw.insert(v, dist);
        return v.id;
    }

    bool remove(int id) {
        std::lock_guard<std::mutex> lk(mu);
        if (!store.count(id)) return false;
        store.erase(id); bf.remove(id); hnsw.remove(id);
        std::vector<VectorItem> rem;
        for (auto& [i, v] : store) rem.push_back(v);
        kdt.rebuild(rem);
        return true;
    }

    struct Hit { int id; std::string meta, cat; std::vector<float> emb; float dist; };
    struct SearchOut { std::vector<Hit> hits; long long us; std::string algo, metric; };

    SearchOut search(const std::vector<float>& q, int k,
                     const std::string& metric, const std::string& algo)
    {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn = getDistFn(metric);
        auto t0  = std::chrono::high_resolution_clock::now();

        std::vector<std::pair<float,int>> raw;
        if      (algo == "bruteforce") raw = bf.knn(q, k, dfn);
        else if (algo == "kdtree")     raw = kdt.knn(q, k, dfn);
        else                           raw = hnsw.knn(q, k, 50, dfn);

        long long us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t0).count();

        SearchOut out; out.us = us; out.algo = algo; out.metric = metric;
        for (auto& [d, id] : raw)
            if (store.count(id))
                out.hits.push_back({id, store[id].metadata, store[id].category, store[id].emb, d});
        return out;
    }

    struct BenchOut { long long bfUs, kdUs, hnswUs; int n; };

    BenchOut benchmark(const std::vector<float>& q, int k, const std::string& metric) {
        std::lock_guard<std::mutex> lk(mu);
        auto dfn  = getDistFn(metric);
        auto time = [&](auto fn) -> long long {
            auto t = std::chrono::high_resolution_clock::now();
            fn();
            return std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - t).count();
        };
        return {
            time([&]{ bf.knn(q, k, dfn); }),
            time([&]{ kdt.knn(q, k, dfn); }),
            time([&]{ hnsw.knn(q, k, 50, dfn); }),
            (int)store.size()
        };
    }
};

//  JSON HELPERS

std::string jS(const std::string& s) {

    std::string o = "\"";

    for (char c : s) {

        if      (c == '"')  o += "\\\"";

        else if (c == '\\') o += "\\\\";

        else if (c == '\n') o += "\\n";

        else if (c == '\r') o += "\\r";

        else if (c == '\t') o += "\\t";

        else                o += c;

    }

    return o + '"';

}



std::string jVec(const std::vector<float>& v) {

    std::ostringstream ss; ss << '[';

    for (size_t i = 0; i < v.size(); i++) {

        if (i) ss << ',';

        ss << std::fixed << std::setprecision(4) << v[i];

    }

    return ss.str() + ']';
}



std::vector<float> parseVec(const std::string& s) {

    std::vector<float> v;
    std::istringstream ss(s); std::string t;

    while (std::getline(ss, t, ','))

        try { v.push_back(std::stof(t)); } catch (...) {}

    return v;

}



// extract a JSON string field value (handles basic escape sequences)

std::string extractStr(const std::string& body, const std::string& key) {

    size_t p = body.find('"' + key + '"');

    if (p == std::string::npos) return "";

    p = body.find(':', p) + 1;

    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) p++;

    if (p >= body.size() || body[p] != '"') return "";

    p++;

    std::string result;

    while (p < body.size()) {

        if (body[p] == '"') break;

        if (body[p] == '\\' && p + 1 < body.size()) {

            p++;

            switch (body[p]) {

                case '"':  result += '"';  break;

                case '\\': result += '\\'; break;

                case 'n':  result += '\n'; break;

                case 'r':  result += '\r'; break;

                case 't':  result += '\t'; break;

                default:   result += body[p]; break;

            }

        } else {

            result += body[p];

        }

        p++;

    }

    return result;

}



// Extract a JSON integer field value
int extractInt(const std::string& body, const std::string& key, int def = 0) {

    size_t p = body.find('"' + key + '"');

    if (p == std::string::npos) return def;

    p = body.find(':', p) + 1;

    while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) p++;

    try { return std::stoi(body.substr(p)); } catch (...) { return def; }

}



bool parseBody(const std::string& b, std::string& meta,

                std::string& cat, std::vector<float>& emb)

{

    meta = extractStr(b, "metadata");

    cat  = extractStr(b, "category");

    auto extractArr = [&](const std::string& key) -> std::vector<float> {

        size_t p = b.find('"' + key + '"');

        if (p == std::string::npos) return {};

        p = b.find('[', p);

        if (p == std::string::npos) return {};

        size_t e = b.find(']', p);

        if (e == std::string::npos) return {};

        return parseVec(b.substr(p + 1, e - p - 1));

    };

    emb = extractArr("embedding");

    return !meta.empty() && !emb.empty();

}



void cors(httplib::Response& res) {

    res.set_header("Access-Control-Allow-Origin",  "*");

    res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");

    res.set_header("Access-Control-Allow-Headers", "Content-Type");

}  

int main() {
    VectorDB db(4); // Testing with 4d vectors
    auto distFn = getDistFn("cosine");
    httplib::Server svr;

    // Handle CORS preflight requests for frontend compatibility
    svr.Options(".*", [](const httplib::Request&, httplib::Response& res) { 
        cors(res); 
    });

    svr.Post("/insert", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        std::string meta, cat; 
        std::vector<float> emb;
        
        if (!parseBody(req.body, meta, cat, emb) || emb.size() != db.dims) {
            res.status = 400;
            res.set_content("{\"error\":\"Invalid format or dimension mismatch\"}", "application/json");
            return;
        }
        
        int id = db.insert(meta, cat, emb, distFn);
        res.set_content("{\"id\":" + std::to_string(id) + "}", "application/json");
    });

    svr.Post("/search", [&](const httplib::Request& req, httplib::Response& res) {
        cors(res);
        int k = extractInt(req.body, "k", 2);
        std::string algo = extractStr(req.body, "algo");
        if (algo.empty()) algo = "hnsw";

        // Extract the query array manually using string positions
        size_t p = req.body.find("\"query\"");
        p = req.body.find('[', p);
        size_t e = req.body.find(']', p);
        std::vector<float> q = parseVec(req.body.substr(p + 1, e - p - 1));

        auto out = db.search(q, k, "cosine", algo);

        // Manually build the JSON response
        std::string json = "{\"time_us\":" + std::to_string(out.us) + ",\"algo\":" + jS(out.algo) + ",\"hits\":[";
        for (size_t i = 0; i < out.hits.size(); i++) {
            if (i > 0) json += ",";
            json += "{\"id\":" + std::to_string(out.hits[i].id) + 
                    ",\"meta\":" + jS(out.hits[i].meta) + 
                    ",\"cat\":" + jS(out.hits[i].cat) + 
                    ",\"dist\":" + std::to_string(out.hits[i].dist) + "}";
        }
        json += "]}";
        
        res.set_content(json, "application/json");
    });

    std::cout << "VectorDB Server booting up...\n";
    std::cout << "Listening on http://localhost:8080\n";
    
    svr.listen("0.0.0.0", 8080);
    return 0;
}