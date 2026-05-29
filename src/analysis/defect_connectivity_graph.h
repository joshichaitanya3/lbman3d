#ifndef LBM_AN_ANALYSIS_DEFECT_CONNECTIVITY_GRAPH_H
#define LBM_AN_ANALYSIS_DEFECT_CONNECTIVITY_GRAPH_H

#include <unordered_map>
#include <set>
#include <cstdint>
#include <optional>
#include <iostream>

using FaceId = std::uint64_t;

class UndirectedGraph {

    std::unordered_map<FaceId, std::set<FaceId>> edge_map;
    
    /* !\brief Performs Hierholzer's algorithm from a starting node
        to find one Eulerian path component.
    */
    std::vector<FaceId> HierholzerAlgorithm(FaceId start_node){
    
        std::vector<FaceId> path = { start_node };
        while(true) {
    
            FaceId u = path.back();
            if (edge_map[u].empty()) break;
            
            FaceId v = *edge_map[u].begin();
            path.push_back(v);
            RemoveEdge(u, v);
        }
    
        return path;
    }

public:
    /* !\brief Adds an undirected edge between nodes u and v.
     */
    void AddEdge(FaceId u, FaceId v) {

        if (u == v) {
            std::cerr << "Warning: Received " << u << " and " << v << " as input to AddEdge." << std::endl; 
            std::cerr << "Self-loops are not supported for this graph!" << std::endl;
            return;
        }
        edge_map[u].insert(v);
        edge_map[v].insert(u);
    }

    /* !\brief Removes an undirected edge between nodes u and v, if it exists.
     */
    void RemoveEdge(FaceId u, FaceId v) {

        if (u == v) {
            std::cerr << "Warning: Received " << u << " and " << v << " as input to RemoveEdge." << std::endl; 
            std::cerr << "Self-loops are not supported for this graph!" << std::endl;
            return;
        }
        edge_map[u].erase(v);
        edge_map[v].erase(u);
    }

    /* !\brief Find a valid starting node for Hierholzer's algorithm.
     *
     * Returns a node with odd degree (if any), or the first non-isolated node.
     * Returns std::nullopt if the graph has no edges.
     */
    std::optional<FaceId> FindStartNode() const {
        // Prefer node with odd degree (for Eulerian path)
        for (const auto& [node, neighbors] : edge_map) {
            if (neighbors.size() % 2 == 1) {
                return node;
            }
        }
        // Fallback: find any node with at least one neighbor
        for (const auto& [node, neighbors] : edge_map) {
            if (!neighbors.empty()) {
                return node;
            }
        }
        return std::nullopt;
    }


    /* !\brief Decompose the graph into a collection of Eulerian path components.
    */
    std::vector<std::vector<FaceId>> FindPaths() {
        std::vector<std::vector<FaceId>> paths;

        while (true) {
            auto start_node = FindStartNode();
            if (!start_node) break;

            paths.push_back(HierholzerAlgorithm(*start_node));
        }

        return paths;
    }

};

#endif // LBM_AN_ANALYSIS_DEFECT_CONNECTIVITY_GRAPH_H
