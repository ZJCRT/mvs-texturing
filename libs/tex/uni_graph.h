/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef TEX_UNIGRAPH_HEADER
#define TEX_UNIGRAPH_HEADER

#include <vector>
#include <cassert>
#include <algorithm>

/**
  * Implementation of a unidirectional graph with fixed amount of nodes using adjacency lists.
  */
class UniGraph {
    public:
        using Cost = float;
        using WeightedEdge = std::pair<std::size_t, Cost>;

    private:
        std::vector<std::vector<WeightedEdge> > adj_lists;
        std::vector<std::size_t> labels;
        std::size_t edges;

    public:
        /**
          * Creates a unidirectional graph without edges.
          * @param nodes number of nodes.
          */
        UniGraph(std::size_t nodes);

        /**
          * Adds an edge between the nodes with indices n1 and n2.
          * If the edge exists weight is added.
          * @warning asserts that the indices are valid.
          */
        void add_edge(std::size_t n1, std::size_t n2, Cost weight = 1.0);

        /**
          * Removes the edge between the nodes with indices n1 and n2.
          * If the edge does not exist nothing happens.
          * @warning asserts that the indices are valid.
          */
        void remove_edge(std::size_t n1, std::size_t n2);

        /**
          * Returns true if an edge between the nodes with indices n1 and n2 exists.
          * @warning asserts that the indices are valid.
          */
        bool has_edge(std::size_t n1, std::size_t n2) const;

        /** Returns the number of edges. */
        std::size_t num_edges() const;

        /** Returns the number of nodes. */
        std::size_t num_nodes() const;

        /**
          * Sets the label of node with index n to label.
          * @warning asserts that the index is valid.
          */
        void set_label(std::size_t n, std::size_t label);

        /**
          * Returns the label of node with index n.
          * @warning asserts that the index is valid.
          */
        std::size_t get_label(std::size_t n) const;

        /**
          * Fills given vector with all subgraphs of the given label.
          * A subgraph is a vector containing all indices of connected nodes with the same label.
          */
        void get_subgraphs(std::size_t label, std::vector<std::vector<std::size_t> > * subgraphs) const;

        std::vector<WeightedEdge> const & get_adj_nodes(std::size_t node) const;
};

inline void
UniGraph::add_edge(std::size_t n1, std::size_t n2, Cost weight /* = 1.0 */) {
    assert(n1 < num_nodes() && n2 < num_nodes());

    auto & n1_edges = adj_lists[n1];
    auto is_connected_to_n2 = [n2](const WeightedEdge & e) { return e.first == n2;};
    auto edge_from_n1 = std::find_if(n1_edges.begin(), n1_edges.end(), is_connected_to_n2);

    if (edge_from_n1 == n1_edges.end()) {
        adj_lists[n1].emplace_back(n2, weight);
        adj_lists[n2].emplace_back(n1, weight);
        ++edges;
    }
    else
    {
        auto & n2_edges = adj_lists[n2];
        auto is_connected_to_n1 = [n1](const WeightedEdge & e) { return e.first == n1; };
        auto edge_from_n2 = std::find_if(n2_edges.begin(), n2_edges.end(), is_connected_to_n1);

        assert(edge_from_n1->second == edge_from_n2->second);

        edge_from_n1->second += weight;
        edge_from_n2->second += weight;
    }
}

inline bool
UniGraph::has_edge(std::size_t n1, std::size_t n2) const {
    assert(n1 < num_nodes() && n2 < num_nodes());

    std::vector<WeightedEdge> const & adj_list = adj_lists[n1];
    auto is_connected_to_n2 = [n2](const WeightedEdge & e) { return e.first == n2;};
    return std::find_if(adj_list.begin(), adj_list.end(), is_connected_to_n2) != adj_list.end();
}

inline std::size_t
UniGraph::num_edges() const {
    return edges;
}

inline std::size_t
UniGraph::num_nodes() const {
    return adj_lists.size();
}

inline void
UniGraph::set_label(std::size_t n, std::size_t label) {
    assert(n < num_nodes());
    labels[n] = label;
}

inline std::size_t
UniGraph::get_label(std::size_t n) const {
    assert(n < num_nodes());
    return labels[n];
}

inline std::vector<UniGraph::WeightedEdge> const &
UniGraph::get_adj_nodes(std::size_t node) const {
    assert(node < num_nodes());
    return adj_lists[node];
}

#endif /* TEX_UNIGRAPH_HEADER */
