/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <util/timer.h>

#include "util.h"
#include "texturing.h"
#include "mapmap/full.h"

TEX_NAMESPACE_BEGIN

const double NOT_VISIBLE_COST_FACTOR = 1.5;

bool
is_visible(int label, const DataCosts::Column & cost_column) {

    if (label == 0) return false;
    for(const auto & label_cost_pair : cost_column) {
        if (label_cost_pair.first == label - 1) return true;
    }
    return false;
}

void
view_selection(DataCosts const & data_costs, UniGraph * graph, Segmentation const & segmentation, Settings const &) {
    using uint_t = unsigned int;
    using cost_t = float;
    constexpr uint_t simd_w = mapmap::sys_max_simd_width<cost_t>();
    using unary_t = mapmap::UnaryTable<cost_t, simd_w>;
    using pairwise_t = mapmap::PairwisePotts<cost_t, simd_w>;


    const size_t num_faces = graph->num_nodes();

    const size_t num_segments = segmentation.get_num_segments();

    // assuming that segment zero means: no segment and that all other segments_ids are between 1 and num_segments
    const size_t NO_SEGMENT = 0;

    // to node ids
    size_t node_id_counter = num_segments - 1; // reserve first nodes for segment placeholders
    std::vector<std::size_t> node_ids(num_faces);
    std::vector<std::vector<std::size_t > > faces_of_segments(num_segments);
    for(size_t i = 0; i < segmentation.get_num_faces(); ++i) {
        auto segment_id = segmentation[i];
        assert(segment_id < num_segments);

        if (segment_id != NO_SEGMENT) {
            node_ids[i] = segment_id - 1; // uses the fact that segment zero means "no segment"!
        }
        else {
            node_ids[i] = node_id_counter++;
        }
        faces_of_segments[segment_id].push_back(i);
    }

    const size_t num_opti_nodes = node_id_counter;

    // accumulate segments
    std::vector<DataCosts::Column> segment_costs(num_segments - 1);
    for (size_t segment_id = 1; segment_id < num_segments; ++segment_id)
    {
        auto node_id = segment_id - 1;
        // view_statistics = mapping of view_id -> (num_visible_faces, summed_costs)
        std::map<std::uint16_t, std::pair<size_t, cost_t > > view_statistics;
        for(auto face : faces_of_segments[segment_id]) {
            DataCosts::Column const & data_costs_for_node = data_costs.col(face);

            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j) {
                auto & label_cost_pair = data_costs_for_node[j];
                auto & statistics = view_statistics[label_cost_pair.first];
                statistics.first += 1;
                statistics.second += label_cost_pair.second;
            }
        }

        size_t num_faces_in_segment = faces_of_segments[segment_id].size();
        std::stringstream view_info_string;
        // only allow views where all faces are visible
        for (auto & view_statistic : view_statistics) {
            std::uint16_t view_id = view_statistic.first;
            size_t num_visible_faces = view_statistic.second.first;
            size_t summed_cost = view_statistic.second.second;

            std::pair<size_t, cost_t> label_cost_pair(view_id, summed_cost);
            // face with missing views increases the costs. 1 is the maximal cost a normal face could have (ssee calculate_costs)
            // todo: better would be an area based penalty.
            label_cost_pair.second += NOT_VISIBLE_COST_FACTOR * (num_faces_in_segment - num_visible_faces);
            segment_costs[node_id].push_back(label_cost_pair);
            view_info_string << view_id << "(" << num_visible_faces << "," << summed_cost << "->" << label_cost_pair.second << ")\n";
        }

        if (view_info_string.str().empty()) {
            std::cout << "segment " << segment_id << " is not completely seen by any allowed view\n";
        }
        else {
            std::cout << "segment " << segment_id << " is visible in allowed views(num_faces, data_cost->costs_adjusted):\n" << view_info_string.str() << std::endl;
        }
    }


    /* Construct graph */
    mapmap::Graph<cost_t> mgraph(num_opti_nodes);

    for (std::size_t i = 0; i < num_faces; ++i) {
        if (data_costs.col(i).empty()) continue;

        size_t node_id = node_ids[i];

        std::vector<std::size_t> adj_faces = graph->get_adj_nodes(i);
        for (auto const & adj_edge : adj_faces) {
            std::size_t adj_face = adj_edge;
            auto adj_node_id = node_ids[adj_face];

            if ((adj_node_id >= num_segments - 1) && data_costs.col(adj_face).empty()) continue;
            if ((adj_node_id < num_segments - 1) && segment_costs[adj_node_id].empty()) continue;

            /* Uni directional */
            if (adj_face < i && adj_node_id != node_id) {
                mgraph.increase_edge_weight(adj_node_id, node_id, 1.0);
            }
        }
    }
    mgraph.update_components();


    mapmap::LabelSet<cost_t, simd_w> label_set(num_opti_nodes, false);

    // set possible labels for segments
    for (std::size_t segment_id = 0; segment_id < num_segments - 1; ++segment_id) {
        DataCosts::Column const & data_costs_for_node = segment_costs[segment_id];

        std::vector<mapmap::_iv_st<cost_t, simd_w> > labels;
        if (data_costs_for_node.empty()) {
            labels.push_back(0);
        } else {
            labels.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j) {
                labels[j] = data_costs_for_node[j].first + 1;
            }
        }

        label_set.set_label_set_for_node(segment_id, labels);
    }

    // set possible labels for unsegmented faces
    for (auto face_id : faces_of_segments[NO_SEGMENT]) {
        DataCosts::Column const & data_costs_for_node = data_costs.col(face_id);

        std::vector<mapmap::_iv_st<cost_t, simd_w> > labels;
        if (data_costs_for_node.empty()) {
            labels.push_back(0);
        } else {
            labels.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j) {
                labels[j] = data_costs_for_node[j].first + 1;
            }
        }

        label_set.set_label_set_for_node(node_ids[face_id], labels);
    }

    std::vector<unary_t> unaries;
    unaries.reserve(data_costs.cols());

    // set weights for segments
    for (std::size_t segment_id = 0; segment_id < num_segments - 1; ++segment_id) {
        DataCosts::Column const & data_costs_for_node = segment_costs[segment_id];

        std::vector<mapmap::_s_t<cost_t, simd_w> > costs;
        if (data_costs_for_node.empty()) {
            costs.push_back(1.0f);
        } else {
            costs.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j) {
                float cost = data_costs_for_node[j].second;
                costs[j] = cost;
            }

        }

        unaries.emplace_back(segment_id, &label_set);
        unaries.back().set_costs(costs);
    }

    // set weights for unsegmented faces
    for (auto face_id : faces_of_segments[NO_SEGMENT]) {
        DataCosts::Column const & data_costs_for_node = data_costs.col(face_id);

        std::vector<mapmap::_s_t<cost_t, simd_w> > costs;
        if (data_costs_for_node.empty()) {
            costs.push_back(1.0f);
        } else {
            costs.resize(data_costs_for_node.size());
            for(std::size_t j = 0; j < data_costs_for_node.size(); ++j) {
                float cost = data_costs_for_node[j].second;
                costs[j] = cost;
            }

        }

        unaries.emplace_back(node_ids[face_id], &label_set);
        unaries.back().set_costs(costs);
    }
    mapmap::StopWhenReturnsDiminish<cost_t, simd_w> terminate(5, 0.01);
    std::vector<mapmap::_iv_st<cost_t, simd_w> > solution;

    auto display = [](const mapmap::luint_t time_ms,
            const mapmap::_iv_st<cost_t, simd_w> objective) {
        std::cout << "\t\t" << time_ms / 1000 << "\t" << objective << std::endl;
    };

    /* Create mapMAP solver object. */
    mapmap::mapMAP<cost_t, simd_w> solver;
    solver.set_graph(&mgraph);
    solver.set_label_set(&label_set);
    for(std::size_t i = 0; i < graph->num_nodes(); ++i)
        solver.set_unary(i, &unaries[i]);
    pairwise_t pairwise(1.0f);
    solver.set_pairwise(&pairwise);
    solver.set_logging_callback(display);
    solver.set_termination_criterion(&terminate);

    /* Pass configuration arguments (optional) for solve. */
    mapmap::mapMAP_control ctr;
    ctr.use_multilevel = true;
    ctr.use_spanning_tree = true;
    ctr.use_acyclic = true;
    ctr.spanning_tree_multilevel_after_n_iterations = 5;
    ctr.force_acyclic = true;
    ctr.min_acyclic_iterations = 5;
    ctr.relax_acyclic_maximal = true;
    ctr.tree_algorithm = mapmap::LOCK_FREE_TREE_SAMPLER;

    /* Set true for deterministic (but slower) mapMAP execution. */
    ctr.sample_deterministic = false;
    ctr.initial_seed = 548923723;

    std::cout << "\tOptimizing:\n\t\tTime[s]\tEnergy" << std::endl;
    solver.optimize(solution, ctr);


    for (std::size_t i = 0; i < num_segments - 1; ++i) {
        std::cout << "segment " << (i+1) << ": solution " << solution[i] << ", label " << label_set.label_from_offset(i, solution[i]) << "\n";
    }
    /* Label 0 is undefined. */
    std::size_t num_labels = data_costs.rows() + 1;
    std::size_t undefined = 0;
    /* Extract resulting labeling from solver. */
    for (std::size_t i = 0; i < num_faces; ++i) {
        size_t node_id = node_ids[i];
        int label = label_set.label_from_offset(node_id, solution[node_id]);
        if (label < 0 || num_labels <= static_cast<std::size_t>(label)) {
            throw std::runtime_error("Incorrect labeling");
        }
        if (!is_visible(label, data_costs.col(i))) label = 0;
        if (label == 0) undefined += 1;
        graph->set_label(i, static_cast<std::size_t>(label));
    }
    std::cout << '\t' << undefined << " faces have not been seen" << std::endl;
}

TEX_NAMESPACE_END
