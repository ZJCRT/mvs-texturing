/*
 * Copyright (C) 2015, Nils Moehrle
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef TEX_TEXTURING_HEADER
#define TEX_TEXTURING_HEADER

#include <vector>
#include <map>

#include "mve/mesh.h"
#include "mve/mesh_info.h"

#include "defines.h"
#include "settings.h"
#include "obj_model.h"
#include "uni_graph.h"
#include "texture_view.h"
#include "texture_patch.h"
#include "texture_atlas.h"
#include "sparse_table.h"

#include "seam_leveling.h"

TEX_NAMESPACE_BEGIN

typedef std::vector<TextureView> TextureViews;
typedef std::vector<TexturePatch::Ptr> TexturePatches;
typedef std::vector<TextureAtlas::Ptr> TextureAtlases;
typedef ObjModel Model;
typedef UniGraph Graph;
typedef SparseTable<std::uint32_t, std::uint16_t, float> DataCosts;
typedef std::vector<std::vector<VertexProjectionInfo> > VertexProjectionInfos;
typedef std::vector<std::vector<FaceProjectionInfo> > FaceProjectionInfos;

/** the Segmentation structure stores the selected segment for each face. */
class Segmentation
{
    std::vector<std::uint16_t> segments;
    std::uint16_t num_segments;

public:

    /// Faces with DEFAULT_SEGMENT_ID do not neccesarily be clustered to the same view
    static const std::uint16_t DEFAULT_SEGMENT_ID = 0; 

    /// If a view does not have a segmentation, its face-view-relation is unclassified.
    /// Attention: The value UNCLASSIFIED should normally NOT be used in this class. 
    /// it is intended for 2d-segmentation-usage only.
    static const std::uint16_t UNCLASSIFIED = std::numeric_limits<std::uint16_t>::max();

    Segmentation()
     : num_segments(0)
    {}

    std::uint16_t operator[](size_t idx) const { return segments[idx]; }
    std::uint16_t & operator[](size_t idx) { return segments[idx]; }

    /** set num faces (call it before accessing via [] !) */
    void set_num_faces(size_t new_size) { segments.resize(new_size); }
    size_t get_num_faces() const { return segments.size(); }

    /** set num segments. just informal. should include the zero segment. */
    void set_num_segments(std::uint16_t new_num_segments) { num_segments = new_num_segments; }
    std::uint16_t get_num_segments() const { return num_segments; }
};

/** a list of view_id representing the allowed views per segment.
 *  unlisted segments are not restricted. */
using ViewsPerSegment = std::map< std::uint16_t, std::vector<std::uint16_t> >;

/**
  * prepares the mesh for texturing
  *  -removes duplicated faces
  *  -ensures normals (face and vertex)
  */
void
prepare_mesh(mve::MeshInfo * mesh_info, mve::TriangleMesh::Ptr mesh);

/**
  * Generates TextureViews from the in_scene.
  */
void
generate_texture_views(std::string const & in_scene,
    TextureViews * texture_views, std::string const & tmp_dir, std::string const & segmentation_image_dir);

/**
  * Builds up the meshes face adjacency graph using the vertex_infos
  */
void
build_adjacency_graph(mve::TriangleMesh::ConstPtr mesh,
    mve::MeshInfo const & mesh_info, UniGraph * graph);

/**
 * Calculates the data costs for each face and texture view combination,
 * if the face is visible within the texture view.
 */
void
calculate_data_costs(mve::TriangleMesh::ConstPtr mesh,
    TextureViews * texture_views, Settings const & settings,
    ViewsPerSegment const & views_per_segment,
    Segmentation * segmentation, DataCosts * data_costs);

void
postprocess_face_infos(Settings const & settings,
    ViewsPerSegment const & views_per_segment,
    FaceProjectionInfos * projected_face_infos,
    DataCosts * data_costs);

/**
 * Runs the view selection procedure and saves the labeling in the graph
 */
void
view_selection(DataCosts const & data_costs, UniGraph * graph, Segmentation const & segmentation, Settings const & settings);

/**
  * Generates texture patches using the graph to determine adjacent faces with the same label.
  */
void generate_texture_patches(UniGraph const & graph,
    mve::TriangleMesh::ConstPtr mesh,
    mve::MeshInfo const & mesh_info,
    TextureViews * texture_views,
    Settings const & settings,
    VertexProjectionInfos * vertex_projection_infos,
    TexturePatches * texture_patches);

/**
  * Runs the seam leveling procedure proposed by Ivanov and Lempitsky
  * [<A HREF="https://www.google.de/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&sqi=2&ved=0CC8QFjAA&url=http%3A%2F%2Fwww.robots.ox.ac.uk%2F~vilem%2FSeamlessMosaicing.pdf&ei=_ZbvUvSZIaPa4ASi7IGAAg&usg=AFQjCNGd4x5HnMMR68Sn2V5dPgmqJWErCA&sig2=4j47bXgovw-uks9LBGl_sA">Seamless mosaicing of image-based texture maps</A>]
  */
void
global_seam_leveling(UniGraph const & graph, mve::TriangleMesh::ConstPtr mesh,
    mve::MeshInfo const & mesh_info,
    VertexProjectionInfos const & vertex_projection_infos,
    TexturePatches * texture_patches);

void
local_seam_leveling(UniGraph const & graph, mve::TriangleMesh::ConstPtr mesh,
    VertexProjectionInfos const & vertex_projection_infos,
    TexturePatches * texture_patches);

void
generate_texture_atlases(TexturePatches * texture_patches,
    Settings const & settings, TextureAtlases * texture_atlases, unsigned int resolution = 0);

void
generate_single_atlas(TexturePatches * texture_patches,
    Settings const & settings, TextureAtlases * texture_atlases);

/**
  * Builds up an model for the mesh by constructing materials and
  * texture atlases form the texture_patches
  */
void
build_model(mve::TriangleMesh::ConstPtr mesh,
    TextureAtlases const & texture_atlas, Model * model);

TEX_NAMESPACE_END

#endif /* TEX_TEXTURING_HEADER */
