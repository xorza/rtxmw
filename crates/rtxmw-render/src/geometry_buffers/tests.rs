use super::*;
use glam::{Vec2, Vec3};
use rtxmw_scene::Submesh;

/// Packs into an empty arena, which is what most of these check.
fn packed(meshes: &[Mesh]) -> Packed {
    pack(&meshes.iter().collect::<Vec<_>>(), 0, 0, 0)
}

/// A mesh whose vertices and normals are distinguishable by index, and whose run is marked a
/// sheet or not so that packing has a value to carry rather than a constant to reproduce.
fn mesh(vertices: u32, indices: &[u32], thin: bool) -> Mesh {
    Mesh {
        positions: (0..vertices).map(|i| Vec3::splat(i as f32)).collect(),
        normals: (0..vertices).map(|i| Vec3::X * i as f32).collect(),
        uvs: (0..vertices).map(|i| Vec2::splat(i as f32 * 0.5)).collect(),
        indices: indices.to_vec(),
        // Empty in, empty out: flattening only opens a run when a block contributes triangles,
        // so a mesh with no geometry has no submeshes rather than one covering nothing.
        submeshes: if indices.is_empty() {
            Vec::new()
        } else {
            vec![Submesh {
                first_index: 0,
                index_count: indices.len() as u32,
                material: 0,
                thin,
            }]
        },
    }
}

#[test]
fn packing_concatenates_meshes_and_leaves_indices_mesh_local() {
    // 3 vertices / 1 triangle, then 4 vertices / 2 triangles.
    let meshes = [
        mesh(3, &[0, 1, 2], true),
        mesh(4, &[0, 1, 2, 1, 2, 3], false),
    ];
    let Packed {
        positions,
        attributes,
        indices,
        ranges,
        submeshes,
    } = packed(&meshes);

    assert_eq!(
        ranges,
        vec![
            MeshRange {
                first_vertex: 0,
                vertex_count: 3,
                first_index: 0,
                index_count: 3,
                first_submesh: 0,
                submesh_count: 1,
            },
            MeshRange {
                first_vertex: 3,
                vertex_count: 4,
                first_index: 3,
                index_count: 6,
                first_submesh: 1,
                submesh_count: 1,
            },
        ]
    );
    assert_eq!(positions.len(), 7);
    assert_eq!(attributes.len(), 7);
    assert_eq!(indices.len(), 9);
    assert_eq!(ranges[1].triangle_count(), 2);

    // The second mesh starts at vertex 3, but its indices are unchanged — 0..3, not 3..6. A
    // rebased buffer would read the first mesh's vertices instead.
    assert_eq!(&indices[3..], &[0, 1, 2, 1, 2, 3]);
    // Submesh offsets, unlike the indices, *are* rebased onto the shared buffer — a build range
    // addresses the whole buffer, so the second mesh's run starts at 3 rather than 0.
    assert_eq!(
        submeshes,
        vec![
            SubmeshRange {
                first_index: 0,
                index_count: 3,
                material: 0,
                thin: true
            },
            SubmeshRange {
                first_index: 3,
                index_count: 6,
                material: 0,
                thin: false
            },
        ]
    );
    // Its vertices did move, so position 3 is the second mesh's vertex 0.
    assert_eq!(positions[3], [0.0, 0.0, 0.0]);
    assert_eq!(positions[6], [3.0, 3.0, 3.0]);
}

#[test]
fn an_append_lands_past_what_the_arena_already_holds() {
    // An arena holding 10 vertices, 30 indices and 2 submeshes takes a 3-vertex, 1-triangle
    // mesh: it starts at vertex 10 and index 30, and its submesh is the third.
    let meshes = [mesh(3, &[0, 1, 2], false)];
    let out = pack(&meshes.iter().collect::<Vec<_>>(), 10, 30, 2);

    assert_eq!(
        out.ranges[0],
        MeshRange {
            first_vertex: 10,
            vertex_count: 3,
            first_index: 30,
            index_count: 3,
            first_submesh: 2,
            submesh_count: 1,
        }
    );
    // The submesh counts from the start of the whole buffer, not of its mesh.
    assert_eq!(out.submeshes[0].first_index, 30);
    // Index *values* stay mesh-local, which is what lets a mesh be appended anywhere without
    // rewriting its indices — the build adds `first_vertex` instead.
    assert_eq!(out.indices, vec![0, 1, 2]);
}

#[test]
fn attributes_stay_aligned_with_the_positions_they_belong_to() {
    let meshes = [mesh(2, &[0, 1, 1], false), mesh(2, &[0, 1, 1], false)];
    let out = packed(&meshes);
    let (positions, attributes) = (out.positions, out.attributes);

    // Vertex 1 of the second mesh is the fourth entry overall, and carries that mesh's own
    // second normal and UV, not a continuation of the first mesh's numbering.
    assert_eq!(positions[3], [1.0, 1.0, 1.0]);
    assert_eq!(
        attributes[3],
        VertexAttributes {
            normal: [1.0, 0.0, 0.0],
            uv: [0.5, 0.5],
        }
    );
}

#[test]
fn an_empty_mesh_keeps_its_slot_as_a_zero_length_range() {
    // The scene indexes meshes by position, so a mesh that flattened to nothing must not shift
    // the ones after it.
    let meshes = [
        mesh(3, &[0, 1, 2], false),
        mesh(0, &[], false),
        mesh(3, &[0, 1, 2], false),
    ];
    let ranges = packed(&meshes).ranges;

    assert_eq!(ranges.len(), 3);
    assert_eq!(
        ranges[1],
        MeshRange {
            first_vertex: 3,
            vertex_count: 0,
            first_index: 3,
            index_count: 0,
            first_submesh: 1,
            submesh_count: 0,
        }
    );
    assert_eq!(ranges[2].first_vertex, 3);
    assert_eq!(ranges[2].first_index, 3);
}

#[test]
fn the_position_stream_is_tightly_packed() {
    // A build is told this stride explicitly; if the type ever grew padding, the build would
    // read every position from the wrong offset.
    assert_eq!(GeometryBuffers::POSITION_STRIDE, 12);
    assert_eq!(size_of::<VertexAttributes>(), 20);
}
