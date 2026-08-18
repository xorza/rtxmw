//! Gets geometry onto the device and proves the bytes that arrived are the ones that were sent.
//!
//! The packing itself is unit-tested; what needs a GPU is the claim that a staged copy lands
//! intact, because a wrong offset or a short copy produces geometry that is subtly deformed rather
//! than obviously broken.

use glam::{Vec2, Vec3};
use rtxmw_gpu::TestGpu;
use rtxmw_render::{GeometryBuffers, MeshRange, VertexAttributes};
use rtxmw_scene::{LoadedCell, Mesh, Submesh};
use rtxmw_vfs::DATA_DIR_VAR;

const CELL: &str = "Seyda Neen, Census and Excise Office";

/// A tetrahedron-ish mesh whose every value is distinguishable from every other.
fn distinct_mesh(offset: f32, vertices: u32, indices: &[u32]) -> Mesh {
    Mesh {
        positions: (0..vertices)
            .map(|i| Vec3::new(offset + i as f32, offset - i as f32, i as f32 * 2.0))
            .collect(),
        normals: (0..vertices)
            .map(|i| Vec3::new(0.0, 1.0, i as f32))
            .collect(),
        uvs: (0..vertices)
            .map(|i| Vec2::new(i as f32 * 0.25, offset))
            .collect(),
        indices: indices.to_vec(),
        submeshes: vec![Submesh {
            first_index: 0,
            index_count: indices.len() as u32,
            material: 0,
        }],
    }
}

#[test]
fn every_uploaded_stream_reads_back_byte_for_byte() {
    let gpu = TestGpu::shared();
    let meshes = [
        distinct_mesh(0.0, 3, &[0, 1, 2]),
        distinct_mesh(100.0, 4, &[0, 1, 2, 0, 2, 3]),
    ];

    let mut uploader = gpu.uploader();
    let mut buffers = GeometryBuffers::new(gpu.memory()).expect("arena failed");
    buffers
        .append(&mut uploader, &meshes.iter().collect::<Vec<_>>(), &[0])
        .expect("upload failed");

    assert_eq!(buffers.vertex_count(), 7);
    assert_eq!(buffers.triangle_count(), 3);
    assert_eq!(
        buffers.ranges(),
        [
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

    let mut bytes = Vec::new();

    uploader
        .download(buffers.positions(), &mut bytes)
        .expect("position readback failed");
    let positions: &[[f32; 3]] = bytemuck::cast_slice(&bytes);
    let expected: Vec<[f32; 3]> = meshes
        .iter()
        .flat_map(|m| m.positions.iter().map(|p| p.to_array()))
        .collect();
    assert_eq!(positions, expected.as_slice());
    // The stride is what a build is told; a padded position stream would still read back the right
    // count while placing every triangle wrong.
    assert_eq!(bytes.len(), 7 * 12);

    uploader
        .download(buffers.attributes(), &mut bytes)
        .expect("attribute readback failed");
    let attributes: &[VertexAttributes] = bytemuck::cast_slice(&bytes);
    let expected: Vec<VertexAttributes> = meshes
        .iter()
        .flat_map(|m| {
            m.normals
                .iter()
                .zip(&m.uvs)
                .map(|(n, uv)| VertexAttributes {
                    normal: n.to_array(),
                    uv: uv.to_array(),
                })
        })
        .collect();
    assert_eq!(attributes, expected.as_slice());

    uploader
        .download(buffers.indices(), &mut bytes)
        .expect("index readback failed");
    let indices: &[u32] = bytemuck::cast_slice(&bytes);
    assert_eq!(indices, [0, 1, 2, 0, 1, 2, 0, 2, 3]);

    drop(uploader);
    gpu.assert_no_validation_errors();
}

#[test]
fn a_scene_with_no_geometry_still_produces_valid_buffers() {
    let gpu = TestGpu::shared();
    let uploader = gpu.uploader();

    // Vulkan rejects a zero-sized buffer, so an empty cell has to come out with real handles rather
    // than an error the caller has to special-case.
    let buffers = GeometryBuffers::new(gpu.memory()).expect("arena failed");
    assert_eq!(buffers.vertex_count(), 0);
    assert_eq!(buffers.triangle_count(), 0);
    assert!(buffers.ranges().is_empty());
    assert_ne!(buffers.positions().device_address(), 0);

    drop(uploader);
    gpu.assert_no_validation_errors();
}

#[test]
fn a_real_interior_uploads_with_every_triangle_accounted_for() {
    let Some(cell) = LoadedCell::load_interior(CELL).expect("cell should load") else {
        eprintln!("skipping: {DATA_DIR_VAR} is not configured (set it, or add it to .env)");
        return;
    };
    let scene = cell.scene;

    let gpu = TestGpu::shared();
    let mut uploader = gpu.uploader();
    let mut buffers = GeometryBuffers::new(gpu.memory()).expect("arena failed");
    let remap: Vec<u32> = (0..scene.materials.materials().len() as u32).collect();
    buffers
        .append(
            &mut uploader,
            &scene.meshes.iter().collect::<Vec<_>>(),
            &remap,
        )
        .expect("upload failed");

    // Nothing may be dropped or duplicated on the way to the device: the buffers hold exactly the
    // distinct meshes, which is what the acceleration structures will be built from.
    assert_eq!(buffers.ranges().len(), scene.meshes.len());
    assert_eq!(buffers.triangle_count(), scene.unique_triangle_count());
    assert_eq!(
        buffers.vertex_count(),
        scene
            .meshes
            .iter()
            .map(|m| m.positions.len())
            .sum::<usize>()
    );

    // Ranges must tile the buffers without gaps or overlap, or a build would read another mesh's
    // vertices.
    let mut vertex_cursor = 0u32;
    let mut index_cursor = 0u32;
    for (id, range) in buffers.ranges().iter().enumerate() {
        assert_eq!(range.first_vertex, vertex_cursor, "mesh {id} vertex offset");
        assert_eq!(range.first_index, index_cursor, "mesh {id} index offset");
        assert_eq!(range.index_count % 3, 0, "mesh {id} has a partial triangle");
        vertex_cursor += range.vertex_count;
        index_cursor += range.index_count;
    }

    println!(
        "{CELL}: {} meshes, {} vertices, {} triangles in {:.2} MiB of device memory",
        buffers.ranges().len(),
        buffers.vertex_count(),
        buffers.triangle_count(),
        buffers.byte_size() as f64 / (1024.0 * 1024.0),
    );

    drop(uploader);
    gpu.assert_no_validation_errors();
}
