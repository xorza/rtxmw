//! The raytraced renderer: acceleration structures, passes and output.

pub mod shaders;

mod geometry_buffers;

pub use crate::geometry_buffers::{GeometryBuffers, MeshRange, VertexAttributes};
