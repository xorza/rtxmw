//! The raytraced renderer: acceleration structures, passes and output.

pub mod shaders;

mod acceleration_structure;
mod geometry_buffers;
mod light_buffer;
mod material_buffers;
mod scene_acceleration;
mod texture_array;
mod visibility_pass;

pub use crate::acceleration_structure::AccelerationStructure;
pub use crate::geometry_buffers::{GeometryBuffers, MeshRange, SubmeshRange, VertexAttributes};
pub use crate::light_buffer::{GpuLight, LightBuffer};
pub use crate::material_buffers::{GpuGeometry, GpuMaterial, MaterialBuffers, NO_TEXTURE};
pub use crate::scene_acceleration::SceneAcceleration;
pub use crate::texture_array::TextureArray;
pub use crate::visibility_pass::{FrameConstants, VisibilityPass};
