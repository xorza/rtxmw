//! The raytraced renderer: acceleration structures, passes and output.

pub mod shaders;

mod acceleration_structure;
mod auto_exposure;
mod composite;
mod denoiser;
mod gbuffer;
mod geometry_buffers;
mod light_buffer;
mod material_buffers;
mod scene_acceleration;
mod scene_renderer;
mod texture_array;
mod tonemap;
mod visibility_pass;

// Only what a crate above this one reaches for. Everything else a frame is made of — the passes,
// their buffers, the G-buffer between them — is assembled by `SceneRenderer` and never named
// outside; publishing it was letting `unreachable_pub` pass on types nothing could reach.
pub use crate::geometry_buffers::{GeometryBuffers, MeshRange, VertexAttributes};
pub use crate::scene_acceleration::SceneAcceleration;
pub use crate::scene_renderer::{SceneRenderer, TARGET_FORMAT};
pub use crate::tonemap::OUTPUT_FORMAT;
pub use crate::visibility_pass::FrameConstants;
