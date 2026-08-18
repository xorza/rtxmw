//! Everything needed to trace one cell, with no window involved.
//!
//! The split that matters is here: this owns the scene, the pass and the image traced into, and
//! knows nothing about surfaces, swapchains or presentation. That is what lets a test drive the
//! *same* code the engine runs rather than a replica of it — and what keeps the renderer's shape
//! independent of how its output reaches a screen.

use ash::vk;
use glam::Vec3;
use rtxmw_gpu::{Device, Image, Memory, RayTracingLimits, Uploader, image_barrier, memory_barrier};
use rtxmw_scene::StaticScene;
use rtxmw_texture::Texture;

use crate::geometry_buffers::GeometryBuffers;
use crate::light_buffer::LightBuffer;
use crate::material_buffers::MaterialBuffers;
use crate::scene_acceleration::SceneAcceleration;
use crate::texture_array::TextureArray;
use crate::visibility_pass::{FrameConstants, VisibilityPass};

/// Half-float rather than 8-bit: the trace writes linear radiance that tone mapping consumes at M8,
/// and an 8-bit intermediate would clip highlights before anything got the chance to.
pub const TARGET_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Ceiling on the bindless array, fixed because the descriptor set layout is.
///
/// A cell uses far fewer — Seyda Neen's office needs 118 — and the whole shipped library holds
/// 4,311 distinct textures, so this leaves room for a cell far larger than any interior.
const MAX_TEXTURES: u32 = 8192;

/// A loaded cell and the pass that traces it.
pub struct SceneRenderer {
    pass: VisibilityPass,
    target: Image,
    scene: Option<LoadedScene>,
}

impl std::fmt::Debug for SceneRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SceneRenderer")
            .field("target", &self.target.extent())
            .field("loaded", &self.scene.is_some())
            .finish_non_exhaustive()
    }
}

/// A cell's device-side data.
///
/// Every field is a lifetime anchor as much as a value: the descriptor set holds the top-level
/// structure's handle, the structures reference their geometry by device address, and nothing else
/// keeps any of it alive — so dropping this is what unloads a cell.
#[derive(Debug)]
#[allow(dead_code)]
struct LoadedScene {
    geometry: GeometryBuffers,
    tables: MaterialBuffers,
    textures: TextureArray,
    lights: LightBuffer,
    acceleration: SceneAcceleration,
    ambient: Vec3,
    light_count: u32,
}

impl SceneRenderer {
    /// Creates the pass and the image it traces into.
    /// The uploader is borrowed rather than owned, here and in every other method: it wraps one
    /// command pool on one queue, and a queue may not be submitted to from two places at once. A
    /// renderer that kept its own would make two renderers on a device a data race.
    pub fn new(device: &Device, memory: &Memory, extent: vk::Extent2D) -> rtxmw_gpu::Result<Self> {
        Ok(Self {
            pass: VisibilityPass::new(device, MAX_TEXTURES)?,
            target: Image::new(
                memory,
                "primary visibility target",
                extent,
                TARGET_FORMAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            )?,
            scene: None,
        })
    }

    /// Uploads `scene` and builds its acceleration structures, replacing whatever was loaded.
    ///
    /// The caller must have waited for device idle: this frees structures a queued frame could
    /// still be reading.
    pub fn load_scene(
        &mut self,
        device: &Device,
        uploader: &mut Uploader,
        limits: RayTracingLimits,
        scene: &StaticScene,
        textures: &[Option<Texture>],
    ) -> rtxmw_gpu::Result<()> {
        self.scene = None;

        let geometry = GeometryBuffers::upload(uploader, &scene.meshes)?;
        let materials = scene.materials.materials();
        let tables = MaterialBuffers::upload(uploader, &geometry, materials)?;
        let acceleration = SceneAcceleration::build(
            device,
            uploader,
            limits,
            &geometry,
            materials,
            &scene.instances,
        )?;
        let array = TextureArray::upload(device, uploader, textures)?;
        assert!(
            array.len() <= MAX_TEXTURES,
            "cell needs {} texture slots but the array is built for {MAX_TEXTURES}",
            array.len()
        );
        let lights = LightBuffer::upload(uploader, &scene.lights)?;

        self.pass.bind(
            acceleration.tlas(),
            &self.target,
            &geometry,
            &tables,
            &lights,
            &array,
        );
        let light_count = lights.count();
        self.scene = Some(LoadedScene {
            geometry,
            tables,
            textures: array,
            lights,
            acceleration,
            // A cell that declares no ambient gets none rather than a guess: it is meant to be lit
            // by what is placed in it.
            ambient: scene.ambient.map_or(Vec3::ZERO, |a| a.colour),
            light_count,
        });
        Ok(())
    }

    /// The image traced into, in `TRANSFER_SRC_OPTIMAL` once [`SceneRenderer::record`] has run.
    pub fn target(&self) -> &Image {
        &self.target
    }

    /// Whether a cell is loaded. With none, [`SceneRenderer::record`] does nothing.
    pub fn has_scene(&self) -> bool {
        self.scene.is_some()
    }

    /// The frame constants for a camera, filled in with the loaded cell's lighting.
    pub fn frame_constants(
        &self,
        view: glam::Mat4,
        projection: glam::Mat4,
        camera_position: Vec3,
    ) -> FrameConstants {
        let (ambient, lights) = self
            .scene
            .as_ref()
            .map_or((Vec3::ZERO, 0), |s| (s.ambient, s.light_count));
        FrameConstants::new(view, projection, camera_position, ambient, lights)
    }

    /// Records the trace into `command_buffer`, leaving the target ready to copy from.
    ///
    /// Does nothing without a loaded cell, so a caller need not branch on it.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state, and `device` must own this renderer.
    pub unsafe fn record(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        constants: &FrameConstants,
    ) {
        if !self.has_scene() {
            return;
        }
        // SAFETY: the caller guarantees the command buffer is recording, and every object below
        // belongs to this device and outlives the submission.
        unsafe {
            // The previous frame left it in TRANSFER_SRC; `UNDEFINED` discards those contents,
            // which is wanted since every pixel is about to be rewritten.
            image_barrier::transition(
                device,
                command_buffer,
                self.target.raw(),
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );
            self.pass
                .record(command_buffer, self.target.extent(), constants);
            memory_barrier::full(device, command_buffer);
            image_barrier::transition(
                device,
                command_buffer,
                self.target.raw(),
                vk::ImageLayout::GENERAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
        }
    }

    /// Records one trace and waits for it, for a caller with no frame loop of its own.
    pub fn render_once(
        &self,
        uploader: &mut Uploader,
        constants: &FrameConstants,
    ) -> rtxmw_gpu::Result<()> {
        uploader.submit_and_wait(|device, cmd| {
            // SAFETY: the command buffer is recording and every object is alive.
            unsafe { self.record(device, cmd, constants) };
        })
    }
}
