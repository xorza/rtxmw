//! The point lights a cell places, as the shader reads them.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use rtxmw_gpu::{Buffer, BufferMemory, Uploader};
use rtxmw_scene::Light;

/// One point light.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Pod, Zeroable)]
pub struct GpuLight {
    pub position: [f32; 3],
    /// Reach in world units. Nothing beyond this receives any of the light.
    pub radius: f32,
    /// Linear RGB, already scaled by the intensity the record does not carry.
    pub colour: [f32; 3],
    pub padding: f32,
}

/// Converts Morrowind's radius into radiant intensity.
///
/// The record gives a colour and a reach and no brightness at all — the original renderer's fixed
/// attenuation curve supplied that, so there is no value here to be faithful to. Scaling by radius
/// squared is what makes a large lamp and a small candle differ by their reach rather than by an
/// arbitrary per-light number, and the constant sets how bright a light is at half its radius.
///
/// Tuned by eye, and provisional: vanilla albedo already has light painted into it, so every one of
/// these is fighting illumination that is already in the texture. See `docs/design.md` §5.1 — the
/// de-lighting spike is what makes this number mean anything.
const INTENSITY: f32 = 0.25;

/// A cell's lights, uploaded once.
#[derive(Debug)]
pub struct LightBuffer {
    buffer: Buffer,
    count: u32,
}

impl LightBuffer {
    /// Uploads every light in `lights`.
    pub fn upload(uploader: &mut Uploader, lights: &[Light]) -> rtxmw_gpu::Result<Self> {
        let table: Vec<GpuLight> = lights.iter().map(|light| GpuLight::new(*light)).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&table);

        let buffer = Buffer::new(
            uploader.memory(),
            "scene lights",
            (bytes.len() as vk::DeviceSize).max(Buffer::MIN_SIZE),
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::TRANSFER_SRC,
            BufferMemory::Device,
        )?;
        uploader.upload(&buffer, bytes)?;

        Ok(Self {
            buffer,
            count: table.len() as u32,
        })
    }

    /// The lights, indexed `0..count`.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// How many lights the buffer holds.
    pub fn count(&self) -> u32 {
        self.count
    }
}

impl GpuLight {
    /// Flattens a scene light, folding its intensity into the colour.
    fn new(light: Light) -> Self {
        let scale = light.radius * light.radius * INTENSITY;
        Self {
            position: light.position.to_array(),
            radius: light.radius,
            colour: (light.colour * scale).to_array(),
            padding: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn the_light_matches_the_layout_the_shader_declares() {
        assert_eq!(size_of::<GpuLight>(), 32);
    }

    #[test]
    fn a_wider_light_is_brighter_in_proportion_to_its_reach() {
        let small = GpuLight::new(Light {
            position: Vec3::ZERO,
            colour: Vec3::ONE,
            radius: 64.0,
        });
        let large = GpuLight::new(Light {
            position: Vec3::ZERO,
            colour: Vec3::ONE,
            radius: 128.0,
        });

        // Twice the radius is four times the intensity, so the illumination at the *same* fraction
        // of each light's reach comes out equal — which is what makes radius the only control the
        // data needs to give.
        assert_eq!(large.colour[0] / small.colour[0], 4.0);
        assert_eq!(small.colour[0], 64.0 * 64.0 * INTENSITY);

        // Colour survives the scaling as a ratio.
        let warm = GpuLight::new(Light {
            position: Vec3::ZERO,
            colour: Vec3::new(1.0, 0.5, 0.25),
            radius: 64.0,
        });
        assert_eq!(warm.colour[1] / warm.colour[0], 0.5);
        assert_eq!(warm.radius, 64.0);
    }
}
