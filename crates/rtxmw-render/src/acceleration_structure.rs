//! One built acceleration structure and the memory holding it.

use ash::vk;
use rtxmw_gpu::{Buffer, BufferMemory, Memory};

/// A `VkAccelerationStructureKHR` together with the buffer it is stored in.
///
/// The two are inseparable — the structure is a view onto a range of a buffer, and destroying
/// either without the other is a use-after-free — so they are one type rather than two.
pub struct AccelerationStructure {
    /// A handle copy, not an owner: the real `Device` outlives this by construction.
    loader: ash::khr::acceleration_structure::Device,
    raw: vk::AccelerationStructureKHR,
    /// The storage `raw` is a view onto; owned so it cannot outlive its structure or be outlived.
    buffer: Buffer,
    device_address: vk::DeviceAddress,
}

// The extension loader is a table of function pointers and implements no `Debug`.
impl std::fmt::Debug for AccelerationStructure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccelerationStructure")
            .field("raw", &self.raw)
            .field("size", &self.buffer.size())
            .finish_non_exhaustive()
    }
}

impl AccelerationStructure {
    /// Allocates storage of `size` bytes and creates an unbuilt structure over it.
    ///
    /// The size comes from `vkGetAccelerationStructureBuildSizesKHR` for a fresh build, or from a
    /// compaction query when copying a built one down to its final size.
    pub fn create(
        memory: &Memory,
        loader: &ash::khr::acceleration_structure::Device,
        name: &str,
        size: vk::DeviceSize,
        kind: vk::AccelerationStructureTypeKHR,
    ) -> rtxmw_gpu::Result<Self> {
        let buffer = Buffer::new(
            memory,
            name,
            size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            BufferMemory::Device,
        )?;

        // Offset zero: the spec wants a multiple of 256 from the buffer base, and one structure per
        // buffer keeps that trivially true.
        let info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(buffer.raw())
            .offset(0)
            .size(size)
            .ty(kind);
        // SAFETY: `buffer` is alive and large enough, and `info` is fully initialised.
        let raw = unsafe { loader.create_acceleration_structure(&info, None)? };

        let address_info =
            vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(raw);
        // SAFETY: `raw` was just created on this device.
        let device_address =
            unsafe { loader.get_acceleration_structure_device_address(&address_info) };

        Ok(Self {
            loader: loader.clone(),
            raw,
            buffer,
            device_address,
        })
    }

    /// The underlying handle.
    pub fn raw(&self) -> vk::AccelerationStructureKHR {
        self.raw
    }

    /// The address a top-level instance references this structure by.
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    /// Bytes of device memory the structure occupies.
    pub fn size(&self) -> vk::DeviceSize {
        self.buffer.size()
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        // SAFETY: builds are submitted through `Uploader`, which waits on a fence before returning,
        // so no build referencing this structure is still in flight.
        unsafe { self.loader.destroy_acceleration_structure(self.raw, None) };
    }
}
