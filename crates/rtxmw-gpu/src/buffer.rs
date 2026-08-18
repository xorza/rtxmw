//! A `VkBuffer` with its device memory attached.

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::error::Result;
use crate::memory::Memory;

/// Where a buffer's memory lives and which side writes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferMemory {
    /// Device-local and not host-visible. Reached only through a transfer.
    Device,
    /// Host-visible, for staging data on its way to the device.
    Upload,
    /// Host-visible and cached, for reading results back.
    Readback,
}

impl BufferMemory {
    fn location(self) -> MemoryLocation {
        match self {
            Self::Device => MemoryLocation::GpuOnly,
            Self::Upload => MemoryLocation::CpuToGpu,
            Self::Readback => MemoryLocation::GpuToCpu,
        }
    }
}

/// A buffer and the allocation backing it, freed together on drop.
#[derive(Debug)]
pub struct Buffer {
    memory: Memory,
    raw: vk::Buffer,
    /// `None` only after `Drop` has taken it.
    allocation: Option<Allocation>,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
}

impl Buffer {
    /// Creates a buffer of `size` bytes and binds memory to it.
    ///
    /// `name` appears in allocator diagnostics and in the panic message when the size is zero.
    pub fn new(
        memory: &Memory,
        name: &str,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        kind: BufferMemory,
    ) -> Result<Self> {
        assert!(size > 0, "`{name}`: Vulkan rejects a zero-sized buffer");

        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: `info` is fully initialised and the device is alive.
        let raw = unsafe { memory.device().create_buffer(&info, None)? };

        // SAFETY: `raw` was just created on this device.
        let requirements = unsafe { memory.device().get_buffer_memory_requirements(raw) };
        let allocation = memory.lock().allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: kind.location(),
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        });
        let allocation = match allocation {
            Ok(allocation) => allocation,
            Err(e) => {
                // SAFETY: nothing has referenced the buffer, and it is not returned to the caller.
                unsafe { memory.device().destroy_buffer(raw, None) };
                return Err(e.into());
            }
        };

        // SAFETY: the handle is only handed straight back to `bind_buffer_memory` for the buffer
        // the allocation was made for, and the allocation outlives that call.
        let device_memory = unsafe { allocation.memory() };
        let offset = allocation.offset();

        // Assembled before binding so that a bind failure drops it, releasing both the handle and
        // the allocation rather than leaking them on the way to reporting the error.
        let buffer = Self {
            memory: memory.clone(),
            raw,
            allocation: Some(allocation),
            size,
            usage,
        };

        // SAFETY: the allocation was made against this buffer's own requirements.
        unsafe {
            memory
                .device()
                .bind_buffer_memory(raw, device_memory, offset)?
        };

        Ok(buffer)
    }

    /// The underlying `VkBuffer`.
    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }

    /// Size in bytes, as requested.
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// The GPU address of byte zero.
    pub fn device_address(&self) -> vk::DeviceAddress {
        assert!(
            self.usage
                .contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
            "buffer was not created with SHADER_DEVICE_ADDRESS, so it has no device address"
        );
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.raw);
        // SAFETY: the buffer is alive and was created with the required usage flag.
        unsafe { self.memory.device().get_buffer_device_address(&info) }
    }

    /// The host mapping, for [`BufferMemory::Upload`] and [`BufferMemory::Readback`] buffers.
    ///
    /// Trimmed to the requested size. The allocation behind it is rounded up to the memory
    /// requirement's alignment, so the raw mapping is longer than the buffer and reading all of it
    /// would hand back padding as if it were data.
    pub fn mapped(&self) -> Option<&[u8]> {
        let size = self.size as usize;
        Some(&self.allocation.as_ref()?.mapped_slice()?[..size])
    }

    /// The writable host mapping, for [`BufferMemory::Upload`] buffers.
    ///
    /// Trimmed to the requested size, as [`Buffer::mapped`] is.
    pub fn mapped_mut(&mut self) -> Option<&mut [u8]> {
        let size = self.size as usize;
        Some(&mut self.allocation.as_mut()?.mapped_slice_mut()?[..size])
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: callers keep a buffer alive until the work reading it has completed — every
        // transfer in this crate waits on a fence before returning.
        unsafe { self.memory.device().destroy_buffer(self.raw, None) };
        if let Some(allocation) = self.allocation.take() {
            self.memory
                .lock()
                .free(allocation)
                .expect("failed to free buffer memory");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::test_gpu::TestGpu;

    #[test]
    fn a_device_buffer_reports_its_size_and_a_nonzero_address() {
        let gpu = TestGpu::shared();
        let buffer = Buffer::new(
            gpu.memory(),
            "address test",
            256,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            BufferMemory::Device,
        )
        .expect("could not create buffer");

        assert_eq!(buffer.size(), 256);
        assert_ne!(buffer.device_address(), 0);
        // Device-local memory is not mapped, so reaching for a host pointer must come back empty
        // rather than handing out one that cannot be written.
        assert!(buffer.mapped().is_none());
        gpu.assert_no_validation_errors();
    }

    #[test]
    fn a_host_visible_buffer_keeps_what_was_written_to_it() {
        let gpu = TestGpu::shared();
        let mut buffer = Buffer::new(
            gpu.memory(),
            "mapping test",
            8,
            vk::BufferUsageFlags::TRANSFER_SRC,
            BufferMemory::Upload,
        )
        .expect("could not create buffer");

        // Eight bytes, against an allocation the driver rounds up to its alignment — so this also
        // pins that the mapping is the buffer, not the padded allocation behind it.
        let written = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mapped = buffer.mapped_mut().expect("upload memory should be mapped");
        assert_eq!(mapped.len(), written.len(), "mapping includes padding");
        mapped.copy_from_slice(&written);
        assert_eq!(buffer.mapped().expect("still mapped"), &written);
        gpu.assert_no_validation_errors();
    }
}
