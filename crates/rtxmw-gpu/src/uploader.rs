//! Moving bytes between host memory and device-local buffers.

use ash::vk;

use crate::buffer::{Buffer, BufferMemory};
use crate::commands::Commands;
use crate::device::Device;
use crate::error::Result;
use crate::image::Image;
use crate::image_barrier;
use crate::memory::Memory;
use crate::memory_barrier;

/// Staged transfers into device-local buffers, plus the one-shot queue they ride on.
///
/// The staging buffer is kept and reused rather than allocated per upload: a cell's geometry
/// arrives as a handful of large writes, and reallocating between them would fragment the heap for
/// no gain.
#[derive(Debug)]
pub struct Uploader {
    memory: Memory,
    commands: Commands,
    /// Grown on demand to the largest single upload seen so far.
    staging: Option<Buffer>,
}

impl Uploader {
    /// Creates an uploader submitting on `device`'s graphics queue.
    pub fn new(device: &Device, memory: &Memory, queue_family: u32) -> Result<Self> {
        Ok(Self {
            memory: memory.clone(),
            commands: Commands::new(device, queue_family)?,
            staging: None,
        })
    }

    /// The device this uploader submits on, for a caller recording its own commands.
    pub fn device(&self) -> &ash::Device {
        self.commands.device()
    }

    /// The allocator these transfers stage through, for callers creating their own buffers.
    pub fn memory(&self) -> &Memory {
        &self.memory
    }

    /// Copies `bytes` to the start of `destination` and waits for the copy to complete.
    pub fn upload(&mut self, destination: &Buffer, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let size = bytes.len() as vk::DeviceSize;
        assert!(
            size <= destination.size(),
            "upload of {size} bytes into a {} byte buffer",
            destination.size()
        );

        self.grow_staging(size)?;
        let staging = self
            .staging
            .as_mut()
            .expect("`grow_staging` leaves a staging buffer in place");
        staging
            .mapped_mut()
            .expect("staging memory is host-visible by construction")[..bytes.len()]
            .copy_from_slice(bytes);

        let from = staging.raw();
        let to = destination.raw();
        self.commands.submit_and_wait(|device, cmd| {
            let region = vk::BufferCopy::default().size(size);
            // SAFETY: the command buffer is recording and both buffers belong to this device.
            unsafe {
                device.cmd_copy_buffer(cmd, from, to, &[region]);
                memory_barrier::full(device, cmd);
            }
        })
    }

    /// Stages `bytes` into `image` and leaves it ready to sample.
    ///
    /// `regions` say which part of the staging buffer becomes which mip level; the caller builds
    /// them because only it knows the format's block layout. The image is transitioned from
    /// `UNDEFINED`, so whatever it held is discarded — an upload writes every level it names.
    pub fn upload_image(
        &mut self,
        image: &Image,
        bytes: &[u8],
        regions: &[vk::BufferImageCopy],
    ) -> Result<()> {
        assert!(!bytes.is_empty(), "an image upload needs pixels");
        let size = bytes.len() as vk::DeviceSize;

        self.grow_staging(size)?;
        let staging = self
            .staging
            .as_mut()
            .expect("`grow_staging` leaves a staging buffer in place");
        staging
            .mapped_mut()
            .expect("staging memory is host-visible by construction")[..bytes.len()]
            .copy_from_slice(bytes);

        let from = staging.raw();
        let to = image.raw();
        let range = image.full_range();
        self.commands.submit_and_wait(|device, cmd| {
            // SAFETY: the command buffer is recording, the image belongs to this device, and the
            // regions were built against the same level table the image was created with.
            unsafe {
                image_barrier::transition_range(
                    device,
                    cmd,
                    to,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    range,
                );
                device.cmd_copy_buffer_to_image(
                    cmd,
                    from,
                    to,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions,
                );
                image_barrier::transition_range(
                    device,
                    cmd,
                    to,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    range,
                );
            }
        })
    }

    /// Records `record` into a one-shot command buffer and blocks until it completes.
    pub fn submit_and_wait(
        &mut self,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) -> Result<()> {
        self.commands.submit_and_wait(record)
    }

    /// Replaces the staging buffer when the current one cannot hold `size` bytes.
    fn grow_staging(&mut self, size: vk::DeviceSize) -> Result<()> {
        if self.staging.as_ref().is_some_and(|s| s.size() >= size) {
            return Ok(());
        }
        // Dropped before the replacement is allocated, so the two never coexist.
        self.staging = None;
        self.staging = Some(Buffer::new(
            &self.memory,
            "upload staging",
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            BufferMemory::Upload,
        )?);
        Ok(())
    }
}

#[cfg(any(test, feature = "internals"))]
pub(crate) mod internals {
    use ash::vk;

    use crate::buffer::{Buffer, BufferMemory};
    use crate::error::Result;

    impl super::Uploader {
        /// Copies the whole of `source` into `out`, replacing its contents.
        ///
        /// Allocates a readback buffer per call, which is why this is not on the production path:
        /// it exists so a test can assert on what actually reached the device.
        pub fn download(&mut self, source: &Buffer, out: &mut Vec<u8>) -> Result<()> {
            let size = source.size();
            let readback = Buffer::new(
                &self.memory,
                "download readback",
                size,
                vk::BufferUsageFlags::TRANSFER_DST,
                BufferMemory::Readback,
            )?;

            let from = source.raw();
            let to = readback.raw();
            self.commands.submit_and_wait(|device, cmd| {
                let region = vk::BufferCopy::default().size(size);
                // SAFETY: the command buffer is recording and both buffers belong to this device.
                unsafe { device.cmd_copy_buffer(cmd, from, to, &[region]) };
            })?;

            out.clear();
            out.extend_from_slice(
                readback
                    .mapped()
                    .expect("readback memory is host-visible by construction"),
            );
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::test_gpu::TestGpu;

    fn device_buffer(gpu: &'static TestGpu, size: vk::DeviceSize) -> Buffer {
        Buffer::new(
            gpu.memory(),
            "upload test",
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
            BufferMemory::Device,
        )
        .expect("could not create buffer")
    }

    #[test]
    fn uploaded_bytes_come_back_unchanged() {
        let gpu = TestGpu::shared();
        let buffer = device_buffer(gpu, 64);
        let written: Vec<u8> = (0..64u8).collect();

        let mut uploader = gpu.uploader();
        uploader
            .upload(&buffer, &written)
            .expect("upload should succeed");

        let mut read = Vec::new();
        uploader
            .download(&buffer, &mut read)
            .expect("download should succeed");
        assert_eq!(read, written);

        drop(uploader);
        gpu.assert_no_validation_errors();
    }

    #[test]
    fn the_staging_buffer_survives_growing_and_shrinking_uploads() {
        let gpu = TestGpu::shared();
        let mut uploader = gpu.uploader();

        // Small, then large enough to force a new staging buffer, then small again. The last leg
        // is the one that matters: it reuses the grown buffer, so a length taken from the staging
        // buffer rather than the payload would copy stale bytes over the tail.
        for size in [16usize, 4096, 16] {
            let buffer = device_buffer(gpu, size as vk::DeviceSize);
            let written: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
            uploader
                .upload(&buffer, &written)
                .expect("upload should succeed");

            let mut read = Vec::new();
            uploader
                .download(&buffer, &mut read)
                .expect("download should succeed");
            assert_eq!(read, written, "{size} byte upload came back wrong");
        }

        drop(uploader);
        gpu.assert_no_validation_errors();
    }

    #[test]
    fn uploading_nothing_leaves_the_destination_alone() {
        let gpu = TestGpu::shared();
        let buffer = device_buffer(gpu, 16);
        let mut uploader = gpu.uploader();

        uploader
            .upload(&buffer, &[7u8; 16])
            .expect("upload should succeed");
        uploader
            .upload(&buffer, &[])
            .expect("empty upload is a no-op");

        let mut read = Vec::new();
        uploader
            .download(&buffer, &mut read)
            .expect("download should succeed");
        assert_eq!(read, [7u8; 16]);

        drop(uploader);
        gpu.assert_no_validation_errors();
    }
}
