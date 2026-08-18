//! Copying one image onto another, rescaling as it goes.

use ash::vk;

/// Blits the whole of `source` onto the whole of `destination`, filtering if the sizes differ.
///
/// This is how a frame reaches the screen. A swapchain image cannot be written by a compute or ray
/// tracing shader — sRGB formats expose no storage capability — so rendering goes to an offscreen
/// HDR image and arrives here. The rescale is not incidental: the design renders at 1920x1080 and
/// presents at 3840x2160, and until an upscaler exists this is what bridges them.
///
/// `source` must be in `TRANSFER_SRC_OPTIMAL` and `destination` in `TRANSFER_DST_OPTIMAL`.
///
/// # Safety
/// `command_buffer` must be in the recording state, and both images must belong to `device`.
pub unsafe fn stretch(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    source: vk::Image,
    source_extent: vk::Extent2D,
    destination: vk::Image,
    destination_extent: vk::Extent2D,
) {
    let layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1);
    let region = vk::ImageBlit2::default()
        .src_subresource(layers)
        .src_offsets(corners(source_extent))
        .dst_subresource(layers)
        .dst_offsets(corners(destination_extent));
    let regions = [region];

    let info = vk::BlitImageInfo2::default()
        .src_image(source)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .dst_image(destination)
        .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .regions(&regions)
        // Linear rather than nearest: a same-size blit samples texel centres and is exact either
        // way, and a rescaling one wants the filtering.
        .filter(vk::Filter::LINEAR);

    // SAFETY: the caller guarantees the command buffer is recording and the images are in the
    // layouts named above.
    unsafe { device.cmd_blit_image2(command_buffer, &info) };
}

/// The two offsets bounding a whole 2D image, as a blit region wants them.
fn corners(extent: vk::Extent2D) -> [vk::Offset3D; 2] {
    [
        vk::Offset3D { x: 0, y: 0, z: 0 },
        vk::Offset3D {
            x: extent.width as i32,
            y: extent.height as i32,
            z: 1,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_region_spans_the_whole_image() {
        let [origin, far] = corners(vk::Extent2D {
            width: 1920,
            height: 1080,
        });
        assert_eq!((origin.x, origin.y, origin.z), (0, 0, 0));
        // Depth one, not zero: a blit region with zero extent in any axis copies nothing.
        assert_eq!((far.x, far.y, far.z), (1920, 1080, 1));
    }
}
