//! Device memory allocation, shared by every GPU resource.

use std::sync::{Arc, Mutex, MutexGuard};

use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

use crate::device::Device;
use crate::error::Result;
use crate::instance::Instance;
use crate::physical_device::PhysicalDevice;

/// A cloneable handle to the device's memory allocator.
///
/// Every buffer and image keeps one so it can return its allocation on drop, which is why this is a
/// handle rather than the allocator itself. The allocator frees its `VkDeviceMemory` when the last
/// handle goes away, so **every clone must be dropped before the [`Device`]** — in a struct owning
/// both, that means declaring the device last.
#[derive(Clone)]
pub struct Memory {
    /// A handle copy, not an owner: the real [`Device`] outlives this by construction.
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,
}

// The allocator holds `ash` function tables, which implement no `Debug`.
impl std::fmt::Debug for Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Memory")
            .field("device", &self.device.handle())
            .finish_non_exhaustive()
    }
}

impl Memory {
    /// Creates the allocator for `device`.
    pub fn new(instance: &Instance, physical: &PhysicalDevice, device: &Device) -> Result<Self> {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.raw().clone(),
            device: device.raw().clone(),
            physical_device: physical.raw(),
            debug_settings: Default::default(),
            // Acceleration structure builds address their inputs by device address, so the memory
            // has to be created with the matching flag — this is not an optimisation.
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        Ok(Self {
            device: device.raw().clone(),
            allocator: Arc::new(Mutex::new(allocator)),
        })
    }

    /// The device this memory belongs to.
    pub(crate) fn device(&self) -> &ash::Device {
        &self.device
    }

    /// Locks the allocator.
    ///
    /// Poisoning is recovered from rather than propagated: a panic elsewhere leaves the allocator's
    /// bookkeeping intact, and refusing every later allocation would turn one failed test into a
    /// cascade of unrelated ones.
    pub(crate) fn lock(&self) -> MutexGuard<'_, Allocator> {
        match self.allocator.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}
