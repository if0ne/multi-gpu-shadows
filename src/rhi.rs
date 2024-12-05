use std::{
    any::TypeId,
    cell::RefCell,
    collections::{hash_map::Entry, VecDeque},
    ffi::CString,
    hash::Hash,
    num::NonZero,
    ops::Range,
    path::PathBuf,
    rc::Rc,
    sync::{atomic::AtomicU64, Arc},
};

use ahash::{HashMap, HashSet, HashSetExt};
use fixedbitset::FixedBitSet;
use oxidx::dx::{
    self, IAdapter3, IBlobExt, ICommandAllocator, ICommandQueue, IDebug, IDebug1, IDebugExt,
    IDescriptorHeap, IDevice, IDeviceChildExt, IFactory4, IFactory6, IFence, IGraphicsCommandList,
    IGraphicsCommandListExt, IResource, ISwapchain1, PSO_NONE, RES_NONE,
};
use parking_lot::Mutex;

use crate::utils::new_uuid;

pub const FRAMES_IN_FLIGHT: usize = 3;

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DeviceMask: u32 {
        const Gpu1 = 0x1;
        const Gpu2 = 0x2;
        const Gpu3 = 0x4;
        const Gpu4 = 0x8;
    }
}

#[derive(Debug)]
pub struct DeviceManager {
    pub factory: dx::Factory4,
    pub debug: Option<dx::Debug1>,

    pub gpus: VecDeque<dx::Adapter3>,
    pub gpu_warp: Option<dx::Adapter3>,
    pub devices: Vec<Arc<Device>>,
    pub device_masks: VecDeque<DeviceMask>,
}

impl DeviceManager {
    pub fn new(use_debug: bool) -> Self {
        let flags = if use_debug {
            dx::FactoryCreationFlags::Debug
        } else {
            dx::FactoryCreationFlags::empty()
        };

        let factory = dx::create_factory4(flags).expect("Failed to create DXGI factory");

        let debug = if use_debug {
            let debug: dx::Debug1 = dx::create_debug()
                .expect("Failed to create debug")
                .try_into()
                .expect("Failed to fetch debug1");

            debug.enable_debug_layer();
            debug.set_enable_gpu_based_validation(true);
            debug.set_callback(Box::new(|_, _, _, msg| {
                println!("[d3d12] {}", msg);
            }));

            Some(debug)
        } else {
            None
        };

        let gpu_warp = factory.enum_warp_adapters().ok();
        let mut gpus = vec![];

        if let Ok(factory) = TryInto::<dx::Factory7>::try_into(factory.clone()) {
            let mut i = 0;

            while let Ok(adapter) =
                factory.enum_adapters_by_gpu_preference(i, dx::GpuPreference::HighPerformance)
            {
                let Ok(desc) = adapter.get_desc1() else {
                    i += 1;
                    continue;
                };

                if desc.flags().contains(dx::AdapterFlags::Sofware) {
                    i += 1;
                    continue;
                }

                if dx::create_device(Some(&adapter), dx::FeatureLevel::Level11).is_ok() {
                    gpus.push(adapter);
                }

                i += 1;
            }
        } else {
            let mut i = 0;
            while let Ok(adapter) = factory.enum_adapters(i) {
                let Ok(desc) = adapter.get_desc1() else {
                    i += 1;
                    continue;
                };

                if desc.flags().contains(dx::AdapterFlags::Sofware) {
                    i += 1;
                    continue;
                }

                if dx::create_device(Some(&adapter), dx::FeatureLevel::Level11).is_ok() {
                    gpus.push(adapter);
                }

                i += 1;
            }

            gpus.sort_by(|l, r| {
                let descs = (
                    l.get_desc1().map(|d| d.vendor_id()),
                    r.get_desc1().map(|d| d.vendor_id()),
                );

                match descs {
                    (Ok(0x8086), Ok(0x8086)) => std::cmp::Ordering::Equal,
                    (Ok(0x8086), Ok(_)) => std::cmp::Ordering::Less,
                    (Ok(_), Ok(0x8086)) => std::cmp::Ordering::Greater,
                    (_, _) => std::cmp::Ordering::Equal,
                }
            });
        }

        Self {
            factory,
            debug,
            gpus: gpus.into_iter().collect(),
            gpu_warp,
            devices: vec![],
            device_masks: VecDeque::from_iter([
                DeviceMask::Gpu1,
                DeviceMask::Gpu2,
                DeviceMask::Gpu3,
                DeviceMask::Gpu4,
            ]),
        }
    }

    pub fn get_high_perf_device(&mut self) -> Option<Arc<Device>> {
        let adapter = self.gpus.pop_front()?;
        let id = self.device_masks.pop_front()?;
        let device = Arc::new(Device::new(adapter, id));

        self.devices.push(Arc::clone(&device));

        Some(device)
    }

    pub fn get_low_perf_device(&mut self) -> Option<Arc<Device>> {
        let adapter = self.gpus.pop_back()?;
        let id = self.device_masks.pop_back()?;
        let device = Arc::new(Device::new(adapter, id));

        self.devices.push(Arc::clone(&device));

        Some(device)
    }

    pub fn get_warp(&mut self) -> Arc<Device> {
        let adapter = self.gpu_warp.take().expect("Failed to fetch warp");
        let id = self
            .device_masks
            .pop_back()
            .expect("Failed to get device id");

        let device = Arc::new(Device::new(adapter, id));
        self.devices.push(Arc::clone(&device));

        device
    }

    pub fn get_devices<'a>(
        &'a self,
        device_mask: DeviceMask,
    ) -> impl Iterator<Item = &'a Arc<Device>> {
        self.devices
            .iter()
            .filter(move |d| device_mask.contains(d.id))
    }

    pub fn get_device(&self, device_id: DeviceMask) -> Option<&Arc<Device>> {
        self.devices.iter().find(|d| device_id.eq(&d.id))
    }
}

#[derive(Debug)]
pub struct Device {
    pub id: DeviceMask,
    pub adapter: dx::Adapter3,
    pub gpu: dx::Device,

    pub rtv_heap: DescriptorHeap,
    pub dsv_heap: DescriptorHeap,
    pub shader_heap: DescriptorHeap,
    pub sampler_heap: DescriptorHeap,

    pub gfx_queue: CommandQueue,
    pub compute_queue: CommandQueue,
    pub copy_queue: CommandQueue,

    pub is_cross_adapter_texture_supported: bool,
}

impl Device {
    pub(crate) fn new(adapter: dx::Adapter3, id: DeviceMask) -> Self {
        let device = dx::create_device(Some(&adapter), dx::FeatureLevel::Level11)
            .expect("Failed to create device");

        let rtv_heap = DescriptorHeap::new(&device, id, dx::DescriptorHeapType::Rtv, 128);
        let dsv_heap = DescriptorHeap::new(&device, id, dx::DescriptorHeapType::Dsv, 128);
        let shader_heap = DescriptorHeap::new(&device, id, dx::DescriptorHeapType::CbvSrvUav, 1024);
        let sampler_heap = DescriptorHeap::new(&device, id, dx::DescriptorHeapType::Sampler, 32);

        let gfx_queue = CommandQueue::new(&device, dx::CommandListType::Direct);
        let compute_queue = CommandQueue::new(&device, dx::CommandListType::Compute);
        let copy_queue = CommandQueue::new(&device, dx::CommandListType::Copy);

        let mut feature = dx::features::OptionsFeature::default();
        device
            .check_feature_support(&mut feature)
            .expect("Failed to fetch options feature");

        Self {
            id,
            adapter,
            gpu: device,
            rtv_heap,
            dsv_heap,
            shader_heap,
            sampler_heap,

            gfx_queue,
            compute_queue,
            copy_queue,

            is_cross_adapter_texture_supported: feature.cross_adapter_row_major_texture_supported(),
        }
    }

    pub fn create_resource(
        &self,
        desc: &dx::ResourceDesc,
        heap_props: &dx::HeapProperties,
        state: dx::ResourceStates,
        clear_value: Option<dx::ClearValue>,
        name: impl ToString,
    ) -> DeviceResource {
        let name = name.to_string();
        let uuid = new_uuid();

        let res = self
            .gpu
            .create_committed_resource(
                heap_props,
                dx::HeapFlags::empty(),
                desc,
                state,
                clear_value.as_ref(),
            )
            .unwrap_or_else(|_| panic!("Failed to create resource {}", name));

        let debug_name = CString::new(name.as_bytes()).expect("Failed to create resource name");
        res.set_debug_object_name(&debug_name).unwrap();

        DeviceResource {
            device_id: self.id,
            res,
            name,
            uuid,
        }
    }

    pub fn create_placed_resource(
        &self,
        desc: &dx::ResourceDesc,
        heap: &dx::Heap,
        state: dx::ResourceStates,
        clear_value: Option<dx::ClearValue>,
        name: impl AsRef<str>,
    ) -> DeviceResource {
        let name = name.as_ref().to_string();
        let uuid = new_uuid();

        let res = self
            .gpu
            .create_placed_resource(heap, 0, desc, state, clear_value.as_ref())
            .unwrap_or_else(|_| panic!("Failed to create resource {}", name));

        let debug_name = CString::new(name.as_bytes()).expect("Failed to create resource name");
        res.set_debug_object_name(&debug_name).unwrap();

        DeviceResource {
            device_id: self.id,
            res,
            name,
            uuid,
        }
    }
}

#[derive(Debug)]
pub struct DescriptorHeap {
    pub device_id: DeviceMask,
    pub heap: dx::DescriptorHeap,
    pub ty: dx::DescriptorHeapType,
    pub size: usize,
    pub inc_size: usize,
    pub shader_visible: bool,
    pub descriptors: Mutex<FixedBitSet>,
}

impl DescriptorHeap {
    pub fn new(
        device: &dx::Device,
        device_id: DeviceMask,
        ty: dx::DescriptorHeapType,
        size: usize,
    ) -> Self {
        let descriptors = Mutex::new(FixedBitSet::with_capacity(size));

        let (shader_visible, flags) =
            if ty == dx::DescriptorHeapType::CbvSrvUav || ty == dx::DescriptorHeapType::Sampler {
                (true, dx::DescriptorHeapFlags::ShaderVisible)
            } else {
                (false, dx::DescriptorHeapFlags::empty())
            };

        let inc_size = device.get_descriptor_handle_increment_size(ty);

        let heap = device
            .create_descriptor_heap(&dx::DescriptorHeapDesc::new(ty, size).with_flags(flags))
            .expect("Failed to create descriptor heap");

        Self {
            device_id,
            heap,
            ty,
            size,
            inc_size,
            shader_visible,
            descriptors,
        }
    }

    pub fn alloc(&self, device: &Arc<Device>) -> Descriptor {
        let mut descriptors = self.descriptors.lock();

        let index = descriptors.zeroes().next().expect("Out of memory");

        descriptors.set(index, true);

        let cpu = self
            .heap
            .get_cpu_descriptor_handle_for_heap_start()
            .advance(index, self.inc_size);
        let gpu = self
            .heap
            .get_gpu_descriptor_handle_for_heap_start()
            .advance(index, self.inc_size);

        Descriptor {
            device: Arc::clone(device),
            ty: self.ty,
            heap_index: index,
            cpu,
            gpu,
        }
    }
}

#[derive(Debug)]
pub struct Descriptor {
    pub device: Arc<Device>,
    pub ty: dx::DescriptorHeapType,
    pub heap_index: usize,
    pub cpu: dx::CpuDescriptorHandle,
    pub gpu: dx::GpuDescriptorHandle,
}

impl Drop for Descriptor {
    fn drop(&mut self) {
        match self.ty {
            dx::DescriptorHeapType::Rtv => self
                .device
                .rtv_heap
                .descriptors
                .lock()
                .set(self.heap_index, false),
            dx::DescriptorHeapType::Dsv => self
                .device
                .dsv_heap
                .descriptors
                .lock()
                .set(self.heap_index, false),
            dx::DescriptorHeapType::CbvSrvUav => self
                .device
                .shader_heap
                .descriptors
                .lock()
                .set(self.heap_index, false),
            dx::DescriptorHeapType::Sampler => self
                .device
                .sampler_heap
                .descriptors
                .lock()
                .set(self.heap_index, false),
        }
    }
}

#[derive(Debug)]
pub struct LocalFence {
    pub fence: dx::Fence,
    pub value: AtomicU64,
}

impl LocalFence {
    pub fn new(device: &dx::Device) -> Self {
        let fence = device
            .create_fence(0, dx::FenceFlags::empty())
            .expect("Failed to create fence");

        Self {
            fence,
            value: Default::default(),
        }
    }

    pub fn wait(&self, value: u64) {
        if self.get_completed_value() < value {
            let event = dx::Event::create(false, false).expect("Failed to create event");
            self.fence
                .set_event_on_completion(value, event)
                .expect("Failed to bind fence to event");
            if event.wait(10_000_000) == 0x00000102 {
                panic!("Device lost")
            }
        }
    }

    pub fn inc_value(&self) -> u64 {
        self.value
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1
    }

    pub fn get_completed_value(&self) -> u64 {
        self.fence.get_completed_value()
    }

    pub fn get_current_value(&self) -> u64 {
        self.value.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[derive(Debug)]
pub struct SharedFence {
    owner: Arc<Device>,
    fence: dx::Fence,
    value: Arc<AtomicU64>,
}

impl SharedFence {
    pub(super) fn new(owner: &Arc<Device>) -> Self {
        let fence = owner
            .gpu
            .create_fence(
                0,
                dx::FenceFlags::Shared | dx::FenceFlags::SharedCrossAdapter,
            )
            .unwrap();

        Self {
            owner: Arc::clone(owner),
            fence,
            value: Default::default(),
        }
    }

    pub fn get_completed_value(&self) -> u64 {
        self.fence.get_completed_value()
    }

    pub fn set_event_on_completion(&self, value: u64, event: dx::Event) {
        self.fence.set_event_on_completion(value, event).unwrap();
    }

    pub fn inc_value(&self) -> u64 {
        self.value
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1
    }

    pub fn get_current_value(&self) -> u64 {
        self.value.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn wait(&self, value: u64) {
        if self.get_completed_value() < value {
            let event = dx::Event::create(false, false).expect("Failed to create event");
            self.fence
                .set_event_on_completion(value, event)
                .expect("Failed to bind fence to event");
            if event.wait(10_000_000) == 0x00000102 {
                panic!("Device lost")
            }
        }
    }
}

impl SharedFence {
    pub fn connect(&self, device: &Arc<Device>) -> Self {
        let handle = self
            .owner
            .gpu
            .create_shared_handle(&self.fence, None)
            .expect("Failed to create shared handle");
        let fence = device
            .gpu
            .open_shared_handle(handle)
            .expect("Failed to open shared handle");
        handle.close().unwrap();

        Self {
            owner: Arc::clone(device),
            fence,
            value: Arc::clone(&self.value),
        }
    }
}

#[derive(Debug)]
pub(crate) struct CommandAllocatorEntry {
    pub(crate) raw: dx::CommandAllocator,
    pub(crate) sync_point: u64,
}

#[derive(Debug)]
pub struct CommandQueue {
    pub queue: Mutex<dx::CommandQueue>,
    pub ty: dx::CommandListType,

    pub fence: LocalFence,

    cmd_allocators: Mutex<VecDeque<CommandAllocatorEntry>>,
    cmd_lists: Mutex<Vec<dx::GraphicsCommandList>>,

    in_record: Mutex<Vec<CommandBuffer>>,
    pending: Mutex<Vec<CommandBuffer>>,

    pub frequency: f64,
}

impl CommandQueue {
    pub fn new(device: &dx::Device, ty: dx::CommandListType) -> Self {
        let queue = device
            .create_command_queue(&dx::CommandQueueDesc::new(ty))
            .expect("Failed to create command queue");

        let fence = LocalFence::new(device);

        let frequency = 1000.0
            / queue
                .get_timestamp_frequency()
                .expect("Failed to fetch timestamp frequency") as f64;

        let cmd_allocators = (0..FRAMES_IN_FLIGHT)
            .map(|_| CommandAllocatorEntry {
                raw: device
                    .create_command_allocator(ty)
                    .expect("Failed to create command allocator"),
                sync_point: 0,
            })
            .collect::<VecDeque<_>>();

        let cmd_list = device
            .create_command_list(0, ty, &cmd_allocators[0].raw, PSO_NONE)
            .expect("Failed to create command list");
        cmd_list.close().expect("Failed to close list");

        Self {
            queue: Mutex::new(queue),
            ty,
            fence,
            frequency,

            cmd_allocators: Mutex::new(cmd_allocators),
            cmd_lists: Mutex::new(vec![cmd_list]),
            in_record: Default::default(),
            pending: Default::default(),
        }
    }

    pub fn set_mark(&self, mark: impl AsRef<str>) {
        let mark = CString::new(mark.as_ref().as_bytes()).expect("Failed to create mark");
        self.queue.lock().begin_event(0u64, &mark);
    }

    pub fn end_mark(&self) {
        self.queue.lock().end_event();
    }

    pub fn wait_on_gpu(&self, value: u64) {
        self.queue
            .lock()
            .wait(&self.fence.fence, value)
            .expect("Failed to queue wait");
    }

    pub fn wait_other_queue(&self, queue: &CommandQueue) {
        self.queue
            .lock()
            .wait(&queue.fence.fence, queue.fence.get_current_value())
            .expect("Failed to queue wait");
    }

    pub fn wait_on_cpu(&self, value: u64) {
        self.fence.wait(value);
    }

    pub fn wait_idle(&self) {
        let value = self.signal_queue();
        self.wait_on_cpu(value);
    }

    pub fn stash_cmd_buffer(&self, cmd_buffer: CommandBuffer) {
        self.in_record.lock().push(cmd_buffer);
    }

    pub fn push_cmd_buffer(&self, cmd_buffer: CommandBuffer) {
        cmd_buffer.list.close().expect("Failed to close list");
        self.pending.lock().push(cmd_buffer);
    }

    pub fn execute(&self) -> u64 {
        let cmd_buffers = self.pending.lock().drain(..).collect::<Vec<_>>();
        let lists = cmd_buffers
            .iter()
            .map(|b| Some(b.list.clone()))
            .collect::<Vec<_>>();

        self.queue.lock().execute_command_lists(&lists);
        let fence_value = self.signal_queue();

        let allocators = cmd_buffers.into_iter().map(|mut buffer| {
            buffer.allocator.sync_point = fence_value;
            buffer.allocator
        });
        self.cmd_allocators.lock().extend(allocators);

        let lists = lists
            .into_iter()
            .map(|list| unsafe { list.unwrap_unchecked() });
        self.cmd_lists.lock().extend(lists);

        fence_value
    }

    pub fn get_command_buffer(&self, device: &Device) -> CommandBuffer {
        if let Some(buffer) = self.in_record.lock().pop() {
            return buffer;
        };

        let allocator = if let Some(allocator) =
            self.cmd_allocators.lock().pop_front().and_then(|a| {
                if self.is_complete(a.sync_point) {
                    Some(a)
                } else {
                    None
                }
            }) {
            allocator
                .raw
                .reset()
                .expect("Failed to reset command allocator");
            allocator
        } else {
            CommandAllocatorEntry {
                raw: device
                    .gpu
                    .create_command_allocator(self.ty)
                    .expect("Failed to create command allocator"),
                sync_point: 0,
            }
        };

        let list = if let Some(list) = self.cmd_lists.lock().pop() {
            list.reset(&allocator.raw, PSO_NONE)
                .expect("Failed to reset list");
            list
        } else {
            let list = device
                .gpu
                .create_command_list(0, self.ty, &allocator.raw, PSO_NONE)
                .expect("Failed to create command list");
            list.close().expect("Failed to close list");
            list
        };

        CommandBuffer {
            device_id: device.id,
            _ty: self.ty,
            list,
            allocator,
        }
    }

    pub fn signal(&self, fence: &LocalFence) -> u64 {
        let value = fence.inc_value();
        self.queue
            .lock()
            .signal(&fence.fence, value)
            .expect("Failed to signal");

        value
    }

    pub fn signal_shared(&self, fence: &SharedFence) -> u64 {
        let value = fence.inc_value();
        self.queue
            .lock()
            .signal(&fence.fence, value)
            .expect("Failed to signal");

        value
    }

    pub fn is_complete(&self, value: u64) -> bool {
        self.fence.get_completed_value() >= value
    }

    pub fn is_ready(&self) -> bool {
        self.fence.get_completed_value() >= self.fence.get_current_value()
    }

    fn signal_queue(&self) -> u64 {
        self.signal(&self.fence)
    }
}

#[derive(Debug)]
pub struct DeviceResource {
    pub device_id: DeviceMask,
    pub res: dx::Resource,
    pub name: String,
    pub uuid: u64,
}

#[derive(Debug)]
pub struct DeviceTexture {
    pub res: DeviceResource,
    pub uuid: u64,
    pub width: u32,
    pub height: u32,
    pub array: u32,
    pub levels: u32,
    pub format: dx::Format,
    pub state: RefCell<dx::ResourceStates>,
}

impl DeviceTexture {
    pub fn new(
        device: &Device,
        width: u32,
        height: u32,
        array: u32,
        format: dx::Format,
        levels: u32,
        flags: dx::ResourceFlags,
        state: dx::ResourceStates,
        clear_value: Option<dx::ClearValue>,
        name: impl AsRef<str>,
    ) -> Self {
        let name = name.as_ref().to_string();
        let uuid = new_uuid();
        let desc = dx::ResourceDesc::texture_2d(width, height)
            .with_alignment(dx::HeapAlignment::ResourcePlacement)
            .with_format(format)
            .with_mip_levels(levels)
            .with_array_size(array as u16)
            .with_layout(dx::TextureLayout::Unknown)
            .with_flags(flags);

        let res = device.create_resource(
            &desc,
            &dx::HeapProperties::default(),
            state,
            clear_value,
            name,
        );

        Self {
            res,
            uuid,
            width,
            height,
            array,
            levels,
            format,
            state: RefCell::new(state),
        }
    }

    pub fn get_size(&self, device: &Device, mip: Option<u32>) -> usize {
        let mip = mip.map(|m| m..(m + 1)).unwrap_or(0..self.levels);

        let desc = self.res.res.get_desc();
        device
            .gpu
            .get_copyable_footprints(&desc, mip, 0, None, None, None)
    }
}

#[derive(Debug)]
pub struct Texture {
    pub textures: Vec<DeviceTexture>,
}

impl Texture {
    pub fn new(
        width: u32,
        height: u32,
        array: u32,
        format: dx::Format,
        levels: u32,
        flags: dx::ResourceFlags,
        state: dx::ResourceStates,
        clear_value: Option<dx::ClearValue>,
        name: impl AsRef<str>,
        devices: &[&Arc<Device>],
    ) -> Self {
        Self {
            textures: devices
                .iter()
                .map(|d| {
                    DeviceTexture::new(
                        &d,
                        width,
                        height,
                        array,
                        format,
                        levels,
                        flags,
                        state,
                        clear_value,
                        &name,
                    )
                })
                .collect(),
        }
    }

    pub fn get_texture(&self, device_id: DeviceMask) -> Option<&'_ DeviceTexture> {
        self.textures
            .iter()
            .find(|b| b.res.device_id.eq(&device_id))
    }
}

#[derive(Debug)]
pub struct SharedTexture {
    pub owner: Arc<Device>,
    pub heap: dx::Heap,

    pub state: SharedTextureState,
    pub desc: dx::ResourceDesc,
}

impl SharedTexture {
    pub fn new(
        device: &Arc<Device>,
        width: u32,
        height: u32,
        format: dx::Format,
        flags: dx::ResourceFlags,
        local_state: dx::ResourceStates,
        shared_state: dx::ResourceStates,
        clear_value: Option<dx::ClearValue>,
        name: impl AsRef<str>,
    ) -> Self {
        let desc = dx::ResourceDesc::texture_2d(width, height)
            .with_alignment(dx::HeapAlignment::ResourcePlacement)
            .with_array_size(1)
            .with_format(format)
            .with_mip_levels(1)
            .with_array_size(1)
            .with_layout(dx::TextureLayout::Unknown)
            .with_flags(flags);

        let (flags, state) = if device.is_cross_adapter_texture_supported {
            (
                dx::ResourceFlags::AllowCrossAdapter | desc.flags(),
                local_state,
            )
        } else {
            (dx::ResourceFlags::AllowCrossAdapter, shared_state)
        };

        let cross_desc = desc
            .clone()
            .with_flags(flags)
            .with_layout(dx::TextureLayout::RowMajor);

        let size = device
            .gpu
            .get_copyable_footprints(&cross_desc, 0..1, 0, None, None, None) * 2;
        let heap = device
            .gpu
            .create_heap(
                &dx::HeapDesc::new(size, dx::HeapProperties::default())
                    .with_flags(dx::HeapFlags::SharedCrossAdapter | dx::HeapFlags::Shared),
            )
            .expect("Failed to create shared heap");

        let cross_res =
            device.create_placed_resource(&cross_desc, &heap, state, clear_value, "Cross Resource");

        let cross_res = DeviceTexture {
            res: cross_res,
            uuid: new_uuid(),
            width,
            height,
            array: 1,
            levels: 1,
            format,
            state: RefCell::new(state),
        };

        if device.is_cross_adapter_texture_supported {
            Self {
                owner: Arc::clone(device),
                heap,
                state: SharedTextureState::CrossAdapter { cross: cross_res },
                desc,
            }
        } else {
            let local_res = DeviceTexture::new(
                device,
                width,
                height,
                1,
                format,
                1,
                desc.flags(),
                local_state,
                clear_value,
                name,
            );

            Self {
                owner: Arc::clone(device),
                heap,
                state: SharedTextureState::Binded {
                    cross: cross_res,
                    local: local_res,
                },
                desc,
            }
        }
    }

    pub fn local_resource(&self) -> &DeviceTexture {
        match &self.state {
            SharedTextureState::CrossAdapter { cross } => cross,
            SharedTextureState::Binded { local, .. } => local,
        }
    }

    pub fn cross_resource(&self) -> &DeviceTexture {
        match &self.state {
            SharedTextureState::CrossAdapter { cross } => cross,
            SharedTextureState::Binded { cross, .. } => cross,
        }
    }

    pub fn get_desc(&self) -> &dx::ResourceDesc {
        &self.desc
    }

    pub fn owner(&self) -> &Device {
        &self.owner
    }

    pub fn connect_texture(
        &self,
        other: &Arc<Device>,
        local_state: dx::ResourceStates,
        share_state: dx::ResourceStates,
        clear_value: Option<dx::ClearValue>,
        name: impl AsRef<str>,
    ) -> Self {
        let handle = self
            .owner
            .gpu
            .create_shared_handle(&self.heap, None)
            .expect("Failed to create shared heap");
        let heap: dx::Heap = other
            .gpu
            .open_shared_handle(handle)
            .expect("Failed to open shared heap");
        handle.close().unwrap();

        let (flags, state) = if other.is_cross_adapter_texture_supported {
            (
                dx::ResourceFlags::AllowCrossAdapter | self.desc.flags(),
                local_state,
            )
        } else {
            (dx::ResourceFlags::AllowCrossAdapter, share_state)
        };

        let cross_res = other.create_placed_resource(
            &self.cross_resource().res.res.get_desc().with_flags(flags),
            &heap,
            state,
            clear_value,
            &name,
        );

        let cross_res = DeviceTexture {
            res: cross_res,
            uuid: new_uuid(),
            width: self.desc.width() as u32,
            height: self.desc.height() as u32,
            array: 1,
            levels: 1,
            format: self.desc.format(),
            state: RefCell::new(state),
        };

        if other.is_cross_adapter_texture_supported {
            Self {
                owner: Arc::clone(&other),
                heap,
                state: SharedTextureState::CrossAdapter { cross: cross_res },
                desc: self.desc.clone(),
            }
        } else {
            let local_res = DeviceTexture::new(
                &other,
                cross_res.width,
                cross_res.height,
                1,
                cross_res.format,
                1,
                self.desc.flags(),
                local_state,
                clear_value,
                name,
            );

            Self {
                owner: Arc::clone(other),
                heap,
                state: SharedTextureState::Binded {
                    cross: cross_res,
                    local: local_res,
                },
                desc: self.desc.clone(),
            }
        }
    }
}

#[derive(Debug)]
pub enum SharedTextureState {
    CrossAdapter {
        cross: DeviceTexture,
    },
    Binded {
        cross: DeviceTexture,
        local: DeviceTexture,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TextureViewType {
    RenderTarget,
    DepthTarget,
    ShaderResource,
    Storage,
}

#[derive(Debug)]
pub struct TextureView {
    pub views: Vec<DeviceTextureView>,
}

impl TextureView {
    pub fn new(
        parent: &Texture,
        format: dx::Format,
        ty: TextureViewType,
        mip: Option<u32>,
        devices: &[&Arc<Device>],
    ) -> Self {
        Self {
            views: devices
                .iter()
                .map(|d| {
                    let parent = parent
                        .get_texture(d.id)
                        .expect("Can not create texture view for this texture");

                    DeviceTextureView::new(d, parent, format, ty, mip)
                })
                .collect(),
        }
    }

    pub fn new_in_array(
        parent: &Texture,
        format: dx::Format,
        ty: TextureViewType,
        range: Range<u32>,
        devices: &[&Arc<Device>],
    ) -> Self {
        Self {
            views: devices
                .iter()
                .map(|d| {
                    let parent = parent
                        .get_texture(d.id)
                        .expect("Can not create texture view for this texture");

                    DeviceTextureView::new_in_array(d, parent, format, ty, range.clone())
                })
                .collect(),
        }
    }

    pub fn get_view(&self, device_id: DeviceMask) -> Option<&'_ DeviceTextureView> {
        self.views.iter().find(|b| b.device_id.eq(&device_id))
    }
}

#[derive(Debug)]
pub struct DeviceTextureView {
    pub device_id: DeviceMask,
    pub ty: TextureViewType,
    pub handle: Descriptor,
}

impl DeviceTextureView {
    pub fn new(
        device: &Arc<Device>,
        parent: &DeviceTexture,
        format: dx::Format,
        ty: TextureViewType,
        mip: Option<u32>,
    ) -> Self {
        let handle = match ty {
            TextureViewType::RenderTarget => {
                let handle = device.rtv_heap.alloc(device);
                device.gpu.create_render_target_view(
                    Some(&parent.res.res),
                    Some(&dx::RenderTargetViewDesc::texture_2d(format, 0, 0)),
                    handle.cpu,
                );
                handle
            }
            TextureViewType::DepthTarget => {
                let handle = device.dsv_heap.alloc(device);
                device.gpu.create_depth_stencil_view(
                    Some(&parent.res.res),
                    Some(&dx::DepthStencilViewDesc::texture_2d(format, 0)),
                    handle.cpu,
                );
                handle
            }
            TextureViewType::ShaderResource => {
                let (mip, detailed) = if let Some(mip) = mip {
                    (1, mip)
                } else {
                    (parent.levels, 0)
                };

                let handle = device.shader_heap.alloc(device);
                device.gpu.create_shader_resource_view(
                    Some(&parent.res.res),
                    Some(&dx::ShaderResourceViewDesc::texture_2d(
                        format, detailed, mip, 0.0, 0,
                    )),
                    handle.cpu,
                );
                handle
            }
            TextureViewType::Storage => {
                let mip = mip.unwrap_or(0);
                let handle = device.shader_heap.alloc(device);

                device.gpu.create_unordered_access_view(
                    Some(&parent.res.res),
                    dx::RES_NONE,
                    Some(&dx::UnorderedAccessViewDesc::texture_2d(format, mip, 0)),
                    handle.cpu,
                );
                handle
            }
        };

        Self {
            device_id: device.id,
            ty,
            handle,
        }
    }

    pub fn new_in_array(
        device: &Arc<Device>,
        parent: &DeviceTexture,
        format: dx::Format,
        ty: TextureViewType,
        range: Range<u32>,
    ) -> Self {
        let handle = match ty {
            TextureViewType::RenderTarget => {
                let handle = device.rtv_heap.alloc(device);
                device.gpu.create_render_target_view(
                    Some(&parent.res.res),
                    Some(&dx::RenderTargetViewDesc::texture_2d_array(
                        format, 0, 0, range,
                    )),
                    handle.cpu,
                );
                handle
            }
            TextureViewType::DepthTarget => {
                let handle = device.dsv_heap.alloc(device);
                device.gpu.create_depth_stencil_view(
                    Some(&parent.res.res),
                    Some(&dx::DepthStencilViewDesc::texture_2d_array(
                        format, 0, range,
                    )),
                    handle.cpu,
                );
                handle
            }
            TextureViewType::ShaderResource => {
                let handle = device.shader_heap.alloc(device);
                device.gpu.create_shader_resource_view(
                    Some(&parent.res.res),
                    Some(&dx::ShaderResourceViewDesc::texture_2d_array(
                        format, 0, 1, 0.0, 0, range,
                    )),
                    handle.cpu,
                );
                handle
            }
            TextureViewType::Storage => {
                let handle = device.shader_heap.alloc(device);

                device.gpu.create_unordered_access_view(
                    Some(&parent.res.res),
                    dx::RES_NONE,
                    Some(&dx::UnorderedAccessViewDesc::texture_2d_array(
                        format, 0, 0, range,
                    )),
                    handle.cpu,
                );
                handle
            }
        };

        Self {
            device_id: device.id,
            ty,
            handle,
        }
    }
}

pub struct Swapchain {
    pub swapchain: dx::Swapchain1,
    pub hwnd: NonZero<isize>,
    pub resources: Vec<(dx::Resource, DeviceTexture, DeviceTextureView, u64)>,
    pub width: u32,
    pub height: u32,
}

impl Swapchain {
    pub fn new(
        factory: &dx::Factory4,
        device: &Arc<Device>,
        width: u32,
        height: u32,
        hwnd: NonZero<isize>,
    ) -> Self {
        let desc = dx::SwapchainDesc1::new(width, height)
            .with_format(dx::Format::Rgba8Unorm)
            .with_usage(dx::FrameBufferUsage::RenderTargetOutput)
            .with_buffer_count(FRAMES_IN_FLIGHT)
            .with_scaling(dx::Scaling::None)
            .with_swap_effect(dx::SwapEffect::FlipSequential)
            .with_flags(dx::SwapchainFlags::AllowTearing);

        let swapchain = factory
            .create_swapchain_for_hwnd(
                &*device.gfx_queue.queue.lock(),
                hwnd,
                &desc,
                None,
                dx::OUTPUT_NONE,
            )
            .expect("Failed to create swapchain");

        let mut swapchain = Self {
            swapchain,
            hwnd,
            resources: vec![],
            width,
            height,
        };
        swapchain.resize(device, width, height, 0);

        swapchain
    }

    pub fn resize(&mut self, device: &Arc<Device>, width: u32, height: u32, sync_point: u64) {
        {
            std::mem::take(&mut self.resources);
        }

        self.swapchain
            .resize_buffers(
                FRAMES_IN_FLIGHT,
                width,
                height,
                dx::Format::Unknown,
                dx::SwapchainFlags::AllowTearing,
            )
            .expect("Failed to resize swapchain");

        for i in 0..FRAMES_IN_FLIGHT {
            let res: dx::Resource = self
                .swapchain
                .get_buffer(i)
                .expect("Failed to get swapchain buffer");

            let texture = DeviceTexture {
                res: DeviceResource {
                    device_id: device.id,
                    res: res.clone(),
                    name: "Swapchain Image".to_string(),
                    uuid: 0,
                },
                uuid: 0,
                width,
                height,
                array: 1,
                levels: 1,
                format: dx::Format::Rgba8Unorm,
                state: RefCell::new(dx::ResourceStates::Common),
            };

            let view = DeviceTextureView::new(
                device,
                &texture,
                texture.format,
                TextureViewType::RenderTarget,
                None,
            );

            self.resources.push((res, texture, view, sync_point));
        }
    }

    pub fn present(&self, vsync: bool) {
        let interval = if vsync { 1 } else { 0 };

        self.swapchain
            .present(interval, dx::PresentFlags::AllowTearing)
            .expect("Failed to present");
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum BindingEntry {
    Constants { size: usize, slot: u32 },
    Cbv { num: u32, slot: u32 },
    Uav { num: u32, slot: u32 },
    Srv { num: u32, slot: u32 },
    Sampler { num: u32, slot: u32 },
}

impl BindingEntry {
    pub(crate) fn as_dx(&self) -> dx::DescriptorRangeType {
        match self {
            BindingEntry::Constants { .. } => unreachable!(),
            BindingEntry::Cbv { .. } => dx::DescriptorRangeType::Cbv,
            BindingEntry::Uav { .. } => dx::DescriptorRangeType::Uav,
            BindingEntry::Srv { .. } => dx::DescriptorRangeType::Srv,
            BindingEntry::Sampler { .. } => dx::DescriptorRangeType::Sampler,
        }
    }

    pub(crate) fn num(&self) -> u32 {
        match *self {
            BindingEntry::Constants { .. } => unreachable!(),
            BindingEntry::Cbv { num, .. } => num,
            BindingEntry::Uav { num, .. } => num,
            BindingEntry::Srv { num, .. } => num,
            BindingEntry::Sampler { num, .. } => num,
        }
    }

    pub(crate) fn slot(&self) -> u32 {
        match *self {
            BindingEntry::Constants { .. } => unreachable!(),
            BindingEntry::Cbv { slot, .. } => slot,
            BindingEntry::Uav { slot, .. } => slot,
            BindingEntry::Srv { slot, .. } => slot,
            BindingEntry::Sampler { slot, .. } => slot,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct StaticSampler {
    pub slot: u32,
    pub filter: dx::Filter,
    pub address_mode: dx::AddressMode,
    pub comp_func: dx::ComparisonFunc,
    pub b_color: dx::BorderColor,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct RootSignatureDesc {
    pub entries: Vec<BindingEntry>,
    pub static_samplers: Vec<StaticSampler>,
    pub bindless: bool,
}

#[derive(Debug, Eq)]
pub struct RootSignature {
    pub root: dx::RootSignature,
    pub desc: RootSignatureDesc,
}

impl Hash for RootSignature {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.desc.hash(state);
    }
}

impl PartialEq for RootSignature {
    fn eq(&self, other: &Self) -> bool {
        self.desc == other.desc
    }
}

impl RootSignature {
    pub fn new(device: &Device, desc: RootSignatureDesc) -> Self {
        let mut parameters = vec![];
        let mut ranges = vec![];

        for entry in desc.entries.iter() {
            ranges.push([dx::DescriptorRange::new(entry.as_dx(), entry.num())
                .with_base_shader_register(entry.slot())
                .with_register_space(0)]);
        }

        for (i, entry) in desc.entries.iter().enumerate() {
            let parameter = if let BindingEntry::Constants { size, slot } = entry {
                dx::RootParameter::constant_32bit(*slot, 0, (*size / 4) as u32)
            } else {
                dx::RootParameter::descriptor_table(&ranges[i])
                    .with_visibility(dx::ShaderVisibility::All)
            };
            parameters.push(parameter);
        }

        let flags = if desc.bindless {
            dx::RootSignatureFlags::AllowInputAssemblerInputLayout
                | dx::RootSignatureFlags::CbvSrvUavHeapDirectlyIndexed
                | dx::RootSignatureFlags::SamplerHeapDirectlyIndexed
        } else {
            dx::RootSignatureFlags::AllowInputAssemblerInputLayout
        };

        let static_samplers = desc
            .static_samplers
            .iter()
            .map(|ss| {
                dx::StaticSamplerDesc::default()
                    .with_filter(ss.filter)
                    .with_address_u(ss.address_mode)
                    .with_address_v(ss.address_mode)
                    .with_address_w(ss.address_mode)
                    .with_comparison_func(ss.comp_func)
                    .with_shader_register(ss.slot)
                    .with_border_color(ss.b_color)
                    .with_visibility(dx::ShaderVisibility::Pixel)
            })
            .collect::<Vec<_>>();

        let de = dx::RootSignatureDesc::default()
            .with_samplers(&static_samplers)
            .with_parameters(&parameters)
            .with_flags(flags);

        let root = device
            .gpu
            .serialize_and_create_root_signature(&de, dx::RootSignatureVersion::V1_0, 0)
            .expect("Failed to create root signature");

        Self { root, desc }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum DepthOp {
    None,
    Less,
    LessEqual,
    Greater,
}

impl DepthOp {
    pub(crate) fn as_dx(&self) -> dx::ComparisonFunc {
        match self {
            DepthOp::None => dx::ComparisonFunc::Never,
            DepthOp::Less => dx::ComparisonFunc::Less,
            DepthOp::LessEqual => dx::ComparisonFunc::LessEqual,
            DepthOp::Greater => dx::ComparisonFunc::Greater,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShaderType {
    Vertex,
    Pixel,
}

#[derive(Clone, Debug, Eq)]
pub struct ShaderDesc {
    pub ty: ShaderType,
    pub path: PathBuf,
    pub entry_point: String,
    pub debug: bool,
    pub defines: Vec<(String, String)>,
}

impl Hash for ShaderDesc {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut set: HashSet<&(String, String)> = HashSet::new();
        for pair in &self.defines {
            set.insert(pair);
        }

        let mut sorted: Vec<_> = set.iter().collect();
        sorted.sort_by(|a, b| a.cmp(b));

        for pair in sorted {
            pair.hash(state);
        }
    }
}

impl PartialEq for ShaderDesc {
    fn eq(&self, other: &Self) -> bool {
        let set_self: HashSet<_> = self.defines.iter().collect();
        let set_other: HashSet<_> = other.defines.iter().collect();

        self.ty == other.ty
            && self.path == other.path
            && self.entry_point == other.entry_point
            && self.debug == other.debug
            && set_self == set_other
    }
}

#[derive(Debug, Eq)]
pub struct CompiledShader {
    pub raw: dx::Blob,
    pub desc: ShaderDesc,
}

impl Hash for CompiledShader {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.desc.hash(state);
    }
}

impl PartialEq for CompiledShader {
    fn eq(&self, other: &Self) -> bool {
        self.desc == other.desc
    }
}

impl CompiledShader {
    pub fn compile(desc: ShaderDesc) -> Self {
        let target = match desc.ty {
            ShaderType::Vertex => c"vs_5_0",
            ShaderType::Pixel => c"ps_5_0",
        };

        let flags = if desc.debug {
            dx::COMPILE_DEBUG | dx::COMPILE_SKIP_OPT
        } else {
            0
        };

        let defines = desc
            .defines
            .iter()
            .map(|(k, v)| {
                (
                    CString::new(k.as_bytes()).expect("CString::new failed"),
                    CString::new(v.as_bytes()).expect("CString::new failed"),
                )
            })
            .collect::<Vec<_>>();

        let defines = defines
            .iter()
            .map(|(k, v)| dx::ShaderMacro::new(k, v))
            .chain(std::iter::once(dx::ShaderMacro::default()))
            .collect::<Vec<_>>();

        let entry_point = CString::new(desc.entry_point.as_bytes()).expect("CString::new failed");

        let raw = dx::Blob::compile_from_file(&desc.path, &defines, &entry_point, target, flags, 0)
            .expect("Failed to compile a shader");

        Self { raw, desc }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct ShaderHandle {
    idx: usize,
    gen: usize,
}

#[derive(Debug, Default)]
pub struct ShaderCache {
    next_idx: usize,
    gen: usize,
    desc_to_handle: HashMap<ShaderDesc, ShaderHandle>,
    handle_to_shader: HashMap<ShaderHandle, CompiledShader>,
}

impl ShaderCache {
    pub fn get_shader_by_desc(&mut self, desc: ShaderDesc) -> ShaderHandle {
        match self.desc_to_handle.entry(desc) {
            Entry::Occupied(occupied_entry) => *occupied_entry.get(),
            Entry::Vacant(vacant_entry) => {
                let compiled_shader = CompiledShader::compile(vacant_entry.key().clone());

                let idx = self.next_idx;
                self.next_idx += 1;

                let handle = ShaderHandle { idx, gen: self.gen };

                self.handle_to_shader.insert(handle, compiled_shader);

                handle
            }
        }
    }

    pub fn get_shader(&self, handle: &ShaderHandle) -> &CompiledShader {
        self.handle_to_shader
            .get(&handle)
            .expect("Cache was cleared")
    }

    pub fn clear(&mut self) {
        self.gen += 1;
        self.next_idx = 0;
        self.desc_to_handle.clear();
        self.handle_to_shader.clear();
    }

    pub fn remove_by_desc(&mut self, desc: &ShaderDesc) {
        if let Some(handle) = self.desc_to_handle.remove(desc) {
            self.handle_to_shader.remove(&handle);
        }
    }

    pub fn remove_by_handle(&mut self, handle: &ShaderHandle) {
        if let Some(shader) = self.handle_to_shader.remove(handle) {
            self.desc_to_handle.remove(&shader.desc);
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct PipelineHandle {
    idx: usize,
    gen: usize,
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct DepthDesc {
    pub op: DepthOp,
    pub format: dx::Format,
    pub read_only: bool,
}

#[derive(Debug, Hash, PartialEq, Eq)]
pub struct InputElementDesc {
    pub semantic: dx::SemanticName,
    pub format: dx::Format,
    pub slot: u32,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum CullMode {
    None,
    Back,
    Front,
}

impl CullMode {
    pub(crate) fn as_dx(&self) -> dx::CullMode {
        match self {
            CullMode::None => dx::CullMode::None,
            CullMode::Back => dx::CullMode::Back,
            CullMode::Front => dx::CullMode::Front,
        }
    }
}

#[derive(Debug)]
pub struct RasterPipelineDesc {
    pub input_elements: Vec<InputElementDesc>,
    pub line: bool,
    pub wireframe: bool,
    pub depth_bias: i32,
    pub slope_bias: f32,
    pub depth: Option<DepthDesc>,
    pub signature: Option<Rc<RootSignature>>,
    pub formats: Vec<dx::Format>,
    pub vs: ShaderHandle,
    pub shaders: Vec<ShaderHandle>,
    pub cull_mode: CullMode,
}

impl Hash for RasterPipelineDesc {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.input_elements.hash(state);
        self.line.hash(state);
        self.wireframe.hash(state);
        self.depth.hash(state);
        self.signature.hash(state);
        self.formats.hash(state);
        self.vs.hash(state);
        self.shaders.hash(state);
    }
}

impl PartialEq for RasterPipelineDesc {
    fn eq(&self, other: &Self) -> bool {
        self.input_elements == other.input_elements
            && self.line == other.line
            && self.wireframe == other.wireframe
            && self.depth == other.depth
            && self.signature == other.signature
            && self.formats == other.formats
            && self.vs == other.vs
            && self.shaders == other.shaders
    }
}

impl Eq for RasterPipelineDesc {}

#[derive(Debug)]
pub struct RasterPipeline {
    pub pso: dx::PipelineState,
    pub root_signature: Option<Rc<RootSignature>>,
}

impl RasterPipeline {
    pub fn new(device: &Device, desc: &RasterPipelineDesc, cache: &ShaderCache) -> Self {
        let vert = cache.get_shader(&desc.vs);

        let input_element_desc = desc
            .input_elements
            .iter()
            .map(|el| dx::InputElementDesc::per_vertex(el.semantic, el.format, el.slot))
            .collect::<Vec<_>>();

        let de = dx::GraphicsPipelineDesc::new(&vert.raw)
            .with_input_layout(&input_element_desc)
            .with_blend_desc(
                dx::BlendDesc::default().with_render_targets(
                    desc.formats
                        .iter()
                        .map(|_| dx::RenderTargetBlendDesc::default()),
                ),
            )
            .with_render_targets(desc.formats.clone())
            .with_rasterizer_state(
                dx::RasterizerDesc::default()
                    .with_fill_mode(if desc.wireframe {
                        dx::FillMode::Wireframe
                    } else {
                        dx::FillMode::Solid
                    })
                    .with_cull_mode(desc.cull_mode.as_dx())
                    .with_depth_bias(desc.depth_bias)
                    .with_slope_scaled_depth_bias(desc.slope_bias),
            )
            .with_primitive_topology(if desc.line {
                dx::PipelinePrimitiveTopology::Line
            } else {
                dx::PipelinePrimitiveTopology::Triangle
            });

        let de = if let Some(depth) = &desc.depth {
            de.with_depth_stencil(
                dx::DepthStencilDesc::default()
                    .enable_depth(depth.op.as_dx())
                    .with_depth_write_mask(if depth.read_only {
                        dx::DepthWriteMask::empty()
                    } else {
                        dx::DepthWriteMask::All
                    }),
                depth.format,
            )
        } else {
            de
        };

        let (mut de, rs) = if let Some(rs) = &desc.signature {
            (de.with_root_signature(&rs.root), Some(Rc::clone(rs)))
        } else {
            (de, None)
        };

        for handle in &desc.shaders {
            let shader = cache.get_shader(handle);

            match shader.desc.ty {
                ShaderType::Pixel => de = de.with_ps(&shader.raw),
                ShaderType::Vertex => unreachable!(),
            }
        }

        let pso = device
            .gpu
            .create_graphics_pipeline(&de)
            .expect("Failed to create pipeline state");

        Self {
            pso,
            root_signature: rs,
        }
    }
}

#[derive(Debug)]
pub struct RasterPipelineCache {
    device: Arc<Device>,
    next_idx: usize,
    gen: usize,
    desc_to_handle: HashMap<RasterPipelineDesc, PipelineHandle>,
    handle_to_pso: HashMap<PipelineHandle, RasterPipeline>,
}

impl RasterPipelineCache {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            device: Arc::clone(&device),
            next_idx: Default::default(),
            gen: Default::default(),
            desc_to_handle: Default::default(),
            handle_to_pso: Default::default(),
        }
    }

    pub fn get_pso_by_desc(
        &mut self,
        desc: RasterPipelineDesc,
        shader_cache: &ShaderCache,
    ) -> PipelineHandle {
        match self.desc_to_handle.entry(desc) {
            Entry::Occupied(occupied_entry) => *occupied_entry.get(),
            Entry::Vacant(vacant_entry) => {
                let pso = RasterPipeline::new(&self.device, vacant_entry.key(), shader_cache);

                let idx = self.next_idx;
                self.next_idx += 1;

                let handle = PipelineHandle { idx, gen: self.gen };

                self.handle_to_pso.insert(handle, pso);

                handle
            }
        }
    }

    pub fn get_pso(&self, handle: &PipelineHandle) -> &RasterPipeline {
        self.handle_to_pso.get(&handle).expect("Cache was cleared")
    }

    pub fn clear(&mut self) {
        self.gen += 1;
        self.next_idx = 0;
        self.desc_to_handle.clear();
        self.handle_to_pso.clear();
    }

    pub fn remove_by_desc(&mut self, desc: &RasterPipelineDesc) {
        if let Some(handle) = self.desc_to_handle.remove(desc) {
            self.handle_to_pso.remove(&handle);
        }
    }

    pub fn remove_by_handle(&mut self, handle: &PipelineHandle) {
        if let Some(shader) = self.handle_to_pso.remove(handle) {
            // TODO: Add removing from handle_to_pso
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BufferType {
    Vertex,
    Index,
    Constant,
    Storage,
    Copy,
}

#[derive(Debug)]
pub struct Buffer {
    pub buffers: Vec<DeviceBuffer>,
}

impl Buffer {
    pub fn new(
        size: usize,
        stride: usize,
        ty: BufferType,
        readback: bool,
        name: impl AsRef<str>,
        inner_ty: TypeId,
        devices: &[&Arc<Device>],
    ) -> Self {
        Self {
            buffers: devices
                .iter()
                .map(|d| DeviceBuffer::new(&d, size, stride, ty, readback, name.as_ref(), inner_ty))
                .collect(),
        }
    }

    pub fn get_buffer(&self, device_id: DeviceMask) -> Option<&'_ DeviceBuffer> {
        self.buffers.iter().find(|b| b.res.device_id.eq(&device_id))
    }

    pub fn write<T: Clone + 'static>(&mut self, index: usize, data: &T) {
        self.buffers
            .iter_mut()
            .for_each(|buffer| buffer.write(index, data.clone()));
    }

    pub fn write_all<T: Clone + 'static>(&mut self, data: &[T]) {
        self.buffers
            .iter_mut()
            .for_each(|buffer| buffer.write_all(data));
    }

    pub fn constant<T: Clone + 'static>(
        count: usize,
        name: impl AsRef<str>,
        devices: &[&Arc<Device>],
    ) -> Self {
        Self {
            buffers: devices
                .iter()
                .map(|d| {
                    let mut b = DeviceBuffer::constant::<T>(d, count, &name);
                    b.build_constant(d, count, size_of::<T>());
                    b
                })
                .collect(),
        }
    }

    pub fn vertex<T: Clone + 'static>(
        count: usize,
        name: impl AsRef<str>,
        devices: &[&Arc<Device>],
    ) -> Self {
        Self {
            buffers: devices
                .iter()
                .map(|d| DeviceBuffer::vertex::<T>(d, count, &name))
                .collect(),
        }
    }

    pub fn index_u16(count: usize, name: impl AsRef<str>, devices: &[&Arc<Device>]) -> Self {
        Self {
            buffers: devices
                .iter()
                .map(|d| DeviceBuffer::index_u16(d, count, &name))
                .collect(),
        }
    }

    pub fn index_u32(count: usize, name: impl AsRef<str>, devices: &[&Arc<Device>]) -> Self {
        Self {
            buffers: devices
                .iter()
                .map(|d| DeviceBuffer::index_u32(d, count, &name))
                .collect(),
        }
    }

    pub fn copy<T: Clone + 'static>(
        count: usize,
        name: impl AsRef<str>,
        devices: &[&Arc<Device>],
    ) -> Self {
        Self {
            buffers: devices
                .iter()
                .map(|d| DeviceBuffer::copy::<T>(d, count, &name))
                .collect(),
        }
    }
}

#[derive(Debug)]
pub struct DeviceBuffer {
    pub res: DeviceResource,
    pub ty: BufferType,
    pub state: RefCell<dx::ResourceStates>,

    pub size: usize,
    pub stride: usize,

    pub srv: Option<Descriptor>,
    pub uav: Option<Descriptor>,
    pub cbv: Vec<Descriptor>,

    pub vbv: Option<dx::VertexBufferView>,
    pub ibv: Option<dx::IndexBufferView>,

    pub inner_ty: TypeId,
}

impl DeviceBuffer {
    pub fn new(
        device: &Device,
        size: usize,
        stride: usize,
        ty: BufferType,
        readback: bool,
        name: &str,
        inner_ty: TypeId,
    ) -> Self {
        let heap_props = match ty {
            BufferType::Constant | BufferType::Copy => dx::HeapProperties::upload(),
            _ => {
                if readback {
                    dx::HeapProperties::readback()
                } else {
                    dx::HeapProperties::default()
                }
            }
        };

        let state = match ty {
            BufferType::Copy | BufferType::Constant => dx::ResourceStates::GenericRead,
            _ => dx::ResourceStates::Common,
        };

        let flags = if ty == BufferType::Storage {
            dx::ResourceFlags::AllowUnorderedAccess
        } else {
            dx::ResourceFlags::empty()
        };

        let desc = dx::ResourceDesc::buffer(size)
            .with_layout(dx::TextureLayout::RowMajor)
            .with_flags(flags);

        let res = device.create_resource(&desc, &heap_props, state, None, name);

        let vbv = if ty == BufferType::Vertex {
            Some(dx::VertexBufferView::new(
                res.res.get_gpu_virtual_address(),
                stride,
                size,
            ))
        } else {
            None
        };

        let ibv = if ty == BufferType::Index {
            Some(dx::IndexBufferView::new(
                res.res.get_gpu_virtual_address(),
                size,
                dx::Format::R32Uint,
            ))
        } else {
            None
        };

        Self {
            res,
            ty,
            state: RefCell::new(state),
            size,
            stride,
            srv: None,
            uav: None,
            cbv: vec![],
            vbv,
            ibv,
            inner_ty,
        }
    }

    pub fn write<T: Clone + 'static>(&mut self, index: usize, data: T) {
        let size = size_of::<T>();
        debug_assert_eq!(TypeId::of::<T>(), self.inner_ty);
        debug_assert!(size * index < self.size);

        let mapped = self.map::<T>();
        mapped.pointer[index] = data;
    }

    pub fn write_all<T: Clone + 'static>(&mut self, data: &[T]) {
        debug_assert_eq!(TypeId::of::<T>(), self.inner_ty);

        let mapped = self.map::<T>();
        mapped.pointer.clone_from_slice(data);
    }

    pub fn constant<T: Clone + 'static>(
        device: &Arc<Device>,
        count: usize,
        name: impl AsRef<str>,
    ) -> Self {
        const { assert!(align_of::<T>() == 256) };

        let size = size_of::<T>() * count;
        let mut buffer = Self::new(
            &device,
            size,
            0,
            BufferType::Constant,
            false,
            name.as_ref(),
            TypeId::of::<T>(),
        );

        buffer.build_constant(device, count, size_of::<T>());

        buffer
    }

    pub fn vertex<T: Clone + 'static>(
        device: &Arc<Device>,
        count: usize,
        name: impl AsRef<str>,
    ) -> Self {
        let size = size_of::<T>() * count;

        Self::new(
            &device,
            size,
            size_of::<T>(),
            BufferType::Vertex,
            false,
            name.as_ref(),
            TypeId::of::<T>(),
        )
    }

    pub fn index_u16(device: &Arc<Device>, count: usize, name: impl AsRef<str>) -> Self {
        Self::new(
            &device,
            size_of::<u16>() * count,
            size_of::<u16>(),
            BufferType::Index,
            false,
            name.as_ref(),
            TypeId::of::<u16>(),
        )
    }

    pub fn index_u32(device: &Arc<Device>, count: usize, name: impl AsRef<str>) -> Self {
        Self::new(
            &device,
            size_of::<u32>() * count,
            size_of::<u32>(),
            BufferType::Index,
            false,
            name.as_ref(),
            TypeId::of::<u32>(),
        )
    }

    pub fn copy<T: Clone + 'static>(
        device: &Arc<Device>,
        count: usize,
        name: impl AsRef<str>,
    ) -> Self {
        let size = size_of::<T>() * count;

        Self::new(
            device,
            size,
            0,
            BufferType::Copy,
            false,
            name.as_ref(),
            TypeId::of::<T>(),
        )
    }

    pub fn build_constant(&mut self, device: &Arc<Device>, count: usize, object_size: usize) {
        if !self.cbv.is_empty() {
            return;
        }

        for i in 0..count {
            let d = device.shader_heap.alloc(device);
            let desc = dx::ConstantBufferViewDesc::new(
                self.res.res.get_gpu_virtual_address() + (i * object_size) as u64,
                object_size,
            );
            device.gpu.create_constant_buffer_view(Some(&desc), d.cpu);

            self.cbv.push(d);
        }
    }

    pub fn build_storage(&mut self, device: &Arc<Device>) {
        if self.uav.is_some() {
            return;
        }

        let d = device.shader_heap.alloc(device);
        let desc = dx::UnorderedAccessViewDesc::buffer(
            dx::Format::Unknown,
            0..self.size,
            1,
            0,
            dx::BufferUavFlags::empty(),
        );
        device
            .gpu
            .create_unordered_access_view(Some(&self.res.res), RES_NONE, Some(&desc), d.cpu);

        self.uav = Some(d);
    }

    pub fn build_shader_resource(&mut self, device: &Arc<Device>) {
        if self.srv.is_some() {
            return;
        }

        let d = device.shader_heap.alloc(device);
        let desc = dx::ShaderResourceViewDesc::buffer(
            dx::Format::Unknown,
            0..(self.size / self.stride),
            self.stride,
            dx::BufferSrvFlags::empty(),
        );
        device
            .gpu
            .create_shader_resource_view(Some(&self.res.res), Some(&desc), d.cpu);

        self.srv = Some(d);
    }

    pub fn map<T>(&mut self) -> BufferMap<'_, T> {
        let size = self.size / size_of::<T>();

        let pointer = self
            .res
            .res
            .map::<T>(0, None)
            .expect("Failed to map buffer");

        unsafe {
            let pointer = std::slice::from_raw_parts_mut(pointer.as_ptr(), size);

            BufferMap {
                buffer: self,
                pointer,
            }
        }
    }
}

pub struct BufferMap<'a, T> {
    buffer: &'a DeviceBuffer,
    pub pointer: &'a mut [T],
}

impl<'a, T> Drop for BufferMap<'a, T> {
    fn drop(&mut self) {
        self.buffer.res.res.unmap(0, None);
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SamplerAddress {
    Wrap,
    Mirror,
    Clamp,
    Border,
}

impl SamplerAddress {
    pub(crate) fn as_dx(&self) -> dx::AddressMode {
        match self {
            SamplerAddress::Wrap => dx::AddressMode::Wrap,
            SamplerAddress::Mirror => dx::AddressMode::Mirror,
            SamplerAddress::Clamp => dx::AddressMode::Clamp,
            SamplerAddress::Border => dx::AddressMode::Border,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SamplerFilter {
    Nearest,
    Linear,
    Anisotropic,
}

impl SamplerFilter {
    pub(crate) fn as_dx(&self) -> dx::Filter {
        match self {
            SamplerFilter::Nearest => dx::Filter::Point,
            SamplerFilter::Linear => dx::Filter::Linear,
            SamplerFilter::Anisotropic => dx::Filter::Anisotropic,
        }
    }
}

pub struct Sampler {
    pub handle: Descriptor,
}

impl Sampler {
    pub fn new(device: &Arc<Device>, address: SamplerAddress, filter: SamplerFilter) -> Self {
        let handle = device.sampler_heap.alloc(device);

        let desc = dx::SamplerDesc::new(filter.as_dx())
            .with_address_u(address.as_dx())
            .with_address_v(address.as_dx())
            .with_address_w(address.as_dx())
            .with_comparison_func(dx::ComparisonFunc::Never);

        device.gpu.create_sampler(&desc, handle.cpu);

        Self { handle }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GeomTopology {
    Triangles,
    Lines,
}

impl GeomTopology {
    pub(crate) fn as_dx(&self) -> dx::PrimitiveTopology {
        match self {
            GeomTopology::Triangles => dx::PrimitiveTopology::Triangle,
            GeomTopology::Lines => dx::PrimitiveTopology::Line,
        }
    }
}

#[derive(Debug)]
pub struct CommandBuffer {
    pub(crate) device_id: DeviceMask,
    pub(crate) _ty: dx::CommandListType,
    pub(crate) list: dx::GraphicsCommandList,
    pub(crate) allocator: CommandAllocatorEntry,
}

impl CommandBuffer {
    pub fn begin(&self, device: &Device) {
        self.list.set_descriptor_heaps(&[
            Some(device.shader_heap.heap.clone()),
            Some(device.sampler_heap.heap.clone()),
        ]);
    }

    pub fn set_mark(&self, mark: impl AsRef<str>) {
        let mark = CString::new(mark.as_ref().as_bytes()).expect("Failed to create mark");
        self.list.set_marker(0u64, &mark);
    }

    pub fn set_render_targets(
        &self,
        views: &[&DeviceTextureView],
        depth: Option<&DeviceTextureView>,
    ) {
        let rtvs = views.iter().map(|view| view.handle.cpu).collect::<Vec<_>>();

        let dsv = depth.map(|view| view.handle.cpu);

        self.list.om_set_render_targets(&rtvs, false, dsv);
    }

    pub fn set_buffer_barrier(&self, buffer: &DeviceBuffer, state: dx::ResourceStates) {
        let mut old_state = buffer.state.borrow_mut();

        if *old_state == state {
            return;
        }

        let barrier = if *old_state == dx::ResourceStates::UnorderedAccess
            && state == dx::ResourceStates::UnorderedAccess
        {
            dx::ResourceBarrier::uav(&buffer.res.res)
        } else {
            dx::ResourceBarrier::transition(&buffer.res.res, *old_state, state, None)
        };

        self.list.resource_barrier(&[barrier]);
        *old_state = state;
    }

    pub fn set_texture_barrier(
        &self,
        texture: &Texture,
        state: dx::ResourceStates,
        mip: Option<u32>,
    ) {
        let Some(texture) = texture.get_texture(self.device_id) else {
            return;
        };

        self.set_device_texture_barrier(texture, state, mip);
    }

    pub fn set_texture_barriers(&self, texture: &[(&Texture, dx::ResourceStates)]) {
        let textures = texture
            .iter()
            .filter_map(|(t, s)| {
                if let Some(t) = t.get_texture(self.device_id) {
                    Some((t, *s))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        self.set_device_texture_barriers(&textures);
    }

    pub fn set_device_texture_barrier(
        &self,
        texture: &DeviceTexture,
        state: dx::ResourceStates,
        mip: Option<u32>,
    ) {
        let mut old_state = texture.state.borrow_mut();

        if *old_state == state {
            return;
        }

        let barrier = if *old_state == dx::ResourceStates::UnorderedAccess
            && state == dx::ResourceStates::UnorderedAccess
        {
            dx::ResourceBarrier::uav(&texture.res.res)
        } else {
            dx::ResourceBarrier::transition(&texture.res.res, *old_state, state, mip)
        };

        self.list.resource_barrier(&[barrier]);
        *old_state = state;
    }

    pub fn set_device_texture_barriers(&self, texture: &[(&DeviceTexture, dx::ResourceStates)]) {
        let barriers = texture
            .iter()
            .filter_map(|(t, s)| {
                let mut old_state = t.state.borrow_mut();

                if *old_state == *s {
                    return None;
                }

                let barrier = if *old_state == dx::ResourceStates::UnorderedAccess
                    && *s == dx::ResourceStates::UnorderedAccess
                {
                    dx::ResourceBarrier::uav(&t.res.res)
                } else {
                    dx::ResourceBarrier::transition(&t.res.res, *old_state, *s, None)
                };
                *old_state = *s;

                Some(barrier)
            })
            .collect::<Vec<_>>();

        if barriers.len() > 0 {
            self.list.resource_barrier(&barriers);
        }
    }

    pub fn clear_render_target(&self, view: &DeviceTextureView, r: f32, g: f32, b: f32) {
        self.list
            .clear_render_target_view(view.handle.cpu, [r, g, b, 1.0], &[]);
    }

    pub fn clear_depth_target(&self, view: &DeviceTextureView) {
        self.list
            .clear_depth_stencil_view(view.handle.cpu, dx::ClearFlags::Depth, 1.0, 0, &[]);
    }

    pub fn set_graphics_pipeline(&self, pipeline: &RasterPipeline) {
        self.list
            .set_graphics_root_signature(pipeline.root_signature.as_ref().map(|rs| &rs.root));
        self.list.set_pipeline_state(&pipeline.pso);
    }

    pub fn set_vertex_buffers(&self, buffers: &[&Buffer]) {
        let buffer_views = buffers
            .iter()
            .filter_map(|b| b.get_buffer(self.device_id))
            .map(|b| b.vbv.expect("Expected vertex buffer"))
            .collect::<Vec<_>>();

        self.list.ia_set_vertex_buffers(0, &buffer_views);
    }

    pub fn set_index_buffer(&self, buffer: &Buffer) {
        if let Some(ib) = buffer.get_buffer(self.device_id) {
            self.list
                .ia_set_index_buffer(Some(&ib.ibv.expect("Expected index buffer")));
        }
    }

    pub fn set_graphics_srv(&self, view: &DeviceTextureView, index: usize) {
        self.list
            .set_graphics_root_descriptor_table(index as u32, view.handle.gpu);
    }

    pub fn set_graphics_cbv(&self, desc: &Descriptor, index: usize) {
        self.list
            .set_graphics_root_descriptor_table(index as u32, desc.gpu);
    }

    pub fn set_graphics_sampler(&self, sampler: &Sampler, index: usize) {
        self.list
            .set_graphics_root_descriptor_table(index as u32, sampler.handle.gpu);
    }

    pub fn set_graphics_push_constants<T: Copy>(&self, data: &[T], index: usize) {
        self.list
            .set_graphics_root_32bit_constants(index as u32, data, 0);
    }

    pub fn set_viewport(&self, width: u32, height: u32) {
        let viewport = dx::Viewport::from_size((width as f32, height as f32));
        let rect = dx::Rect::default().with_size((width as i32, height as i32));

        self.list.rs_set_viewports(&[viewport]);
        self.list.rs_set_scissor_rects(&[rect]);
    }

    pub fn set_topology(&self, topo: GeomTopology) {
        self.list.ia_set_primitive_topology(topo.as_dx());
    }

    pub fn draw(&self, count: u32) {
        self.list.draw_instanced(count, 1, 0, 0);
    }

    pub fn draw_indexed(&self, count: u32, start_index: u32, base_vertex: i32) {
        self.list
            .draw_indexed_instanced(count, 1, start_index, base_vertex, 0);
    }

    pub fn copy_texture_to_texture(&self, dst: &DeviceTexture, src: &DeviceTexture) {
        self.list.copy_resource(&dst.res.res, &src.res.res);
    }

    pub fn load_texture_from_memory(&self, dst: &Texture, upload_buffer: &Buffer, data: &[u8]) {
        let Some(upload_buffer) = upload_buffer.get_buffer(self.device_id) else {
            return;
        };

        let Some(dst) = dst.get_texture(self.device_id) else {
            return;
        };

        self.load_device_texture_from_memory(dst, upload_buffer, data);
    }

    pub fn load_device_texture_from_memory(
        &self,
        dst: &DeviceTexture,
        upload_buffer: &DeviceBuffer,
        data: &[u8],
    ) {
        debug_assert!(
            self.list.update_subresources_fixed::<1, _, _>(
                &dst.res.res,
                &upload_buffer.res.res,
                0,
                0..1,
                &[dx::SubresourceData::new(data).with_row_pitch(4 * dst.width as usize)],
            ) > 0
        );
    }

    pub fn copy_buffer_to_buffer(&self, dst: &Buffer, src: &Buffer) {
        if let (Some(dst), Some(src)) = (
            dst.get_buffer(self.device_id),
            src.get_buffer(self.device_id),
        ) {
            self.list.copy_resource(&dst.res.res, &src.res.res);
        }
    }
}
