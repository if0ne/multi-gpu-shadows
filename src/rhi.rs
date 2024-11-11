use std::{cell::RefCell, collections::HashMap, num::NonZero, rc::Rc};

use oxidx::dx::{
    self, ICommandQueue, IDebug, IDebug1, IDebugExt, IDescriptorHeap, IDevice, IFactory4,
    IFactory6, IFence, IResource, ISwapchain1,
};

use crate::utils::new_uuid;

pub const FRAMES_IN_FLIGHT: usize = 3;

pub struct Device {
    pub factory: dx::Factory4,
    pub adapter: dx::Adapter3,
    pub gpu: dx::Device,
    pub debug: Option<dx::Debug1>,
    pub rtv_heap: Rc<DescriptorHeap>,
    pub dsv_heap: Rc<DescriptorHeap>,
    pub shader_heap: Rc<DescriptorHeap>,
    pub sampler_heap: Rc<DescriptorHeap>,
}

impl Device {
    pub fn new(use_debug: bool) -> Self {
        let flags = if use_debug {
            dx::FactoryCreationFlags::Debug
        } else {
            dx::FactoryCreationFlags::empty()
        };

        let factory = dx::create_factory4(flags).expect("Failed to create DXGI factory");
        let adapter = Self::get_adapter(&factory, true);

        let device = dx::create_device(Some(&adapter), dx::FeatureLevel::Level11)
            .expect("Failed to create device");

        let debug = if use_debug {
            let debug: dx::Debug1 = dx::create_debug()
                .expect("Failed to create debug")
                .try_into()
                .expect("Faile to fetch debug1");

            debug.enable_debug_layer();
            debug.set_enable_gpu_based_validation(true);
            debug.set_callback(Box::new(|_, _, _, msg| {
                println!("[d3d12] {}", msg);
            }));

            Some(debug)
        } else {
            None
        };

        let rtv_heap = DescriptorHeap::new(&device, dx::DescriptorHeapType::Rtv, 2048);
        let dsv_heap = DescriptorHeap::new(&device, dx::DescriptorHeapType::Dsv, 2048);
        let shader_heap =
            DescriptorHeap::new(&device, dx::DescriptorHeapType::CbvSrvUav, 1_000_000);
        let sampler_heap = DescriptorHeap::new(&device, dx::DescriptorHeapType::Sampler, 1024);

        Self {
            factory,
            adapter,
            gpu: device,
            debug,
            rtv_heap: Rc::new(rtv_heap),
            dsv_heap: Rc::new(dsv_heap),
            shader_heap: Rc::new(shader_heap),
            sampler_heap: Rc::new(sampler_heap),
        }
    }

    fn get_adapter(factory: &dx::Factory4, high_perf: bool) -> dx::Adapter3 {
        if let Ok(factory) = TryInto::<dx::Factory7>::try_into(factory.clone()) {
            let mut i = 0;
            let pref = if high_perf {
                dx::GpuPreference::HighPerformance
            } else {
                dx::GpuPreference::Unspecified
            };

            while let Ok(adapter) = factory.enum_adapters_by_gpu_preference(i, pref) {
                if dx::create_device(Some(&adapter), dx::FeatureLevel::Level11).is_ok() {
                    return adapter;
                }

                i += 1;
            }
        }

        let mut i = 0;
        while let Ok(adapter) = factory.enum_adapters(i) {
            if dx::create_device(Some(&adapter), dx::FeatureLevel::Level11).is_ok() {
                return adapter;
            }

            i += 1;
        }

        panic!("Failed to fetch adapter")
    }
}

pub struct DescriptorHeap {
    pub heap: dx::DescriptorHeap,
    pub ty: dx::DescriptorHeapType,
    pub size: usize,
    pub inc_size: usize,
    pub shader_visible: bool,
    pub descriptors: RefCell<Vec<bool>>,
}

impl DescriptorHeap {
    pub fn new(device: &dx::Device, ty: dx::DescriptorHeapType, size: usize) -> Self {
        let descriptors = RefCell::new(vec![false; size]);

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
            heap,
            ty,
            size,
            inc_size,
            shader_visible,
            descriptors,
        }
    }

    pub fn alloc(self: &Rc<Self>) -> Descriptor {
        let mut descriptors = self.descriptors.borrow_mut();

        let Some((index, val)) = descriptors
            .iter_mut()
            .enumerate()
            .skip_while(|(_, b)| **b == true)
            .next()
        else {
            panic!("Out of memory in descriptor heap");
        };
        *val = true;

        let cpu = self
            .heap
            .get_cpu_descriptor_handle_for_heap_start()
            .advance(index, self.inc_size);
        let gpu = self
            .heap
            .get_gpu_descriptor_handle_for_heap_start()
            .advance(index, self.inc_size);

        Descriptor {
            heap_index: index,
            cpu,
            gpu,
            parent: Rc::clone(self),
        }
    }
}

pub struct Descriptor {
    pub heap_index: usize,
    pub cpu: dx::CpuDescriptorHandle,
    pub gpu: dx::GpuDescriptorHandle,
    pub parent: Rc<DescriptorHeap>,
}

impl Drop for Descriptor {
    fn drop(&mut self) {
        self.parent.descriptors.borrow_mut()[self.heap_index] = false;
    }
}

pub struct Fence {
    pub fence: dx::Fence,
    pub value: RefCell<u64>,
}

impl Fence {
    pub fn new(device: &Device) -> Self {
        let fence = device
            .gpu
            .create_fence(0, dx::FenceFlags::empty())
            .expect("Failed to create fence");

        Self {
            fence,
            value: Default::default(),
        }
    }

    pub fn wait(&self, value: u64) {
        if self.completed_value() < value {
            let event = dx::Event::create(false, false).expect("Failed to create event");
            self.fence
                .set_event_on_completion(value, event)
                .expect("Failed to bind fence to event");
            if event.wait(10_000_000) == 0x00000102 {
                panic!("Device lost")
            }
        }
    }

    pub fn completed_value(&self) -> u64 {
        self.fence.get_completed_value()
    }
}

pub struct CommandQueue {
    pub queue: dx::CommandQueue,
    pub ty: dx::CommandListType,
}

impl CommandQueue {
    pub fn new(device: &Device, ty: dx::CommandListType) -> Self {
        let queue = device
            .gpu
            .create_command_queue(&dx::CommandQueueDesc::new(ty))
            .expect("Failed to create command queue");

        Self { queue, ty }
    }

    pub fn wait(&self, fence: &Fence, value: u64) {
        self.queue
            .wait(&fence.fence, value)
            .expect("Failed to queue wait");
    }

    pub fn signal(&self, fence: &Fence) -> u64 {
        let mut guard = fence.value.borrow_mut();
        *guard += 1;
        self.queue
            .signal(&fence.fence, *guard)
            .expect("Failed to signal");
        *guard
    }
}

pub struct GpuResource {
    pub res: dx::Resource,
    pub name: String,
    pub uuid: u64,
}

pub struct GpuResourceTracker {
    pub tracked: RefCell<Vec<Rc<GpuResource>>>,
    pub device: Rc<Device>,
}

impl GpuResourceTracker {
    pub fn new(device: &Rc<Device>) -> Self {
        Self {
            device: Rc::clone(device),
            tracked: Default::default(),
        }
    }

    pub fn alloc(
        &self,
        desc: &dx::ResourceDesc,
        heap_props: &dx::HeapProperties,
        state: dx::ResourceStates,
        name: impl ToString,
    ) -> Rc<GpuResource> {
        let name = name.to_string();
        let uuid = new_uuid();

        let res = self
            .device
            .gpu
            .create_committed_resource(heap_props, dx::HeapFlags::empty(), desc, state, None)
            .expect(&format!("Failed to create resource {}", name));

        let res = Rc::new(GpuResource { res, name, uuid });
        self.register(Rc::clone(&res));

        res
    }

    pub fn register(&self, res: Rc<GpuResource>) {
        self.tracked.borrow_mut().push(res);
    }

    pub fn free(&self, res: &GpuResource) {
        self.tracked.borrow_mut().retain(|r| r.uuid != res.uuid);
    }

    pub fn report(&self) {
        for res in self.tracked.borrow().iter() {
            println!("[d3d12] report reosurce {}", res.name)
        }
    }
}

pub struct Texture {
    pub res: Rc<GpuResource>,
    pub uuid: u64,
    pub width: u32,
    pub height: u32,
    pub levels: u32,
    pub format: dx::Format,
    pub state: RefCell<dx::ResourceStates>,
}

impl Texture {
    pub fn new(
        tracker: &GpuResourceTracker,
        width: u32,
        height: u32,
        format: dx::Format,
        levels: u32,
        flags: dx::ResourceFlags,
        name: impl ToString,
    ) -> Self {
        let name = name.to_string();
        let uuid = new_uuid();
        let desc = dx::ResourceDesc::texture_2d(width, height)
            .with_alignment(dx::HeapAlignment::ResourcePlacement)
            .with_array_size(1)
            .with_format(format)
            .with_mip_levels(levels)
            .with_layout(dx::TextureLayout::Unknown)
            .with_flags(flags);

        let res = tracker.alloc(
            &desc,
            &dx::HeapProperties::default(),
            dx::ResourceStates::Common,
            name,
        );

        Self {
            res,
            uuid,
            width,
            height,
            levels,
            format,
            state: RefCell::new(dx::ResourceStates::Common),
        }
    }

    pub fn get_size(&self, device: dx::Device, mip: Option<u32>) -> usize {
        let mip = mip.map(|m| m..(m + 1)).unwrap_or(0..self.levels);

        let desc = self.res.res.get_desc();
        device.get_copyable_footprints(&desc, mip, 0, &mut vec![], &mut vec![], &mut vec![])
    }
}

pub enum TextureViewType {
    RenderTarget,
    DepthTarget,
    ShaderResource,
    Storage,
}

pub struct TextureView {
    pub parent: Rc<Texture>,
    pub ty: TextureViewType,
    pub handle: Descriptor,
}

impl TextureView {
    pub fn new(
        device: &Device,
        parent: Rc<Texture>,
        ty: TextureViewType,
        mip: Option<u32>,
    ) -> Self {
        let handle = match ty {
            TextureViewType::RenderTarget => {
                let handle = device.rtv_heap.alloc();
                device.gpu.create_render_target_view(
                    Some(&parent.res.res),
                    Some(&dx::RenderTargetViewDesc::texture_2d(parent.format, 0, 0)),
                    handle.cpu,
                );
                handle
            }
            TextureViewType::DepthTarget => {
                let handle = device.dsv_heap.alloc();
                device.gpu.create_depth_stencil_view(
                    Some(&parent.res.res),
                    Some(&dx::DepthStencilViewDesc::texture_2d(parent.format, 0)),
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

                let handle = device.shader_heap.alloc();
                device.gpu.create_shader_resource_view(
                    Some(&parent.res.res),
                    Some(&dx::ShaderResourceViewDesc::texture_2d(
                        parent.format,
                        detailed,
                        mip,
                        0.0,
                        0,
                    )),
                    handle.cpu,
                );
                handle
            }
            TextureViewType::Storage => {
                let mip = mip.unwrap_or(0);
                let handle = device.shader_heap.alloc();

                device.gpu.create_unordered_access_view(
                    Some(&parent.res.res),
                    dx::RES_NONE,
                    Some(&dx::UnorderedAccessViewDesc::texture_2d(
                        parent.format,
                        mip,
                        0,
                    )),
                    handle.cpu,
                );
                handle
            }
        };

        Self { parent, ty, handle }
    }
}

pub struct Swapchain {
    pub swapchain: dx::Swapchain1,
    pub hwnd: NonZero<isize>,
    pub resources: Vec<(dx::Resource, Rc<Texture>, TextureView)>,
    pub width: u32,
    pub height: u32,
}

impl Swapchain {
    pub fn new(
        device: &Device,
        present_queue: &CommandQueue,
        width: u32,
        height: u32,
        hwnd: NonZero<isize>,
    ) -> Self {
        let desc = dx::SwapchainDesc1::new(width, height)
            .with_format(dx::Format::Rgba8Unorm)
            .with_usage(dx::FrameBufferUsage::RenderTargetOutput)
            .with_buffer_count(FRAMES_IN_FLIGHT)
            .with_scaling(dx::Scaling::None)
            .with_swap_effect(dx::SwapEffect::FlipSequential);

        let swapchain = device
            .factory
            .create_swapchain_for_hwnd(&present_queue.queue, hwnd, &desc, None, dx::OUTPUT_NONE)
            .expect("Failed to create swapchain");

        let mut swapchain = Self {
            swapchain,
            hwnd,
            resources: vec![],
            width,
            height,
        };
        swapchain.resize(device, width, height);

        swapchain
    }

    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        {
            std::mem::take(&mut self.resources);
        }

        self.swapchain
            .resize_buffers(
                FRAMES_IN_FLIGHT,
                width,
                height,
                dx::Format::Unknown,
                dx::SwapchainFlags::empty(),
            )
            .expect("Failed to resize swapchain");

        for i in 0..FRAMES_IN_FLIGHT {
            let res: dx::Resource = self
                .swapchain
                .get_buffer(i)
                .expect("Failed to get swapchain buffer");
            let descriptor = device.rtv_heap.alloc();

            device
                .gpu
                .create_render_target_view(Some(&res), None, descriptor.cpu);

            let texture = Rc::new(Texture {
                res: Rc::new(GpuResource {
                    res: res.clone(),
                    name: "Swapchain Image".to_string(),
                    uuid: 0,
                }),
                uuid: 0,
                width,
                height,
                levels: 1,
                format: dx::Format::Rgba8Unorm,
                state: RefCell::new(dx::ResourceStates::Common),
            });

            let view = TextureView {
                parent: Rc::clone(&texture),
                ty: TextureViewType::RenderTarget,
                handle: descriptor,
            };

            self.resources.push((res, texture, view));
        }
    }

    pub fn present(&self, vsync: bool) {
        let interval = if vsync { 1 } else { 0 };

        self.swapchain
            .present(interval, dx::PresentFlags::empty())
            .expect("Failed to present");
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BindingEntry {
    Constants(usize),
    Cbv,
    Uav,
    Srv,
    Sampler,
}

impl BindingEntry {
    pub(crate) fn as_dx(&self) -> dx::DescriptorRangeType {
        match self {
            BindingEntry::Constants(_) => unreachable!(),
            BindingEntry::Cbv => dx::DescriptorRangeType::Cbv,
            BindingEntry::Uav => dx::DescriptorRangeType::Uav,
            BindingEntry::Srv => dx::DescriptorRangeType::Srv,
            BindingEntry::Sampler => dx::DescriptorRangeType::Sampler,
        }
    }
}

pub struct RootSignature {
    pub root: dx::RootSignature,
}

impl RootSignature {
    pub fn new(device: &Device, entries: &[BindingEntry], bindless: bool) -> Self {
        let mut parameters = vec![];
        let mut ranges = vec![];

        for (i, entry) in entries.iter().enumerate() {
            ranges.push(
                dx::DescriptorRange::new(entry.as_dx(), 1)
                    .with_base_shader_register(i)
                    .with_register_space(0),
            );
        }

        for (i, entry) in entries.iter().enumerate() {
            let parameter = if let BindingEntry::Constants(size) = entry {
                dx::RootParameter::constant_32bit(i, 0, *size / 4)
            } else {
                dx::RootParameter::descriptor_table(&[ranges[i]])
                    .with_visibility(dx::ShaderVisibility::All)
            };
            parameters.push(parameter);
        }

        let flags = if bindless {
            dx::RootSignatureFlags::AllowInputAssemblerInputLayout
                | dx::RootSignatureFlags::CbvSrvUavHeapDirectlyIndexed
                | dx::RootSignatureFlags::SamplerHeapDirectlyIndexed
        } else {
            dx::RootSignatureFlags::AllowInputAssemblerInputLayout
        };

        let desc = dx::RootSignatureDesc::default()
            .with_parameters(&parameters)
            .with_flags(flags);

        let root = device
            .gpu
            .serialize_and_create_root_signature(&desc, dx::RootSignatureVersion::V1_0, 0)
            .expect("Failed to create root signature");

        Self { root }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DepthOp {
    None,
    Less,
}

impl DepthOp {
    pub(crate) fn as_dx(&self) -> dx::ComparisonFunc {
        match self {
            DepthOp::None => dx::ComparisonFunc::Never,
            DepthOp::Less => dx::ComparisonFunc::Less,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShaderType {
    Vertex,
    Pixel,
}

pub struct CompiledShader {
    pub bytes: Vec<u8>,
}

pub struct PipelineDesc {
    pub line: bool,
    pub depth: bool,
    pub depth_format: dx::Format,
    pub op: DepthOp,
    pub wireframe: bool,
    pub signature: Option<Rc<RootSignature>>,
    pub formats: Vec<dx::Format>,
    pub shaders: HashMap<ShaderType, CompiledShader>,
}

pub struct GraphicsPipeline {
    pub pso: dx::PipelineState,
    pub root_signature: Option<Rc<RootSignature>>,
}

impl GraphicsPipeline {
    pub fn new(device: &Device, desc: &PipelineDesc) -> Self {
        todo!()
    }
}
