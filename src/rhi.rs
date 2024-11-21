use std::{
    cell::RefCell, collections::HashMap, ffi::CString, num::NonZero, ops::Range, path::Path,
    rc::Rc, sync::atomic::AtomicU64,
};

use oxidx::dx::{
    self, IAdapter3, IBlobExt, ICommandAllocator, ICommandQueue, IDebug, IDebug1, IDebugExt,
    IDescriptorHeap, IDevice, IFactory4, IFactory6, IFence, IGraphicsCommandList, IResource,
    IShaderReflection, ISwapchain1, PSO_NONE, RES_NONE,
};
use parking_lot::Mutex;

use crate::utils::{new_uuid, Id};

pub const FRAMES_IN_FLIGHT: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceSettings<'a> {
    pub use_debug: bool,
    pub ty: AdapterType,
    pub excluded_adapters: &'a [&'a dx::Adapter3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterType {
    Software,
    Hardware { high_perf: bool },
}

pub struct Device {
    pub id: Id<Device>,
    pub factory: dx::Factory4,
    pub adapter: dx::Adapter3,
    pub gpu: dx::Device,
    pub debug: Option<dx::Debug1>,

    pub rtv_heap: DescriptorHeap,
    pub dsv_heap: DescriptorHeap,
    pub shader_heap: DescriptorHeap,
    pub sampler_heap: DescriptorHeap,
}

impl Device {
    pub fn fetch(settings: DeviceSettings<'_>) -> Option<Self> {
        let flags = if settings.use_debug {
            dx::FactoryCreationFlags::Debug
        } else {
            dx::FactoryCreationFlags::empty()
        };

        let factory = dx::create_factory4(flags).expect("Failed to create DXGI factory");

        let debug = if settings.use_debug {
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

        let adapter = Self::get_adapter(&factory, &settings)?;

        let device = dx::create_device(Some(&adapter), dx::FeatureLevel::Level11)
            .expect("Failed to create device");

        let rtv_heap = DescriptorHeap::new(&device, dx::DescriptorHeapType::Rtv, 128);
        let dsv_heap = DescriptorHeap::new(&device, dx::DescriptorHeapType::Dsv, 128);
        let shader_heap = DescriptorHeap::new(&device, dx::DescriptorHeapType::CbvSrvUav, 1024);
        let sampler_heap = DescriptorHeap::new(&device, dx::DescriptorHeapType::Sampler, 32);

        let id = Id::new();

        Some(Self {
            id,

            factory,
            adapter,
            gpu: device,
            debug,

            rtv_heap,
            dsv_heap,
            shader_heap,
            sampler_heap,
        })
    }

    pub fn create_resource(
        &self,
        desc: &dx::ResourceDesc,
        heap_props: &dx::HeapProperties,
        state: dx::ResourceStates,
        name: impl ToString,
    ) -> GpuResource {
        let name = name.to_string();
        let uuid = new_uuid();

        let res = self
            .gpu
            .create_committed_resource(heap_props, dx::HeapFlags::empty(), desc, state, None)
            .expect(&format!("Failed to create resource {}", name));

        GpuResource { res, name, uuid }
    }

    fn get_adapter(factory: &dx::Factory4, settings: &DeviceSettings) -> Option<dx::Adapter3> {
        match settings.ty {
            AdapterType::Software => {
                Self::get_software_adapter(factory, settings.excluded_adapters)
            }
            AdapterType::Hardware { high_perf } => {
                Self::get_hardware_adapter(factory, high_perf, settings.excluded_adapters)
            }
        }
    }

    fn get_software_adapter(
        factory: &dx::Factory4,
        excluded: &[&dx::Adapter3],
    ) -> Option<dx::Adapter3> {
        let adapter = factory.enum_warp_adapters().ok()?;

        if Self::is_exclude_adapter_contains_adapter(excluded, &adapter) {
            return None;
        }

        Some(adapter)
    }

    fn get_hardware_adapter(
        factory: &dx::Factory4,
        high_perf: bool,
        excluded: &[&dx::Adapter3],
    ) -> Option<dx::Adapter3> {
        if let Ok(factory) = TryInto::<dx::Factory7>::try_into(factory.clone()) {
            let mut i = 0;
            let pref = if high_perf {
                dx::GpuPreference::HighPerformance
            } else {
                dx::GpuPreference::Unspecified
            };

            while let Ok(adapter) = factory.enum_adapters_by_gpu_preference(i, pref) {
                if Self::is_exclude_adapter_contains_adapter(excluded, &adapter) {
                    i += 1;
                    continue;
                }

                if dx::create_device(Some(&adapter), dx::FeatureLevel::Level11).is_ok() {
                    return Some(adapter);
                }

                i += 1;
            }
        }

        let mut i = 0;
        while let Ok(adapter) = factory.enum_adapters(i) {
            if Self::is_exclude_adapter_contains_adapter(excluded, &adapter) {
                i += 1;
                continue;
            }

            if dx::create_device(Some(&adapter), dx::FeatureLevel::Level11).is_ok() {
                return Some(adapter);
            }

            i += 1;
        }

        None
    }

    fn is_exclude_adapter_contains_adapter(
        excluded: &[&dx::Adapter3],
        adapter: &dx::Adapter3,
    ) -> bool {
        excluded
            .iter()
            .find(|a| {
                match (
                    a.get_desc1().map(|d| d.adapter_luid()),
                    adapter.get_desc1().map(|d| d.adapter_luid()),
                ) {
                    (Ok(l1), Ok(l2)) => l1 == l2,
                    _ => false,
                }
            })
            .is_some()
    }
}

#[derive(Debug)]
pub struct DescriptorHeap {
    pub heap: dx::DescriptorHeap,
    pub ty: dx::DescriptorHeapType,
    pub size: usize,
    pub inc_size: usize,
    pub shader_visible: bool,
    pub descriptors: Mutex<Vec<bool>>,
}

impl DescriptorHeap {
    pub fn new(device: &dx::Device, ty: dx::DescriptorHeapType, size: usize) -> Self {
        let descriptors = Mutex::new(vec![false; size]);

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

    pub fn alloc(&self) -> Descriptor {
        let mut descriptors = self.descriptors.lock();

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
        }
    }

    pub fn free(&self, descriptor: Descriptor) {
        self.descriptors.lock()[descriptor.heap_index] = false;
    }
}

#[derive(Debug)]
pub struct Descriptor {
    pub heap_index: usize,
    pub cpu: dx::CpuDescriptorHandle,
    pub gpu: dx::GpuDescriptorHandle,
}

pub struct Fence {
    pub fence: dx::Fence,
    pub value: AtomicU64,
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

    pub fn inc_value(&self) -> u64 {
        self.value
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1
    }

    pub fn completed_value(&self) -> u64 {
        self.fence.get_completed_value()
    }

    pub fn get_current_value(&self) -> u64 {
        self.value.load(std::sync::atomic::Ordering::Relaxed)
    }
}

pub struct CommandQueue {
    pub queue: Mutex<dx::CommandQueue>,
    pub ty: dx::CommandListType,

    pub fence: Fence,

    pub frequency: f64,
}

impl CommandQueue {
    pub fn new(device: &Device, ty: dx::CommandListType) -> Self {
        let queue = device
            .gpu
            .create_command_queue(&dx::CommandQueueDesc::new(ty))
            .expect("Failed to create command queue");

        let fence = Fence::new(device);

        let frequency = 1000.0
            / queue
                .get_timestamp_frequency()
                .expect("Failed to fetch timestamp frequency") as f64;

        Self {
            queue: Mutex::new(queue),
            ty,
            fence,
            frequency,
        }
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

    pub fn signal(&self) -> u64 {
        let value = self.fence.inc_value();
        self.queue
            .lock()
            .signal(&self.fence.fence, value)
            .expect("Failed to signal");
        value
    }

    pub fn submit(&self, buffers: &[&CommandBuffer]) {
        let lists = buffers
            .iter()
            .map(|b| Some(b.list.clone()))
            .collect::<Vec<_>>();

        self.queue.lock().execute_command_lists(&lists);
    }
}

#[derive(Debug)]
pub struct GpuResource {
    pub res: dx::Resource,
    pub name: String,
    pub uuid: u64,
}

#[derive(Debug)]
pub struct Texture {
    pub res: GpuResource,
    pub uuid: u64,
    pub width: u32,
    pub height: u32,
    pub levels: u32,
    pub format: dx::Format,
    pub state: RefCell<dx::ResourceStates>,
}

impl Texture {
    pub fn new(
        device: &Device,
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

        let res = device.create_resource(
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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
            .create_swapchain_for_hwnd(
                &*present_queue.queue.lock(),
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
                res: GpuResource {
                    res: res.clone(),
                    name: "Swapchain Image".to_string(),
                    uuid: 0,
                },
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
            ranges.push([dx::DescriptorRange::new(entry.as_dx(), 1)
                .with_base_shader_register(i as u32)
                .with_register_space(0)]);
        }

        for (i, entry) in entries.iter().enumerate() {
            let parameter = if let BindingEntry::Constants(size) = entry {
                dx::RootParameter::constant_32bit(i as u32, 0, (*size / 4) as u32)
            } else {
                dx::RootParameter::descriptor_table(&ranges[i])
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShaderType {
    Vertex,
    Pixel,
}

pub struct CompiledShader {
    pub ty: ShaderType,
    pub raw: dx::Blob,
}

impl CompiledShader {
    pub fn compile(path: impl AsRef<Path>, ty: ShaderType) -> Self {
        let target = match ty {
            ShaderType::Vertex => c"vs_5_0",
            ShaderType::Pixel => c"ps_5_0",
        };

        let raw = dx::Blob::compile_from_file(path, &[], c"Main", target, 0, 0)
            .expect("Failed to compile a shader");

        Self { ty, raw }
    }
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
        let vert = desc
            .shaders
            .get(&ShaderType::Vertex)
            .expect("Not found vertex shader");
        let pixel = desc.shaders.get(&ShaderType::Pixel);

        let vert_reflection = vert.raw.reflect().expect("Failed to get reflection");
        let vertex_desc = vert_reflection
            .get_desc()
            .expect("Failed to get shader reflection");

        let n = vertex_desc.get_input_parameters();
        let mut input_element_desc = vec![];
        let mut input_sematics_name = vec![CString::new("").unwrap(); n as usize];

        for i in 0..n {
            let param_desc = vert_reflection
                .get_input_parameter_desc(i as usize)
                .expect("Failed fetch input parameter desc");

            input_sematics_name[i as usize] = param_desc.semantic_name().to_owned();

            let format = if param_desc.mask() == 1 {
                if param_desc.component_type() == dx::RegisterComponentType::Uint32 {
                    dx::Format::R32Uint
                } else if param_desc.component_type() == dx::RegisterComponentType::Sint32 {
                    dx::Format::R32Sint
                } else if param_desc.component_type() == dx::RegisterComponentType::Float32 {
                    dx::Format::R32Float
                } else {
                    dx::Format::Unknown
                }
            } else if param_desc.mask() <= 3 {
                if param_desc.component_type() == dx::RegisterComponentType::Uint32 {
                    dx::Format::Rg32Uint
                } else if param_desc.component_type() == dx::RegisterComponentType::Sint32 {
                    dx::Format::Rg32Sint
                } else if param_desc.component_type() == dx::RegisterComponentType::Float32 {
                    dx::Format::Rg32Float
                } else {
                    dx::Format::Unknown
                }
            } else if param_desc.mask() <= 7 {
                if param_desc.component_type() == dx::RegisterComponentType::Uint32 {
                    dx::Format::Rgb32Uint
                } else if param_desc.component_type() == dx::RegisterComponentType::Sint32 {
                    dx::Format::Rgb32Sint
                } else if param_desc.component_type() == dx::RegisterComponentType::Float32 {
                    dx::Format::Rgb32Float
                } else {
                    dx::Format::Unknown
                }
            } else if param_desc.mask() <= 15 {
                if param_desc.component_type() == dx::RegisterComponentType::Uint32 {
                    dx::Format::Rgba32Uint
                } else if param_desc.component_type() == dx::RegisterComponentType::Sint32 {
                    dx::Format::Rgba32Sint
                } else if param_desc.component_type() == dx::RegisterComponentType::Float32 {
                    dx::Format::Rgba32Float
                } else {
                    dx::Format::Unknown
                }
            } else {
                dx::Format::Unknown
            };

            let input_element = dx::InputElementDesc::from_raw_per_vertex(
                &input_sematics_name[i as usize],
                param_desc.semantic_index(),
                format,
                0,
            );

            input_element_desc.push(input_element);
        }

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
                    .with_cull_mode(dx::CullMode::Front),
            )
            .with_primitive_topology(if desc.line {
                dx::PipelinePrimitiveTopology::Line
            } else {
                dx::PipelinePrimitiveTopology::Triangle
            });

        let de = if desc.depth {
            de.with_depth_stencil(
                dx::DepthStencilDesc::default().enable_depth(desc.op.as_dx()),
                desc.depth_format,
            )
        } else {
            de
        };

        let (de, rs) = if let Some(rs) = &desc.signature {
            (de.with_root_signature(&rs.root), Some(Rc::clone(rs)))
        } else {
            (de, None)
        };

        let de = if let Some(ps) = pixel {
            de.with_ps(&ps.raw)
        } else {
            de
        };

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
    pub res: GpuResource,
    pub ty: BufferType,
    pub state: RefCell<dx::ResourceStates>,

    pub size: usize,
    pub stride: usize,

    pub srv: Option<Descriptor>,
    pub uav: Option<Descriptor>,
    pub cbv: Option<Descriptor>,

    pub vbv: Option<dx::VertexBufferView>,
    pub ibv: Option<dx::IndexBufferView>,
}

impl Buffer {
    pub fn new(
        device: &Device,
        size: usize,
        stride: usize,
        ty: BufferType,
        readback: bool,
        name: impl ToString,
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

        let desc = dx::ResourceDesc::buffer(size).with_flags(flags);

        let res = device.create_resource(&desc, &heap_props, state, name);

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
            cbv: None,
            vbv,
            ibv,
        }
    }

    pub fn build_constant(&mut self, device: &Device) {
        if self.cbv.is_some() {
            return;
        }

        let d = device.shader_heap.alloc();
        let desc =
            dx::ConstantBufferViewDesc::new(self.res.res.get_gpu_virtual_address(), self.size);
        device.gpu.create_constant_buffer_view(Some(&desc), d.cpu);

        self.cbv = Some(d);
    }

    pub fn build_storage(&mut self, device: &Device) {
        if self.uav.is_some() {
            return;
        }

        let d = device.shader_heap.alloc();
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

    pub fn build_shader_resource(&mut self, device: &Device) {
        if self.srv.is_some() {
            return;
        }

        let d = device.shader_heap.alloc();
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

    pub fn map<'a, T>(&'a mut self, range: Option<Range<usize>>) -> BufferMap<'a, T> {
        let size = if let Some(ref r) = range {
            r.end - r.start
        } else {
            self.size / size_of::<T>()
        };

        let pointer = self
            .res
            .res
            .map::<T>(0, range.clone())
            .expect("Failed to map buffer");

        unsafe {
            let pointer = std::slice::from_raw_parts_mut(pointer.as_ptr(), size);

            BufferMap {
                buffer: self,
                range,
                pointer,
            }
        }
    }
}

pub struct BufferMap<'a, T> {
    buffer: &'a Buffer,
    range: Option<Range<usize>>,
    pub pointer: &'a mut [T],
}

impl<'a, T> Drop for BufferMap<'a, T> {
    fn drop(&mut self) {
        self.buffer.res.res.unmap(0, self.range.clone());
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
    pub fn new(device: &Device, address: SamplerAddress, filter: SamplerFilter) -> Self {
        let handle = device.sampler_heap.alloc();

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

pub struct CommandBuffer {
    pub ty: dx::CommandListType,
    pub list: dx::GraphicsCommandList,
    pub allocator: dx::CommandAllocator,
}

impl CommandBuffer {
    pub fn new(device: &Device, ty: dx::CommandListType, close: bool) -> Self {
        let allocator = device
            .gpu
            .create_command_allocator(ty)
            .expect("Failed to create command allocator");
        let list: dx::GraphicsCommandList = device
            .gpu
            .create_command_list(0, ty, &allocator, PSO_NONE)
            .expect("Failed to create command list");

        if close {
            list.close().expect("Failed to close");
        }

        Self {
            ty,
            list,
            allocator,
        }
    }

    pub fn begin(&self, device: &Device, reset: bool) {
        if reset {
            self.allocator.reset();
            self.list
                .reset(&self.allocator, PSO_NONE)
                .expect("Failed to reset list");
        }

        self.list.set_descriptor_heaps(&[
            Some(device.shader_heap.heap.clone()),
            Some(device.sampler_heap.heap.clone()),
        ]);
    }

    pub fn end(&self) {
        self.list.close().expect("Failed to close list");
    }

    pub fn set_render_targets(&self, views: &[&TextureView], depth: Option<&TextureView>) {
        let rtvs = views.iter().map(|view| view.handle.cpu).collect::<Vec<_>>();

        let dsv = depth.map(|view| view.handle.cpu);

        self.list.om_set_render_targets(&rtvs, false, dsv);
    }

    pub fn set_buffer_barrier(&self, buffer: &Buffer, state: dx::ResourceStates) {
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

    pub fn set_image_barrier(
        &self,
        texture: &Texture,
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

    pub fn clear_render_target(&self, view: &TextureView, r: f32, g: f32, b: f32) {
        self.list
            .clear_render_target_view(view.handle.cpu, [r, g, b, 1.0], &[]);
    }

    pub fn clear_depth_target(&self, view: &TextureView) {
        self.list
            .clear_depth_stencil_view(view.handle.cpu, dx::ClearFlags::Depth, 1.0, 0, &[]);
    }

    pub fn set_graphics_pipeline(&self, pipeline: &GraphicsPipeline) {
        self.list
            .set_graphics_root_signature(pipeline.root_signature.as_ref().map(|rs| &rs.root));
        self.list.set_pipeline_state(&pipeline.pso);
    }

    pub fn set_vertex_buffer(&self, buffer: &Buffer) {
        self.list
            .ia_set_vertex_buffers(0, &[buffer.vbv.expect("Expected vertex buffer")]);
    }

    pub fn set_index_buffer(&self, buffer: &Buffer) {
        self.list
            .ia_set_index_buffer(Some(&buffer.ibv.expect("Expected index buffer")));
    }

    pub fn set_graphics_srv(&self, view: &TextureView, index: usize) {
        self.list
            .set_graphics_root_descriptor_table(index as u32, view.handle.gpu);
    }

    pub fn set_graphics_cbv(&self, buffer: &Buffer, index: usize) {
        self.list.set_graphics_root_descriptor_table(
            index as u32,
            buffer.cbv.as_ref().expect("Expected constant buffer").gpu,
        );
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

    pub fn copy_texture_to_texture(&self, dst: &Texture, src: &Texture) {
        self.list.copy_resource(&dst.res.res, &src.res.res);
    }

    pub fn copy_buffer_to_texture(&self, device: &Device, dst: &Texture, src: &Buffer) {
        let desc = dst.res.res.get_desc();
        let mut footprints = vec![dx::PlacedSubresourceFootprint::default(); dst.levels as usize];
        let mut num_rows = vec![Default::default(); dst.levels as usize];
        let mut row_sizes = vec![Default::default(); dst.levels as usize];

        let total_size = device.gpu.get_copyable_footprints(
            &desc,
            0..dst.levels,
            0,
            &mut footprints,
            &mut num_rows,
            &mut row_sizes,
        );

        for i in 0..(dst.levels as usize) {
            let src_copy = dx::TextureCopyLocation::placed_footprint(&src.res.res, footprints[i]);
            let dst_copy = dx::TextureCopyLocation::subresource(&dst.res.res, i as u32);

            self.list
                .copy_texture_region(&dst_copy, 0, 0, 0, &src_copy, None);
        }
    }

    pub fn copy_buffer_to_buffer(&self, dst: &Buffer, src: &Buffer) {
        self.list.copy_resource(&dst.res.res, &src.res.res);
    }
}
