use std::{
    path::PathBuf,
    rc::Rc,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use oxidx::dx;

use crate::{
    csm_pass::CascadedShadowMapsPass,
    rhi::{self, FRAMES_IN_FLIGHT},
    zpass::ZPass,
    GpuGlobals,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MgpuState {
    #[default]
    WaitForWrite,
    WaitForCopy(u64),
    WaitForRead(u64),
}

#[derive(Debug)]
pub struct MgpuShadowMaskPass {
    pub width: u32,
    pub height: u32,

    pub camera_buffers: rhi::DeviceBuffer,
    pub sender: [rhi::SharedTexture; FRAMES_IN_FLIGHT],
    pub sender_rtv: [rhi::DeviceTextureView; FRAMES_IN_FLIGHT],
    pub sender_fence: rhi::SharedFence,

    pub recv: [rhi::SharedTexture; FRAMES_IN_FLIGHT],
    pub recv_srv: [rhi::DeviceTextureView; FRAMES_IN_FLIGHT],
    pub recv_fence: rhi::SharedFence,

    pub pso: rhi::PipelineHandle,

    pub working_texture: AtomicUsize,
    pub copy_texture: AtomicUsize,
    pub states: [MgpuState; FRAMES_IN_FLIGHT],
}

impl MgpuShadowMaskPass {
    pub fn new(
        primary_gpu: &Arc<rhi::Device>,
        secondary_gpu: &Arc<rhi::Device>,
        width: u32,
        height: u32,
        shader_cache: &mut rhi::ShaderCache,
        secondary_pso_cache: &mut rhi::RasterPipelineCache,
    ) -> Self {
        let sender = std::array::from_fn(|_| {
            rhi::SharedTexture::new(
                secondary_gpu,
                width,
                height,
                1,
                dx::Format::R8Unorm,
                dx::ResourceFlags::AllowRenderTarget,
                dx::ResourceStates::RenderTarget,
                dx::ResourceStates::CopySource,
                Some(dx::ClearValue::color(
                    dx::Format::R8Unorm,
                    [1.0, 1.0, 1.0, 1.0],
                )),
                "Secondary Shadow Mask",
            )
        });

        let sender_rtv = std::array::from_fn(|i| {
            rhi::DeviceTextureView::new(
                &secondary_gpu,
                sender[i].local_resource(),
                dx::Format::R8Unorm,
                rhi::TextureViewType::RenderTarget,
                None,
            )
        });

        let sender_fence = rhi::SharedFence::new(&secondary_gpu);

        let recv = std::array::from_fn(|i| {
            sender[i].connect_texture(
                &primary_gpu,
                dx::ResourceStates::Common,
                dx::ResourceStates::Common,
                None,
                "Primary Shadow Mask",
            )
        });

        let recv_srv = std::array::from_fn(|i| {
            rhi::DeviceTextureView::new(
                &primary_gpu,
                recv[i].local_resource(),
                dx::Format::R8Unorm,
                rhi::TextureViewType::ShaderResource,
                None,
            )
        });

        let recv_fence = sender_fence.connect(&primary_gpu);

        let camera_buffers = rhi::DeviceBuffer::constant::<GpuGlobals>(
            &secondary_gpu,
            FRAMES_IN_FLIGHT,
            "Camera Buffers",
        );

        let rs = Rc::new(rhi::RootSignature::new(
            &secondary_gpu,
            rhi::RootSignatureDesc {
                entries: vec![
                    rhi::BindingEntry::Cbv { num: 1, slot: 0 },
                    rhi::BindingEntry::Cbv { num: 1, slot: 1 },
                    rhi::BindingEntry::Srv { num: 1, slot: 2 },
                    rhi::BindingEntry::Srv { num: 1, slot: 3 },
                ],
                static_samplers: vec![rhi::StaticSampler {
                    slot: 0,
                    filter: dx::Filter::ComparisonLinear,
                    address_mode: dx::AddressMode::Border,
                    comp_func: dx::ComparisonFunc::LessEqual,
                    b_color: dx::BorderColor::OpaqueWhite,
                }],
                bindless: false,
            },
        ));

        let vs = shader_cache.get_shader_by_desc(rhi::ShaderDesc {
            ty: rhi::ShaderType::Vertex,
            path: PathBuf::from("assets/FullscreenVS.hlsl"),
            entry_point: "Main".to_string(),
            debug: false,
            defines: vec![],
        });

        let ps = shader_cache.get_shader_by_desc(rhi::ShaderDesc {
            ty: rhi::ShaderType::Pixel,
            path: PathBuf::from("assets/ShadowMask.hlsl"),
            entry_point: "Main".to_string(),
            debug: true,
            defines: vec![],
        });

        let pso = secondary_pso_cache.get_pso_by_desc(
            rhi::RasterPipelineDesc {
                input_elements: vec![
                    rhi::InputElementDesc {
                        semantic: dx::SemanticName::Position(0),
                        format: dx::Format::Rgb32Float,
                        slot: 0,
                    },
                    rhi::InputElementDesc {
                        semantic: dx::SemanticName::TexCoord(0),
                        format: dx::Format::Rg32Float,
                        slot: 1,
                    },
                ],
                vs: vs,
                line: false,
                depth: None,
                wireframe: false,
                signature: Some(rs),
                formats: vec![dx::Format::R8Unorm],
                shaders: vec![ps],
                depth_bias: 0,
                slope_bias: 0.0,
                cull_mode: rhi::CullMode::None,
            },
            &shader_cache,
        );

        Self {
            width,
            height,
            pso,
            sender,
            sender_rtv,
            sender_fence,
            recv,
            recv_srv,
            recv_fence,
            camera_buffers,
            working_texture: Default::default(),
            copy_texture: Default::default(),
            states: Default::default(),
        }
    }

    pub fn render(
        &self,
        device: &Arc<rhi::Device>,
        pso_cache: &rhi::RasterPipelineCache,
        depth: &ZPass,
        csm: &CascadedShadowMapsPass,
    ) {
        let frame_idx = self.working_texture.load(Ordering::Relaxed);

        let list = device.gfx_queue.get_command_buffer(&device);
        list.set_mark("Shadow Mask Pass");

        list.set_viewport(self.width, self.height);
        list.set_graphics_pipeline(pso_cache.get_pso(&self.pso));
        list.set_topology(rhi::GeomTopology::Triangles);
        list.set_device_texture_barriers(&[
            (
                &self.sender[frame_idx].local_resource(),
                dx::ResourceStates::RenderTarget,
            ),
            (&depth.depth, dx::ResourceStates::PixelShaderResource),
            (&csm.texture, dx::ResourceStates::PixelShaderResource),
        ]);

        list.clear_render_target(&self.sender_rtv[frame_idx], 1.0, 1.0, 1.0);
        list.set_render_targets(&[&self.sender_rtv[frame_idx]], None);
        list.set_graphics_cbv(&self.camera_buffers.cbv[frame_idx], 0);

        list.set_graphics_cbv(&csm.gpu_csm_buffer.cbv[frame_idx], 1);

        list.set_graphics_srv(&depth.depth_srv, 2);
        list.set_graphics_srv(&csm.srv, 3);
        list.draw(3);

        device.gfx_queue.stash_cmd_buffer(list);
    }

    pub fn next_working_texture(&self) {
        let idx = (self.working_texture.load(Ordering::Acquire) + 1) % FRAMES_IN_FLIGHT;
        self.working_texture.store(idx, Ordering::Release);
    }

    pub fn next_copy_texture(&self) {
        let idx = (self.copy_texture.load(Ordering::Acquire) + 1) % FRAMES_IN_FLIGHT;
        self.copy_texture.store(idx, Ordering::Release);
    }
}
