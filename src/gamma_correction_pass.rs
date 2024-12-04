use std::{path::PathBuf, rc::Rc, sync::Arc};

use oxidx::dx;

use crate::{gbuffer_pass::GbufferPass, rhi};

#[derive(Debug)]
pub struct GammaCorrectionPass {
    pub pso: rhi::PipelineHandle,
}

impl GammaCorrectionPass {
    pub fn new(
        device: &Arc<rhi::Device>,
        shader_cache: &mut rhi::ShaderCache,
        pso_cache: &mut rhi::RasterPipelineCache,
    ) -> Self {
        let rs = Rc::new(rhi::RootSignature::new(
            &device,
            rhi::RootSignatureDesc {
                entries: vec![rhi::BindingEntry::Srv { num: 1, slot: 0 }],
                static_samplers: vec![rhi::StaticSampler {
                    slot: 0,
                    filter: dx::Filter::Linear,
                    address_mode: dx::AddressMode::Wrap,
                    comp_func: dx::ComparisonFunc::None,
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
            path: PathBuf::from("assets/GammaCorr.hlsl"),
            entry_point: "Main".to_string(),
            debug: true,
            defines: vec![],
        });

        let pso = pso_cache.get_pso_by_desc(
            rhi::RasterPipelineDesc {
                input_elements: vec![],
                vs: vs,
                line: false,
                depth: None,
                wireframe: false,
                signature: Some(rs),
                formats: vec![],
                shaders: vec![ps],
                depth_bias: 0,
                slope_bias: 0.0,
            },
            &shader_cache,
        );

        Self { pso }
    }

    pub fn render(
        &self,
        device: &Arc<rhi::Device>,
        pso_cache: &rhi::RasterPipelineCache,
        gbuffer: &GbufferPass,
        swapchain_view: &rhi::DeviceTextureView,
    ) {
        let list = device.gfx_queue.get_command_buffer(&device);

        list.set_viewport(gbuffer.width, gbuffer.height);
        list.set_graphics_pipeline(pso_cache.get_pso(&self.pso));
        list.set_topology(rhi::GeomTopology::Triangles);
        list.set_device_texture_barriers(&[(
            &gbuffer.accum,
            dx::ResourceStates::PixelShaderResource,
        )]);

        list.clear_render_target(&swapchain_view, 1.0, 1.0, 1.0);
        list.set_render_targets(&[swapchain_view], None);

        list.set_graphics_srv(&gbuffer.accum_srv, 0);
        list.draw(3);

        device.gfx_queue.stash_cmd_buffer(list);
    }
}
