use std::{path::PathBuf, rc::Rc, sync::Arc};

use oxidx::dx;

use crate::{csm_pass::CascadedShadowMapsPass, rhi, zpass::ZPass};

#[derive(Debug)]
pub struct ShadowMaskPass {
    pub width: u32,
    pub height: u32,
    pub texture: rhi::DeviceTexture,
    pub texture_rtv: rhi::DeviceTextureView,
    pub texture_srv: rhi::DeviceTextureView,
    pub pso: rhi::PipelineHandle,
}

impl ShadowMaskPass {
    pub fn new(
        device: &Arc<rhi::Device>,
        width: u32,
        height: u32,
        shader_cache: &mut rhi::ShaderCache,
        pso_cache: &mut rhi::RasterPipelineCache,
    ) -> Self {
        let texture = rhi::DeviceTexture::new(
            device,
            width,
            height,
            1,
            dx::Format::R8Unorm,
            1,
            dx::ResourceFlags::AllowRenderTarget,
            dx::ResourceStates::RenderTarget,
            Some(dx::ClearValue::color(
                dx::Format::R8Unorm,
                [1.0, 1.0, 1.0, 1.0],
            )),
            "Shadow Mask",
        );

        let texture_rtv = rhi::DeviceTextureView::new(
            &device,
            &texture,
            dx::Format::R8Unorm,
            rhi::TextureViewType::RenderTarget,
            None,
        );
        let texture_srv = rhi::DeviceTextureView::new(
            &device,
            &texture,
            dx::Format::R8Unorm,
            rhi::TextureViewType::ShaderResource,
            None,
        );

        let rs = Rc::new(rhi::RootSignature::new(
            &device,
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

        let pso = pso_cache.get_pso_by_desc(
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
            texture,
            texture_rtv,
            texture_srv,
            width,
            height,
            pso,
        }
    }

    pub fn render(
        &self,
        device: &Arc<rhi::Device>,
        camera_buffer: &rhi::Buffer,
        pso_cache: &rhi::RasterPipelineCache,
        frame_idx: usize,
        depth: &ZPass,
        csm: &CascadedShadowMapsPass,
    ) {
        let list = device.gfx_queue.get_command_buffer(&device);

        list.set_viewport(self.width, self.height);
        list.set_graphics_pipeline(pso_cache.get_pso(&self.pso));
        list.set_topology(rhi::GeomTopology::Triangles);
        list.set_device_texture_barriers(&[
            (&self.texture, dx::ResourceStates::RenderTarget),
            (&depth.depth, dx::ResourceStates::PixelShaderResource),
            (&csm.texture, dx::ResourceStates::PixelShaderResource),
        ]);

        list.clear_render_target(&self.texture_rtv, 1.0, 1.0, 1.0);
        list.set_render_targets(&[&self.texture_rtv], None);
        list.set_graphics_cbv(
            &camera_buffer
                .get_buffer(list.device_id)
                .expect("Failed to get buffer")
                .cbv[frame_idx],
            0,
        );

        list.set_graphics_cbv(&csm.gpu_csm_buffer.cbv[frame_idx], 1);

        list.set_graphics_srv(&depth.depth_srv, 2);
        list.set_graphics_srv(&csm.srv, 3);
        list.draw(3);

        device.gfx_queue.stash_cmd_buffer(list);
    }
}
