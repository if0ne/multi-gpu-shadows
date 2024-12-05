use std::{path::PathBuf, rc::Rc, sync::Arc};

use oxidx::dx;

use crate::{gltf::GpuMesh, rhi, zpass::ZPass, TexturePlaceholders};

#[derive(Debug)]
pub struct GbufferPass {
    pub width: u32,
    pub height: u32,

    pub diffuse: rhi::DeviceTexture,
    pub diffuse_rtv: rhi::DeviceTextureView,
    pub diffuse_srv: rhi::DeviceTextureView,

    pub normal: rhi::DeviceTexture,
    pub normal_rtv: rhi::DeviceTextureView,
    pub normal_srv: rhi::DeviceTextureView,

    pub material: rhi::DeviceTexture,
    pub material_rtv: rhi::DeviceTextureView,
    pub material_srv: rhi::DeviceTextureView,

    pub accum: rhi::DeviceTexture,
    pub accum_rtv: rhi::DeviceTextureView,
    pub accum_srv: rhi::DeviceTextureView,

    pub vs: rhi::ShaderHandle,
    pub ps: rhi::ShaderHandle,
    pub pso: rhi::PipelineHandle,
}

impl GbufferPass {
    pub fn new(
        width: u32,
        height: u32,
        device: &Arc<rhi::Device>,
        shader_cache: &mut rhi::ShaderCache,
        pso_cache: &mut rhi::RasterPipelineCache,
    ) -> Self {
        let diffuse = rhi::DeviceTexture::new(
            device,
            width,
            height,
            1,
            dx::Format::Rgba32Float,
            1,
            dx::ResourceFlags::AllowRenderTarget,
            dx::ResourceStates::RenderTarget,
            Some(dx::ClearValue::color(
                dx::Format::Rgba32Float,
                [0.301, 0.5607, 0.675, 1.0],
            )),
            "Diffuse Texture",
        );

        let diffuse_rtv = rhi::DeviceTextureView::new(
            device,
            &diffuse,
            diffuse.format,
            rhi::TextureViewType::RenderTarget,
            None,
        );

        let diffuse_srv = rhi::DeviceTextureView::new(
            device,
            &diffuse,
            diffuse.format,
            rhi::TextureViewType::ShaderResource,
            None,
        );

        let normal = rhi::DeviceTexture::new(
            device,
            width,
            height,
            1,
            dx::Format::Rgba32Float,
            1,
            dx::ResourceFlags::AllowRenderTarget,
            dx::ResourceStates::RenderTarget,
            Some(dx::ClearValue::color(
                dx::Format::Rgba32Float,
                [0.0, 0.0, 0.0, 1.0],
            )),
            "Normal Texture",
        );

        let normal_rtv = rhi::DeviceTextureView::new(
            device,
            &normal,
            normal.format,
            rhi::TextureViewType::RenderTarget,
            None,
        );

        let normal_srv = rhi::DeviceTextureView::new(
            device,
            &normal,
            normal.format,
            rhi::TextureViewType::ShaderResource,
            None,
        );

        let material = rhi::DeviceTexture::new(
            device,
            width,
            height,
            1,
            dx::Format::Rgba32Float,
            1,
            dx::ResourceFlags::AllowRenderTarget,
            dx::ResourceStates::RenderTarget,
            Some(dx::ClearValue::color(
                dx::Format::Rgba32Float,
                [0.0, 0.0, 0.0, 1.0],
            )),
            "Material Texture",
        );

        let material_rtv = rhi::DeviceTextureView::new(
            device,
            &material,
            material.format,
            rhi::TextureViewType::RenderTarget,
            None,
        );

        let material_srv = rhi::DeviceTextureView::new(
            device,
            &material,
            material.format,
            rhi::TextureViewType::ShaderResource,
            None,
        );

        let accum = rhi::DeviceTexture::new(
            device,
            width,
            height,
            1,
            dx::Format::Rgba32Float,
            1,
            dx::ResourceFlags::AllowRenderTarget,
            dx::ResourceStates::RenderTarget,
            Some(dx::ClearValue::color(
                dx::Format::Rgba32Float,
                [0.0, 0.0, 0.0, 1.0],
            )),
            "Accum Texture",
        );

        let accum_rtv = rhi::DeviceTextureView::new(
            device,
            &accum,
            accum.format,
            rhi::TextureViewType::RenderTarget,
            None,
        );

        let accum_srv = rhi::DeviceTextureView::new(
            device,
            &accum,
            accum.format,
            rhi::TextureViewType::ShaderResource,
            None,
        );

        let depth = rhi::DeviceTexture::new(
            &device,
            width,
            height,
            1,
            dx::Format::D24UnormS8Uint,
            1,
            dx::ResourceFlags::AllowDepthStencil,
            dx::ResourceStates::DepthWrite,
            Some(dx::ClearValue::depth(dx::Format::D24UnormS8Uint, 1.0, 0)),
            "Depth Buffer",
        );

        let depth_dsv = rhi::DeviceTextureView::new(
            &device,
            &depth,
            depth.format,
            rhi::TextureViewType::DepthTarget,
            None,
        );

        let depth_srv = rhi::DeviceTextureView::new(
            &device,
            &depth,
            dx::Format::R24UnormX8Typeless,
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
                    filter: dx::Filter::Linear,
                    address_mode: dx::AddressMode::Wrap,
                    comp_func: dx::ComparisonFunc::None,
                    b_color: dx::BorderColor::OpaqueWhite,
                }],
                bindless: false,
            },
        ));

        let vs = shader_cache.get_shader_by_desc(rhi::ShaderDesc {
            ty: rhi::ShaderType::Vertex,
            path: PathBuf::from("assets/GPass.hlsl"),
            entry_point: "VSMain".to_string(),
            debug: true,
            defines: vec![],
        });

        let ps = shader_cache.get_shader_by_desc(rhi::ShaderDesc {
            ty: rhi::ShaderType::Pixel,
            path: PathBuf::from("assets/GPass.hlsl"),
            entry_point: "PSMain".to_string(),
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
                        semantic: dx::SemanticName::Normal(0),
                        format: dx::Format::Rgb32Float,
                        slot: 1,
                    },
                    rhi::InputElementDesc {
                        semantic: dx::SemanticName::TexCoord(0),
                        format: dx::Format::Rg32Float,
                        slot: 2,
                    },
                    rhi::InputElementDesc {
                        semantic: dx::SemanticName::Tangent(0),
                        format: dx::Format::Rgba32Float,
                        slot: 3,
                    },
                ],
                vs,
                line: false,
                depth: Some(rhi::DepthDesc {
                    op: rhi::DepthOp::LessEqual,
                    format: dx::Format::D24UnormS8Uint,
                    read_only: true,
                }),
                wireframe: false,
                signature: Some(rs),
                formats: vec![
                    dx::Format::Rgba32Float,
                    dx::Format::Rgba32Float,
                    dx::Format::Rgba32Float,
                ],
                shaders: vec![ps],
                depth_bias: 0,
                slope_bias: 0.0,
                cull_mode: rhi::CullMode::Back,
            },
            &shader_cache,
        );

        Self {
            width,
            height,
            diffuse,
            diffuse_rtv,
            diffuse_srv,
            normal,
            normal_rtv,
            normal_srv,
            material,
            material_rtv,
            material_srv,
            accum,
            accum_rtv,
            accum_srv,
            vs,
            ps,
            pso,
        }
    }

    pub fn render(
        &self,
        device: &Arc<rhi::Device>,
        gpu_mesh: &GpuMesh,
        placeholder: &TexturePlaceholders,
        pso_cache: &rhi::RasterPipelineCache,
        camera_buffer: &rhi::DeviceBuffer,
        frame_idx: usize,
        depth: &ZPass,
    ) {
        let list = device.gfx_queue.get_command_buffer(&device);

        list.set_device_texture_barriers(&[
            (&self.diffuse, dx::ResourceStates::RenderTarget),
            (&self.normal, dx::ResourceStates::RenderTarget),
            (&self.material, dx::ResourceStates::RenderTarget),
            (&self.accum, dx::ResourceStates::RenderTarget),
            (&depth.depth, dx::ResourceStates::DepthRead),
        ]);

        list.clear_render_target(&self.diffuse_rtv, 0.301, 0.5607, 0.675);
        list.clear_render_target(&self.normal_rtv, 0.0, 0.0, 0.0);
        list.clear_render_target(&self.material_rtv, 0.0, 0.0, 0.0);
        list.clear_render_target(&self.accum_rtv, 0.0, 0.0, 0.0);

        list.set_render_targets(
            &[&self.diffuse_rtv, &self.normal_rtv, &self.material_rtv],
            Some(&depth.depth_dsv),
        );
        list.set_viewport(self.width, self.height);
        list.set_graphics_pipeline(pso_cache.get_pso(&self.pso));
        list.set_topology(rhi::GeomTopology::Triangles);

        list.set_graphics_cbv(&camera_buffer.cbv[frame_idx], 0);

        list.set_vertex_buffers(&[
            &gpu_mesh.pos_vb,
            &gpu_mesh.normal_vb,
            &gpu_mesh.uv_vb,
            &gpu_mesh.tangent_vb,
        ]);
        list.set_index_buffer(&gpu_mesh.ib);

        for submesh in &gpu_mesh.sub_meshes {
            list.set_graphics_cbv(
                &gpu_mesh
                    .gpu_materials
                    .get_buffer(device.id)
                    .expect("Not found device")
                    .cbv[submesh.material_idx],
                1,
            );

            if let Some(map) = gpu_mesh.materials[submesh.material_idx].diffuse_map {
                list.set_graphics_srv(
                    &gpu_mesh.image_views[map]
                        .get_view(device.id)
                        .expect("Not found texture view"),
                    2,
                );
            } else {
                list.set_graphics_srv(&placeholder.diffuse_placeholder_view, 2);
            }

            if let Some(map) = gpu_mesh.materials[submesh.material_idx].normal_map {
                list.set_graphics_srv(
                    &gpu_mesh.image_views[map]
                        .get_view(device.id)
                        .expect("Not found texture view"),
                    3,
                );
            } else {
                list.set_graphics_srv(&placeholder.normal_placeholder_view, 3);
            }

            list.draw_indexed(
                submesh.index_count,
                submesh.start_index_location,
                submesh.base_vertex_location as i32,
            );
        }

        device.gfx_queue.stash_cmd_buffer(list);
    }
}
