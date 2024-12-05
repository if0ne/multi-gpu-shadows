use std::{path::PathBuf, rc::Rc, sync::Arc};

use oxidx::dx;

use crate::{gltf::GpuMesh, rhi};

#[derive(Debug)]
pub struct ZPass {
    pub width: u32,
    pub height: u32,
    pub depth: rhi::DeviceTexture,
    pub depth_dsv: rhi::DeviceTextureView,
    pub depth_srv: rhi::DeviceTextureView,
    pub pso: rhi::PipelineHandle,
}

impl ZPass {
    pub fn new(
        device: &Arc<rhi::Device>,
        width: u32,
        height: u32,
        shader_cache: &mut rhi::ShaderCache,
        pso_cache: &mut rhi::RasterPipelineCache,
    ) -> Self {
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
                entries: vec![rhi::BindingEntry::Cbv { num: 1, slot: 0 }],
                static_samplers: vec![],
                bindless: false,
            },
        ));

        let vs = shader_cache.get_shader_by_desc(rhi::ShaderDesc {
            ty: rhi::ShaderType::Vertex,
            path: PathBuf::from("assets/Zpass.hlsl"),
            entry_point: "Main".to_string(),
            debug: false,
            defines: vec![],
        });

        let pso = pso_cache.get_pso_by_desc(
            rhi::RasterPipelineDesc {
                input_elements: vec![rhi::InputElementDesc {
                    semantic: dx::SemanticName::Position(0),
                    format: dx::Format::Rgb32Float,
                    slot: 0,
                }],
                vs: vs,
                line: false,
                depth: Some(rhi::DepthDesc {
                    op: rhi::DepthOp::LessEqual,
                    format: dx::Format::D24UnormS8Uint,
                    read_only: false,
                }),
                wireframe: false,
                signature: Some(rs),
                formats: vec![],
                shaders: vec![],
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
            depth,
            depth_dsv,
            depth_srv,
        }
    }

    pub fn render(
        &self,
        device: &Arc<rhi::Device>,
        camera_buffer: &rhi::Buffer,
        pso_cache: &rhi::RasterPipelineCache,
        frame_idx: usize,
        gpu_mesh: &GpuMesh,
    ) {
        let list = device.gfx_queue.get_command_buffer(&device);
        list.set_mark("Z pre-pass");

        list.set_viewport(self.width, self.height);
        list.set_graphics_pipeline(pso_cache.get_pso(&self.pso));
        list.set_topology(rhi::GeomTopology::Triangles);
        list.set_device_texture_barriers(&[(&self.depth, dx::ResourceStates::DepthWrite)]);

        list.clear_depth_target(&self.depth_dsv);
        list.set_render_targets(&[], Some(&self.depth_srv));
        list.set_graphics_cbv(
            &camera_buffer
                .get_buffer(list.device_id)
                .expect("Failed to get buffer")
                .cbv[frame_idx],
            0,
        );

        list.set_vertex_buffers(&[&gpu_mesh.pos_vb]);
        list.set_index_buffer(&gpu_mesh.ib);

        for submesh in &gpu_mesh.sub_meshes {
            list.draw_indexed(
                submesh.index_count,
                submesh.start_index_location,
                submesh.base_vertex_location as i32,
            );
        }

        device.gfx_queue.stash_cmd_buffer(list);
    }
}
