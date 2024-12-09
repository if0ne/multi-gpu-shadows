use std::{path::PathBuf, rc::Rc, sync::Arc};

use glam::{vec4, Vec4, Vec4Swizzles};
use oxidx::dx;

use crate::{
    camera::Camera,
    gltf::GpuMesh,
    rhi::{self, FRAMES_IN_FLIGHT},
};

#[derive(Clone, Debug)]
#[repr(C)]
#[repr(align(256))]
pub struct GpuCSM {
    pub proj_vies: [glam::Mat4; 4],
    pub distances: Vec4,
}

#[derive(Clone, Debug)]
#[repr(C)]
#[repr(align(256))]
pub struct GpuCSMProjView {
    pub proj_vies: glam::Mat4,
}

#[derive(Debug)]
pub struct CascadedShadowMapsPass {
    pub size: u32,

    pub cascade_proj_views: [glam::Mat4; 4],
    pub distances: [f32; 4],
    pub lamda: f32,

    pub gpu_csm_buffer: rhi::DeviceBuffer,
    pub gpu_csm_proj_view_buffer: rhi::DeviceBuffer,
    pub texture: rhi::DeviceTexture,
    pub dsv: rhi::DeviceTextureView,
    pub srv: rhi::DeviceTextureView,

    pub pso: rhi::PipelineHandle,
}

impl CascadedShadowMapsPass {
    pub fn new(
        device: &Arc<rhi::Device>,
        size: u32,
        lamda: f32,
        shader_cache: &mut rhi::ShaderCache,
        pso_cache: &mut rhi::RasterPipelineCache,
    ) -> Self {
        let texture = rhi::DeviceTexture::new(
            device,
            2 * size,
            2 * size,
            1,
            dx::Format::D32Float,
            1,
            dx::ResourceFlags::AllowDepthStencil,
            dx::ResourceStates::DepthWrite,
            Some(dx::ClearValue::depth(dx::Format::D32Float, 1.0, 0)),
            "CSM",
        );

        let gpu_csm_buffer =
            rhi::DeviceBuffer::constant::<GpuCSM>(&device, FRAMES_IN_FLIGHT, "CSM Buffer");

        let gpu_csm_proj_view_buffer = rhi::DeviceBuffer::constant::<GpuCSMProjView>(
            &device,
            FRAMES_IN_FLIGHT * 4,
            "CSM Proj View Buffer",
        );

        let dsv = rhi::DeviceTextureView::new(
            device,
            &texture,
            dx::Format::D32Float,
            rhi::TextureViewType::DepthTarget,
            None,
        );

        let srv = rhi::DeviceTextureView::new(
            device,
            &texture,
            dx::Format::R32Float,
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
            path: PathBuf::from("assets/Csm.hlsl"),
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
                    format: dx::Format::D32Float,
                    read_only: false,
                }),
                wireframe: false,
                signature: Some(rs),
                formats: vec![],
                shaders: vec![],
                depth_bias: 10000,
                slope_bias: 5.0,
                cull_mode: rhi::CullMode::Back,
            },
            &shader_cache,
        );

        Self {
            cascade_proj_views: [glam::Mat4::IDENTITY; 4],
            distances: [0.0; 4],
            lamda,
            size,
            gpu_csm_buffer,
            gpu_csm_proj_view_buffer,
            texture,
            dsv,
            srv,
            pso,
        }
    }

    pub fn update(&mut self, camera: &Camera, light_dir: glam::Vec3, frame_index: usize) {
        let cascade_count = self.distances.len();

        let near_clip = camera.near;
        let far_clip = camera.far;
        let clip_range = far_clip - near_clip;

        let min_z = near_clip;
        let max_z = near_clip + clip_range;

        let range = max_z - min_z;
        let ratio: f32 = max_z / min_z;

        for (i, distance) in self.distances.iter_mut().enumerate() {
            let p = (i as f32 + 1.0) / cascade_count as f32;
            let log = min_z * ratio.powf(p);
            let uniform = min_z + range * p;
            *distance = self.lamda * (log - uniform) + uniform;
        }

        let mut cur_near = camera.near;

        for i in 0..cascade_count {
            let cur_far = self.distances[i];

            let mut corners = [
                glam::vec3(-1.0, -1.0, 0.0),
                glam::vec3(-1.0, -1.0, 1.0),
                glam::vec3(-1.0, 1.0, 0.0),
                glam::vec3(-1.0, 1.0, 1.0),
                glam::vec3(1.0, -1.0, 0.0),
                glam::vec3(1.0, -1.0, 1.0),
                glam::vec3(1.0, 1.0, 0.0),
                glam::vec3(1.0, 1.0, 1.0),
            ];

            let frust_proj =
                glam::Mat4::perspective_lh(camera.fov, camera.aspect_ratio, cur_near, cur_far);
            let cam_view = camera.view;

            let frust_proj_view = (frust_proj * cam_view).inverse();

            for corner in corners.iter_mut() {
                let temp = frust_proj_view * glam::vec4(corner.x, corner.y, corner.z, 1.0);
                let temp = temp / temp.w;

                *corner = temp.xyz();
            }

            let center = corners
                .into_iter()
                .fold(glam::Vec3::ZERO, |center, corner| center + corner)
                / 8.0;

            let light_view = glam::Mat4::look_at_lh(center, center + light_dir, glam::Vec3::Y);

            let mut min_x = f32::MAX;
            let mut max_x = f32::MIN;
            let mut min_y = f32::MAX;
            let mut max_y = f32::MIN;
            let mut min_z = f32::MAX;
            let mut max_z = f32::MIN;

            for corner in corners {
                let temp = light_view * glam::vec4(corner.x, corner.y, corner.z, 1.0);

                min_x = min_x.min(temp.x);
                max_x = max_x.max(temp.x);
                min_y = min_y.min(temp.y);
                max_y = max_y.max(temp.y);
                min_z = min_z.min(temp.z);
                max_z = max_z.max(temp.z);
            }

            let light_proj = glam::Mat4::orthographic_lh(min_x, max_x, min_y, max_y, min_z, max_z);

            self.cascade_proj_views[i] = light_proj * light_view;

            cur_near = cur_far;
        }

        self.gpu_csm_buffer.write(
            frame_index,
            GpuCSM {
                proj_vies: self.cascade_proj_views,
                distances: vec4(
                    self.distances[0],
                    self.distances[1],
                    self.distances[2],
                    self.distances[3],
                ),
            },
        );

        self.cascade_proj_views
            .iter()
            .enumerate()
            .for_each(|(i, pv)| {
                self.gpu_csm_proj_view_buffer.write(
                    4 * frame_index + i,
                    GpuCSMProjView { proj_vies: *pv },
                );
            });
    }

    pub fn render(
        &self,
        device: &Arc<rhi::Device>,
        gpu_mesh: &GpuMesh,
        pso_cache: &rhi::RasterPipelineCache,
        frame_idx: usize,
    ) {
        let list = device.gfx_queue.get_command_buffer(&device);

        list.set_viewport(self.size, self.size);
        list.set_graphics_pipeline(pso_cache.get_pso(&self.pso));
        list.set_topology(rhi::GeomTopology::Triangles);
        list.set_device_texture_barrier(&self.texture, dx::ResourceStates::DepthWrite, None);
        list.clear_depth_target(&self.dsv);
        list.set_render_targets(&[], Some(&self.dsv));
        for i in 0..4 {
            let row = i / 2;
            let col = i % 2;
            list.set_viewport_with_offset(self.size, self.size, self.size * col, self.size * row);

            list.set_graphics_cbv(
                &self.gpu_csm_proj_view_buffer.cbv[4 * frame_idx + i as usize],
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
        }

        device.gfx_queue.stash_cmd_buffer(list);
    }
}
