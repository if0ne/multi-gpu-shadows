use std::{any::TypeId, sync::Arc};

use glam::{vec4, Vec4, Vec4Swizzles};
use oxidx::dx;

use crate::{camera::Camera, rhi};

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
pub struct CascadedShadowMaps {
    pub size: u32,

    pub cascade_proj_views: [glam::Mat4; 4],
    pub distances: [f32; 4],
    pub lamda: f32,

    pub gpu_csm_buffer: rhi::DeviceBuffer,
    pub gpu_csm_proj_view_buffer: rhi::DeviceBuffer,
    pub texture: rhi::DeviceTexture,
    pub dsvs: [rhi::DeviceTextureView; 4],
    pub srv: rhi::DeviceTextureView,
}

impl CascadedShadowMaps {
    pub fn new(device: &Arc<rhi::Device>, size: u32, lamda: f32) -> Self {
        let texture = rhi::DeviceTexture::new(
            device,
            size,
            size,
            4,
            dx::Format::D32Float,
            1,
            dx::ResourceFlags::AllowDepthStencil,
            dx::ResourceStates::DepthWrite,
            Some(dx::ClearValue::depth(dx::Format::D32Float, 1.0, 0)),
            "CSM",
        );

        let mut gpu_csm_buffer = rhi::DeviceBuffer::new(
            &device,
            size_of::<GpuCSM>(),
            0,
            rhi::BufferType::Constant,
            false,
            "CSM Buffer",
            TypeId::of::<GpuCSM>(),
        );
        gpu_csm_buffer.build_constant(device, 1, size_of::<GpuCSM>());

        let mut gpu_csm_proj_view_buffer = rhi::DeviceBuffer::new(
            &device,
            4 * size_of::<GpuCSMProjView>(),
            0,
            rhi::BufferType::Constant,
            false,
            "CSM Proj View Buffer",
            TypeId::of::<GpuCSMProjView>(),
        );
        gpu_csm_proj_view_buffer.build_constant(device, 4, size_of::<GpuCSMProjView>());

        let dsvs = std::array::from_fn(|i| {
            let i = i as u32;
            rhi::DeviceTextureView::new_in_array(
                device,
                &texture,
                dx::Format::D32Float,
                rhi::TextureViewType::DepthTarget,
                i..(i + 1),
            )
        });

        let srv = rhi::DeviceTextureView::new_in_array(
            device,
            &texture,
            dx::Format::R32Float,
            rhi::TextureViewType::ShaderResource,
            0..4,
        );

        Self {
            cascade_proj_views: [glam::Mat4::IDENTITY; 4],
            distances: [0.0; 4],
            lamda,
            size,
            gpu_csm_buffer,
            gpu_csm_proj_view_buffer,
            texture,
            dsvs,
            srv,
        }
    }

    pub fn update(&mut self, camera: &Camera, light_dir: glam::Vec3) {
        let cascade_count = self.distances.len();

        for (i, distance) in self.distances.iter_mut().enumerate() {
            let ratio = ((i + 1) as f32) / (cascade_count as f32);
            let clog = camera.near * camera.far.powf(ratio);
            let cuni = camera.near + (camera.far - camera.near) * ratio;
            *distance = self.lamda * clog + (1.0 - self.lamda) * cuni;
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
            0,
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

        self.gpu_csm_proj_view_buffer.write_all(
            &self
                .cascade_proj_views
                .map(|pv| GpuCSMProjView { proj_vies: pv }),
        );
    }
}
