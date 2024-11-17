use glam::{Mat4, Vec4Swizzles};

use crate::utils::MatrixExt;

#[derive(Clone, Debug)]
#[repr(C)]
#[repr(align(256))]
pub struct GpuCamera {
    pub world: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

#[derive(Clone, Debug)]
pub struct Camera {
    pub view: Mat4,
    pub far: f32,
    pub near: f32,
    pub fov: f32,
    pub aspect_ratio: f32,
}

impl Camera {
    pub fn proj(&self) -> Mat4 {
        Mat4::perspective_lh(self.fov, self.aspect_ratio, self.near, self.far)
    }
}

pub struct FpsController {
    sensivity: f32,
    speed: f32,
    yaw: f32,
    pitch: f32,

    position: glam::Vec3,
}

impl FpsController {
    pub fn new(sensivity: f32, speed: f32) -> Self {
        Self {
            sensivity,
            speed,
            yaw: 0.0,
            pitch: 0.0,
            position: glam::Vec3::ZERO,
        }
    }

    pub fn update_position(&mut self, dt: f32, camera: &mut Camera, direction: glam::Vec3) {
        let rot_mat = glam::Mat4::from_euler(glam::EulerRot::YXZ, self.yaw, self.pitch, 0.0);

        let dir = direction.normalize();

        let direction =
            rot_mat.forward() * dir.x + glam::Vec3::Y * dir.y + rot_mat.x_axis.xyz() * dir.z;

        let direction = if direction.length() != 0.0 {
            direction.normalize()
        } else {
            direction
        };

        self.position += direction * self.speed * dt;

        camera.view = glam::Mat4::look_at_lh(
            self.position,
            self.position + rot_mat.forward(),
            rot_mat.up(),
        );
    }

    pub fn update_yaw_pitch(&mut self, camera: &mut Camera, x: f32, y: f32) {
        self.yaw += x * 0.003 * self.sensivity;
        self.pitch -= y * 0.003 * self.sensivity;

        let rot_mat = glam::Mat4::from_euler(glam::EulerRot::YXZ, self.yaw, self.pitch, 0.0);

        camera.view = glam::Mat4::look_at_lh(
            self.position,
            self.position + rot_mat.forward(),
            rot_mat.up(),
        );
    }
}
