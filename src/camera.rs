use glam::Mat4;

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

    pub position: glam::Vec3,
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
        let forward = glam::Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        );
        let right = glam::Vec3::Y.cross(forward).normalize();
        let up = right.cross(forward);

        self.position +=
            (forward * direction.z + right * direction.x + up * direction.y) * self.speed * dt;

        camera.view = Mat4::look_at_lh(self.position, self.position + forward, glam::Vec3::Y);
    }

    pub fn update_yaw_pitch(&mut self, camera: &mut Camera, x: f32, y: f32) {
        self.yaw -= x * self.sensivity;
        self.pitch -= y * self.sensivity;

        self.pitch = self.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.1,
            std::f32::consts::FRAC_PI_2 - 0.1,
        );

        let forward = glam::Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        );

        camera.view = Mat4::look_at_lh(self.position, self.position + forward, glam::Vec3::Y);
    }
}
