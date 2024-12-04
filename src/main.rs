mod camera;
mod csm_pass;
mod dir_light_pass;
mod gamma_correction_pass;
mod gbuffer_pass;
mod gltf;
mod rhi;
mod shadow_mask_pass;
mod timer;
mod utils;

use std::{cell::RefCell, collections::HashMap, num::NonZero, sync::Arc, time::Duration};

use camera::{Camera, FpsController};
use csm_pass::CascadedShadowMapsPass;
use dir_light_pass::DirectionalLightPass;
use gamma_correction_pass::GammaCorrectionPass;
use gbuffer_pass::GbufferPass;
use glam::{vec2, vec3, Mat4, Vec2, Vec3};
use gltf::{GpuMesh, GpuMeshBuilder, Mesh};
use oxidx::dx;
use rhi::FRAMES_IN_FLIGHT;
use shadow_mask_pass::ShadowMaskPass;
use timer::GameTimer;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::Window,
};

pub struct WindowContext {
    pub window: Window,
    pub hwnd: NonZero<isize>,
    pub swapchain: rhi::Swapchain,
}

#[derive(Clone, Debug)]
#[repr(C)]
#[repr(align(256))]
pub struct GpuGlobals {
    pub view: Mat4,
    pub proj: Mat4,
    pub proj_view: Mat4,
    pub inv_view: Mat4,
    pub inv_proj: Mat4,
    pub inv_proj_view: Mat4,

    pub eye_pos: Vec3,
    pub _pad0: f32,

    pub screen_dim: Vec2,
    pub _pad1: Vec2,
}

#[derive(Clone, Debug)]
#[repr(C)]
#[repr(align(256))]
pub struct GpuMaterial {
    pub diffuse: [f32; 4],
    pub fresnel_r0: f32,
    pub roughness: f32,
}

#[derive(Clone, Debug)]
#[repr(C)]
#[repr(align(256))]
pub struct GpuTransform {
    pub world: Mat4,
}

pub struct Application {
    pub device_manager: rhi::DeviceManager,
    pub shader_cache: rhi::ShaderCache,

    pub pso_cache: rhi::RasterPipelineCache,
    pub device: Arc<rhi::Device>,
    pub placeholder: TexturePlaceholders,

    pub keys: HashMap<PhysicalKey, bool>,

    pub camera_buffers: rhi::Buffer,
    pub curr_frame: usize,

    pub wnd_ctx: Option<WindowContext>,
    pub wnd_title: String,
    pub timer: GameTimer,
    pub app_paused: bool,

    pub camera: Camera,
    pub camera_controller: FpsController,

    pub gpu_mesh: GpuMesh,

    pub window_width: u32,
    pub window_height: u32,

    pub gbuffer: GbufferPass,
    pub csm: CascadedShadowMapsPass,
    pub shadow_mask: ShadowMaskPass,
    pub dir_light_pass: DirectionalLightPass,
    pub gamma_correction_pass: GammaCorrectionPass,
}

pub struct TexturePlaceholders {
    pub diffuse_placeholder: rhi::DeviceTexture,
    pub diffuse_placeholder_view: rhi::DeviceTextureView,

    pub normal_placeholder: rhi::DeviceTexture,
    pub normal_placeholder_view: rhi::DeviceTextureView,
}

impl TexturePlaceholders {
    pub fn new(device: &Arc<rhi::Device>) -> Self {
        let cmd = device.gfx_queue.get_command_buffer(&device);

        let diffuse_placeholder = rhi::DeviceTexture::new(
            &device,
            1,
            1,
            1,
            dx::Format::Rgba8Unorm,
            1,
            dx::ResourceFlags::empty(),
            dx::ResourceStates::CopyDest,
            None,
            "Texture",
        );

        let total_size = diffuse_placeholder.get_size(&device, None);

        let staging = rhi::DeviceBuffer::new(
            &device,
            total_size,
            0,
            rhi::BufferType::Copy,
            false,
            "Staging Buffer",
            std::any::TypeId::of::<u8>(),
        );

        cmd.load_device_texture_from_memory(&diffuse_placeholder, &staging, &[255, 255, 255, 255]);
        cmd.set_device_texture_barrier(
            &diffuse_placeholder,
            dx::ResourceStates::PixelShaderResource,
            None,
        );

        let normal_placeholder = rhi::DeviceTexture::new(
            &device,
            1,
            1,
            1,
            dx::Format::Rgba8Unorm,
            1,
            dx::ResourceFlags::empty(),
            dx::ResourceStates::CopyDest,
            None,
            "Texture",
        );

        let total_size = normal_placeholder.get_size(&device, None);

        let staging = rhi::DeviceBuffer::new(
            &device,
            total_size,
            0,
            rhi::BufferType::Copy,
            false,
            "Staging Buffer",
            std::any::TypeId::of::<u8>(),
        );

        cmd.load_device_texture_from_memory(&normal_placeholder, &staging, &[127, 127, 255, 255]);
        cmd.set_device_texture_barrier(
            &normal_placeholder,
            dx::ResourceStates::PixelShaderResource,
            None,
        );

        device.gfx_queue.push_cmd_buffer(cmd);
        device.gfx_queue.wait_on_cpu(device.gfx_queue.execute());

        let diffuse_placeholder_view = rhi::DeviceTextureView::new(
            &device,
            &diffuse_placeholder,
            diffuse_placeholder.format,
            rhi::TextureViewType::ShaderResource,
            None,
        );

        let normal_placeholder_view = rhi::DeviceTextureView::new(
            &device,
            &normal_placeholder,
            normal_placeholder.format,
            rhi::TextureViewType::ShaderResource,
            None,
        );

        Self {
            diffuse_placeholder,
            diffuse_placeholder_view,
            normal_placeholder,
            normal_placeholder_view,
        }
    }
}

impl Application {
    pub fn new(width: u32, height: u32) -> Self {
        let mut device_manager = rhi::DeviceManager::new(true);
        let device = device_manager
            .get_high_perf_device()
            .expect("Failed to fetch high perf gpu");

        let mesh = Mesh::load("./assets/fantasy_island/scene.gltf");

        let gpu_mesh = GpuMesh::new(GpuMeshBuilder {
            mesh,
            devices: &[&device],
            normal_vb: device.id,
            uv_vb: device.id,
            tangent_vb: device.id,
            materials: device.id,
        });

        let mut shader_cache = rhi::ShaderCache::default();
        let mut pso_cache = rhi::RasterPipelineCache::new(&device);
        let placeholder = TexturePlaceholders::new(&device);

        let gbuffer = GbufferPass::new(width, height, &device, &mut shader_cache, &mut pso_cache);
        let csm =
            CascadedShadowMapsPass::new(&device, 2 * 1024, 0.5, &mut shader_cache, &mut pso_cache);
        let shadow_mask =
            ShadowMaskPass::new(&device, width, height, &mut shader_cache, &mut pso_cache);
        let dir_light_pass = DirectionalLightPass::new(&device, &mut shader_cache, &mut pso_cache);
        let gamma_correction_pass =
            GammaCorrectionPass::new(&device, &mut shader_cache, &mut pso_cache);

        let camera = Camera {
            view: glam::Mat4::IDENTITY,
            far: 500.0,
            near: 0.1,
            fov: 90.0f32.to_radians(),
            aspect_ratio: width as f32 / height as f32,
        };

        let controller = FpsController::new(0.003, 100.0);

        let camera_buffers =
            rhi::Buffer::constant::<GpuGlobals>(FRAMES_IN_FLIGHT, "Camera Buffers", &[&device]);

        Self {
            device_manager,
            shader_cache,
            pso_cache,
            placeholder,

            device,
            camera_buffers,
            wnd_ctx: None,
            wnd_title: "Multi-GPU Shadows Sample".to_string(),
            timer: GameTimer::default(),
            app_paused: false,
            curr_frame: 0,
            camera,
            camera_controller: controller,
            window_width: width,
            window_height: height,
            keys: HashMap::new(),
            gpu_mesh,

            csm,
            gbuffer,
            shadow_mask,
            dir_light_pass,
            gamma_correction_pass,
        }
    }

    pub fn update(&mut self) {
        let mut direction = glam::Vec3::ZERO;

        if self
            .keys
            .get(&PhysicalKey::Code(KeyCode::KeyW))
            .is_some_and(|v| *v)
        {
            direction.z += 1.0;
        }

        if self
            .keys
            .get(&PhysicalKey::Code(KeyCode::KeyS))
            .is_some_and(|v| *v)
        {
            direction.z -= 1.0;
        }

        if self
            .keys
            .get(&PhysicalKey::Code(KeyCode::KeyD))
            .is_some_and(|v| *v)
        {
            direction.x += 1.0;
        }

        if self
            .keys
            .get(&PhysicalKey::Code(KeyCode::KeyA))
            .is_some_and(|v| *v)
        {
            direction.x -= 1.0;
        }

        if direction.length() != 0.0 {
            self.camera_controller.update_position(
                self.timer.delta_time(),
                &mut self.camera,
                direction.normalize(),
            );
        }

        self.camera_buffers.write(
            self.curr_frame,
            &GpuGlobals {
                view: self.camera.view,
                proj: self.camera.proj(),
                eye_pos: self.camera_controller.position,
                proj_view: self.camera.proj() * self.camera.view,
                inv_view: self.camera.view.inverse(),
                inv_proj: self.camera.proj().inverse(),
                inv_proj_view: (self.camera.proj() * self.camera.view).inverse(),
                screen_dim: vec2(self.window_width as f32, self.window_height as f32),

                _pad0: 0.0,
                _pad1: Vec2::ZERO,
            },
        );

        self.csm
            .update(&self.camera, vec3(-1.0, -1.0, -1.0), self.curr_frame);
    }

    pub fn render(&mut self) {
        let Some(ctx) = &mut self.wnd_ctx else {
            return;
        };

        let list = self.device.gfx_queue.get_command_buffer(&self.device);
        let (_, texture, view, sync_point) = &mut ctx.swapchain.resources[self.curr_frame];

        list.begin(&self.device);

        list.set_device_texture_barrier(texture, dx::ResourceStates::RenderTarget, None);

        self.device.gfx_queue.stash_cmd_buffer(list);

        self.csm.render(
            &self.device,
            &self.gpu_mesh,
            &self.pso_cache,
            self.curr_frame,
        );

        self.gbuffer.render(
            &self.device,
            &self.gpu_mesh,
            &self.placeholder,
            &self.pso_cache,
            &self.camera_buffers,
            self.curr_frame,
        );

        self.shadow_mask.render(
            &self.device,
            &self.camera_buffers,
            &self.pso_cache,
            self.curr_frame,
            (&self.gbuffer.depth, &self.gbuffer.depth_srv),
            &self.csm,
        );

        self.dir_light_pass.render(
            &self.device,
            &self.camera_buffers,
            &self.pso_cache,
            self.curr_frame,
            &self.gbuffer,
            &self.shadow_mask,
        );

        self.gamma_correction_pass
            .render(&self.device, &self.pso_cache, &self.gbuffer, &view);

        let list = self.device.gfx_queue.get_command_buffer(&self.device);
        list.set_device_texture_barrier(texture, dx::ResourceStates::Present, None);

        self.device.gfx_queue.push_cmd_buffer(list);
        *sync_point = self.device.gfx_queue.execute();
        ctx.swapchain.present(false);
        self.curr_frame = (self.curr_frame + 1) % FRAMES_IN_FLIGHT;
        self.device
            .gfx_queue
            .wait_on_cpu(ctx.swapchain.resources[self.curr_frame].3);
    }

    pub fn bind_window(&mut self, window: Window) {
        let Ok(RawWindowHandle::Win32(hwnd)) = window.window_handle().map(|h| h.as_raw()) else {
            unreachable!()
        };
        let hwnd = hwnd.hwnd;

        let swapchain = rhi::Swapchain::new(
            &self.device_manager.factory,
            &self.device,
            self.window_width,
            self.window_height,
            hwnd,
        );

        self.wnd_ctx = Some(WindowContext {
            window,
            hwnd,
            swapchain,
        });
    }

    fn calculate_frame_stats(&self) {
        thread_local! {
            static FRAME_COUNT: RefCell<i32> = Default::default();
            static TIME_ELAPSED: RefCell<f32> = Default::default();
        }

        FRAME_COUNT.with_borrow_mut(|frame_cnt| {
            *frame_cnt += 1;
        });

        TIME_ELAPSED.with_borrow_mut(|time_elapsed| {
            if self.timer.total_time() - *time_elapsed > 1.0 {
                FRAME_COUNT.with_borrow_mut(|frame_count| {
                    let fps = *frame_count as f32;
                    let mspf = 1000.0 / fps;

                    if let Some(ref context) = self.wnd_ctx {
                        context
                            .window
                            .set_title(&format!("{} Fps: {fps} Ms: {mspf}", self.wnd_title))
                    }

                    *frame_count = 0;
                    *time_elapsed += 1.0;
                });
            }
        })
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        self.device.gfx_queue.wait_idle();
        self.device.compute_queue.wait_idle();
        self.device.copy_queue.wait_idle();
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title(&self.wnd_title)
            .with_inner_size(PhysicalSize::new(self.window_width, self.window_height));

        let window = event_loop.create_window(window_attributes).unwrap();
        window
            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
            .expect("Failet to lock cursor");
        window.set_cursor_visible(false);
        self.bind_window(window);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::Focused(focused) => {
                if focused {
                    self.app_paused = false;
                    self.timer.start();
                } else {
                    self.app_paused = true;
                    self.timer.stop();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => match event.state {
                winit::event::ElementState::Pressed => {
                    if event.physical_key == KeyCode::Escape {
                        event_loop.exit();
                    }
                    self.keys.insert(event.physical_key, true);
                }
                winit::event::ElementState::Released => {
                    self.keys.insert(event.physical_key, false);
                }
            },
            WindowEvent::MouseInput { state, .. } => match state {
                winit::event::ElementState::Pressed =>
                    /*self.sample.on_mouse_down(button)*/
                    {}
                winit::event::ElementState::Released =>
                    /*self.sample.on_mouse_up(button)*/
                    {}
            },
            WindowEvent::Resized(size) => {
                let Some(ref mut context) = self.wnd_ctx else {
                    return;
                };

                if context.window.is_minimized().is_some_and(|minized| minized) {
                    self.app_paused = true;
                } else {
                    self.app_paused = false;

                    self.window_width = size.width;
                    self.window_height = size.height;

                    self.device.gfx_queue.wait_idle();
                    context.swapchain.resize(
                        &self.device,
                        size.width,
                        size.height,
                        self.device.gfx_queue.fence.get_current_value(),
                    );
                    self.curr_frame = 0;

                    self.gbuffer = GbufferPass::new(
                        size.width,
                        size.height,
                        &self.device,
                        &mut self.shader_cache,
                        &mut self.pso_cache,
                    );

                    self.shadow_mask = ShadowMaskPass::new(
                        &self.device,
                        size.width,
                        size.height,
                        &mut self.shader_cache,
                        &mut self.pso_cache,
                    );
                }
            }
            WindowEvent::RedrawRequested => {
                self.timer.tick();
                self.calculate_frame_stats();

                if self.app_paused {
                    std::thread::sleep(Duration::from_millis(100));
                    return;
                }
                self.update();
                self.render();
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            _ => (),
        }
    }

    #[allow(clippy::single_match)]
    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.camera_controller.update_yaw_pitch(
                    &mut self.camera,
                    delta.0 as f32,
                    delta.1 as f32,
                );
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(context) = self.wnd_ctx.as_ref() {
            context.window.request_redraw();
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = Application::new(1280, 720);
    event_loop.run_app(&mut app).unwrap();
}
