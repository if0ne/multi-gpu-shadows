mod camera;
mod csm_pass;
mod dir_light_pass;
mod gamma_correction_pass;
mod gbuffer_pass;
mod gltf;
mod m_shadow_mask_pass;
mod rhi;
mod shadow_mask_pass;
mod timer;
mod utils;
mod zpass;

use std::{
    cell::RefCell,
    collections::HashMap,
    num::NonZero,
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use camera::{Camera, FpsController};
use csm_pass::CascadedShadowMapsPass;
use dir_light_pass::DirectionalLightPass;
use gamma_correction_pass::GammaCorrectionPass;
use gbuffer_pass::GbufferPass;
use glam::{vec2, vec3, Mat4, Vec2, Vec3};
use gltf::{GpuMesh, GpuMeshBuilder, Mesh};
use m_shadow_mask_pass::{MgpuShadowMaskPass, MgpuState};
use oxidx::dx;
use rhi::FRAMES_IN_FLIGHT;
use timer::GameTimer;
use tracing::info;
use tracing_subscriber::{
    fmt::{self, writer::MakeWriterExt},
    layer::SubscriberExt,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::Window,
};
use zpass::ZPass;

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

    pub primary_gpu: Arc<rhi::Device>,
    pub primary_pso_cache: rhi::RasterPipelineCache,
    pub primary_placeholder: TexturePlaceholders,

    pub secondary_gpu: Arc<rhi::Device>,
    pub secondary_pso_cache: rhi::RasterPipelineCache,

    pub keys: HashMap<PhysicalKey, bool>,

    pub camera_buffers: rhi::DeviceBuffer,
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

    pub p_zpass: ZPass,
    pub p_gbuffer: GbufferPass,
    pub p_dir_light_pass: DirectionalLightPass,
    pub p_gamma_correction_pass: GammaCorrectionPass,

    pub s_zpass: ZPass,
    pub s_csm: CascadedShadowMapsPass,

    pub mgpu_shadow_mask: MgpuShadowMaskPass,
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
        let primary_gpu = device_manager
            .get_high_perf_device()
            .expect("Failed to fetch high perf gpu");

        let secondary_gpu = device_manager
            .get_high_perf_device()
            .unwrap_or_else(|| device_manager.get_warp());

        let mesh = Mesh::load("./assets/fantasy_island/scene.gltf");

        let gpu_mesh = GpuMesh::new(GpuMeshBuilder {
            mesh,
            devices: &[&primary_gpu, &secondary_gpu],
            normal_vb: primary_gpu.id,
            uv_vb: primary_gpu.id,
            tangent_vb: primary_gpu.id,
            materials: primary_gpu.id,
        });

        let mut shader_cache = rhi::ShaderCache::default();

        let mut primary_pso_cache = rhi::RasterPipelineCache::new(&primary_gpu);
        let primary_placeholder = TexturePlaceholders::new(&primary_gpu);

        let mut secondary_pso_cache = rhi::RasterPipelineCache::new(&secondary_gpu);

        let p_zpass = ZPass::new(
            &primary_gpu,
            width,
            height,
            &mut shader_cache,
            &mut primary_pso_cache,
        );
        let p_gbuffer = GbufferPass::new(
            width,
            height,
            &primary_gpu,
            &mut shader_cache,
            &mut primary_pso_cache,
        );
        let p_dir_light_pass =
            DirectionalLightPass::new(&primary_gpu, &mut shader_cache, &mut primary_pso_cache);
        let p_gamma_correction_pass =
            GammaCorrectionPass::new(&primary_gpu, &mut shader_cache, &mut primary_pso_cache);

        let s_zpass = ZPass::new(
            &secondary_gpu,
            width,
            height,
            &mut shader_cache,
            &mut secondary_pso_cache,
        );
        let s_csm = CascadedShadowMapsPass::new(
            &secondary_gpu,
            2 * 1024,
            0.5,
            &mut shader_cache,
            &mut secondary_pso_cache,
        );

        let mgpu_shadow_mask = MgpuShadowMaskPass::new(
            &primary_gpu,
            &secondary_gpu,
            width,
            height,
            &mut shader_cache,
            &mut secondary_pso_cache,
        );

        let camera = Camera {
            view: glam::Mat4::IDENTITY,
            far: 500.0,
            near: 0.1,
            fov: 90.0f32.to_radians(),
            aspect_ratio: width as f32 / height as f32,
        };

        let controller = FpsController::new(0.003, 100.0);

        let camera_buffers = rhi::DeviceBuffer::constant::<GpuGlobals>(
            &primary_gpu,
            FRAMES_IN_FLIGHT,
            "Camera Buffers",
        );

        Self {
            device_manager,
            shader_cache,

            primary_pso_cache,
            primary_placeholder,

            secondary_gpu,
            secondary_pso_cache,

            primary_gpu,
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

            p_zpass,
            p_gbuffer,
            p_dir_light_pass,
            p_gamma_correction_pass,

            s_zpass,
            s_csm,

            mgpu_shadow_mask,
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
            GpuGlobals {
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
    }

    pub fn render(&mut self) {
        let Some(ctx) = &mut self.wnd_ctx else {
            return;
        };

        info!("Beginning rendering");

        let copy_texture = self.mgpu_shadow_mask.copy_texture.load(Ordering::Acquire);
        let working_texture = self
            .mgpu_shadow_mask
            .working_texture
            .load(Ordering::Acquire);

        if self.secondary_gpu.gfx_queue.is_ready()
            && self.mgpu_shadow_mask.states[working_texture] == MgpuState::WaitForWrite
        {
            info!("Recording to secondary gpu");
            self.s_csm
                .update(&self.camera, vec3(-1.0, -1.0, -1.0), working_texture);

            self.mgpu_shadow_mask.camera_buffers.write(
                working_texture,
                GpuGlobals {
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

            let list = self
                .secondary_gpu
                .gfx_queue
                .get_command_buffer(&self.secondary_gpu);

            list.begin(&self.secondary_gpu);
            self.secondary_gpu.gfx_queue.stash_cmd_buffer(list);

            info!("Zpass");
            self.s_zpass.render(
                &self.secondary_gpu,
                &self.mgpu_shadow_mask.camera_buffers,
                &self.secondary_pso_cache,
                self.curr_frame,
                &self.gpu_mesh,
            );

            info!("Csm");
            self.s_csm.render(
                &self.secondary_gpu,
                &self.gpu_mesh,
                &self.secondary_pso_cache,
                self.curr_frame,
            );

            info!("ShadowMask");
            self.mgpu_shadow_mask.render(
                &self.secondary_gpu,
                &self.secondary_pso_cache,
                &self.s_zpass,
                &self.s_csm,
            );

            let list = self
                .secondary_gpu
                .gfx_queue
                .get_command_buffer(&self.secondary_gpu);

            info!("Try copy");

            match &self.mgpu_shadow_mask.sender[working_texture].state {
                rhi::SharedTextureState::CrossAdapter { .. } => {
                    // noop
                }
                rhi::SharedTextureState::Binded { cross, local } => {
                    list.set_device_texture_barriers(&[
                        (cross, dx::ResourceStates::Common),
                        (local, dx::ResourceStates::Common),
                    ]);
                    list.copy_texture_to_texture(local, cross);
                }
            };

            info!("Copied");

            self.secondary_gpu.gfx_queue.push_cmd_buffer(list);
            self.secondary_gpu.gfx_queue.execute();
            self.mgpu_shadow_mask.states[working_texture] = MgpuState::WaitForCopy(
                self.secondary_gpu
                    .gfx_queue
                    .signal_shared(&self.mgpu_shadow_mask.sender_fence),
            );
        }

        dbg!(working_texture);
        dbg!(copy_texture);
        dbg!(self.mgpu_shadow_mask.recv_fence.get_completed_value());
        dbg!(self.mgpu_shadow_mask.states[copy_texture]);

        let list = self
            .primary_gpu
            .gfx_queue
            .get_command_buffer(&self.primary_gpu);
        let (_, texture, view, sync_point) = &mut ctx.swapchain.resources[self.curr_frame];

        list.begin(&self.primary_gpu);

        list.set_device_texture_barrier(texture, dx::ResourceStates::RenderTarget, None);

        self.primary_gpu.gfx_queue.stash_cmd_buffer(list);

        info!("Primary ZPass");
        self.primary_gpu.gfx_queue.set_mark("Primary Z Prepass");
        self.p_zpass.render(
            &self.primary_gpu,
            &self.camera_buffers,
            &self.primary_pso_cache,
            self.curr_frame,
            &self.gpu_mesh,
        );
        self.primary_gpu.gfx_queue.end_mark();

        info!("Primary G Pass");
        self.primary_gpu.gfx_queue.set_mark("G Pass");
        self.p_gbuffer.render(
            &self.primary_gpu,
            &self.gpu_mesh,
            &self.primary_placeholder,
            &self.primary_pso_cache,
            &self.camera_buffers,
            self.curr_frame,
            &self.p_zpass,
        );
        self.primary_gpu.gfx_queue.end_mark();

        if let MgpuState::WaitForCopy(v) = self.mgpu_shadow_mask.states[copy_texture] {
            if self.primary_gpu.copy_queue.is_ready()
                && self.mgpu_shadow_mask.recv_fence.get_completed_value() >= v
            {
                self.mgpu_shadow_mask.next_working_texture();

                info!("Copy shadow mask to primary gpu");
                let list = self
                    .primary_gpu
                    .copy_queue
                    .get_command_buffer(&self.primary_gpu);

                match &self.mgpu_shadow_mask.recv[copy_texture].state {
                    rhi::SharedTextureState::CrossAdapter { .. } => {
                        // noop
                    }
                    rhi::SharedTextureState::Binded { cross, local } => {
                        list.set_device_texture_barriers(&[
                            (cross, dx::ResourceStates::CopySource),
                            (local, dx::ResourceStates::CopyDest),
                        ]);
                        list.copy_texture_to_texture(local, cross);
                    }
                };

                self.primary_gpu.copy_queue.push_cmd_buffer(list);
                self.mgpu_shadow_mask.states[copy_texture] =
                    MgpuState::WaitForRead(self.primary_gpu.copy_queue.execute());
            }
        }

        info!("Primary Direction Light Pass");
        self.primary_gpu.gfx_queue.set_mark("Direction Light Pass");
        let (texture_mask, view_mask) =
            if let MgpuState::WaitForRead(v) = self.mgpu_shadow_mask.states[copy_texture] {
                if self.primary_gpu.copy_queue.is_complete(v) {
                    dbg!("Get copy texture");
                    self.mgpu_shadow_mask.next_copy_texture();
                    self.mgpu_shadow_mask.states[copy_texture] = MgpuState::WaitForWrite;
                    (
                        &self.mgpu_shadow_mask.recv[copy_texture],
                        &self.mgpu_shadow_mask.recv_srv[copy_texture],
                    )
                } else {
                    let copy_texture = if copy_texture == 0 {
                        FRAMES_IN_FLIGHT - 1
                    } else {
                        copy_texture - 1
                    };

                    (
                        &self.mgpu_shadow_mask.recv[copy_texture],
                        &self.mgpu_shadow_mask.recv_srv[copy_texture],
                    )
                }
            } else {
                let copy_texture = if copy_texture == 0 {
                    FRAMES_IN_FLIGHT - 1
                } else {
                    copy_texture - 1
                };

                (
                    &self.mgpu_shadow_mask.recv[copy_texture],
                    &self.mgpu_shadow_mask.recv_srv[copy_texture],
                )
            };

        self.p_dir_light_pass.render(
            &self.primary_gpu,
            &self.camera_buffers,
            &self.primary_pso_cache,
            self.curr_frame,
            &self.p_gbuffer,
            (texture_mask.local_resource(), view_mask),
        );
        self.primary_gpu.gfx_queue.end_mark();

        info!("Primary Gamma Correction Pass");
        self.primary_gpu.gfx_queue.set_mark("Gamma Correction Pass");
        self.p_gamma_correction_pass.render(
            &self.primary_gpu,
            &self.primary_pso_cache,
            &self.p_gbuffer,
            &view,
        );
        self.primary_gpu.gfx_queue.end_mark();

        let list = self
            .primary_gpu
            .gfx_queue
            .get_command_buffer(&self.primary_gpu);
        list.set_device_texture_barrier(texture, dx::ResourceStates::Present, None);

        self.primary_gpu.gfx_queue.push_cmd_buffer(list);
        *sync_point = self.primary_gpu.gfx_queue.execute();
        ctx.swapchain.present(false);
        self.curr_frame = (self.curr_frame + 1) % FRAMES_IN_FLIGHT;
        self.primary_gpu
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
            &self.primary_gpu,
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
        self.primary_gpu.gfx_queue.wait_idle();
        self.primary_gpu.compute_queue.wait_idle();
        self.primary_gpu.copy_queue.wait_idle();

        self.secondary_gpu.gfx_queue.wait_idle();
        self.secondary_gpu.compute_queue.wait_idle();
        self.secondary_gpu.copy_queue.wait_idle();
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

                    self.primary_gpu.gfx_queue.wait_idle();
                    self.secondary_gpu.gfx_queue.wait_idle();
                    self.primary_gpu.copy_queue.wait_idle();

                    context.swapchain.resize(
                        &self.primary_gpu,
                        size.width,
                        size.height,
                        self.primary_gpu.gfx_queue.fence.get_current_value(),
                    );
                    self.curr_frame = 0;

                    self.p_zpass = ZPass::new(
                        &self.primary_gpu,
                        size.width,
                        size.height,
                        &mut self.shader_cache,
                        &mut self.primary_pso_cache,
                    );

                    self.s_zpass = ZPass::new(
                        &self.secondary_gpu,
                        size.width,
                        size.height,
                        &mut self.shader_cache,
                        &mut self.secondary_pso_cache,
                    );

                    self.p_gbuffer = GbufferPass::new(
                        size.width,
                        size.height,
                        &self.primary_gpu,
                        &mut self.shader_cache,
                        &mut self.primary_pso_cache,
                    );

                    self.mgpu_shadow_mask = MgpuShadowMaskPass::new(
                        &self.primary_gpu,
                        &self.secondary_gpu,
                        size.width,
                        size.height,
                        &mut self.shader_cache,
                        &mut self.secondary_pso_cache,
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
    let console_log = fmt::Layer::new().with_ansi(true).with_writer(
        std::io::stdout
            .with_min_level(tracing::Level::ERROR)
            .with_max_level(tracing::Level::INFO),
    );

    let subscriber = tracing_subscriber::registry().with(console_log);

    let _ = tracing::subscriber::set_global_default(subscriber);

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = Application::new(1280, 720);
    event_loop.run_app(&mut app).unwrap();
}
