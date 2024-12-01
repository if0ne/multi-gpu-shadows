mod camera;
mod gbuffer;
mod gltf;
mod rhi;
mod timer;
mod utils;

use std::{cell::RefCell, collections::HashMap, num::NonZero, rc::Rc, sync::Arc, time::Duration};

use camera::{Camera, FpsController};
use gbuffer::Gbuffer;
use glam::{vec3, vec4, Mat4, Vec3, Vec4};
use gltf::{GpuMesh, GpuMeshBuilder, Mesh};
use oxidx::dx;
use rhi::{DeviceManager, FRAMES_IN_FLIGHT};
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
    pub eye_pos: Vec3,
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

#[derive(Clone, Debug)]
#[repr(C)]
#[repr(align(256))]
pub struct GpuDirectionalLight {
    pub strength: Vec3,
    pub _pad: f32,
    pub direction: Vec3,
}

#[derive(Clone, Debug)]
#[repr(C)]
#[repr(align(256))]
pub struct GpuAmbientLight {
    pub color: Vec4,
}

pub struct Application {
    pub device_manager: DeviceManager,

    pub device: Arc<rhi::Device>,

    pub keys: HashMap<PhysicalKey, bool>,

    pub camera_buffers: rhi::Buffer,
    pub dir_light_buffer: rhi::Buffer,
    pub ambient_light_buffer: rhi::Buffer,
    pub curr_frame: usize,

    pub wnd_ctx: Option<WindowContext>,
    pub wnd_title: String,
    pub timer: GameTimer,
    pub app_paused: bool,

    pub camera: Camera,
    pub camera_controller: FpsController,

    pub pso: rhi::GraphicsPipeline,

    pub gpu_mesh: GpuMesh,

    pub window_width: u32,
    pub window_height: u32,

    pub gbuffer: Gbuffer,
    pub diffuse_placeholder: rhi::DeviceTexture,
    pub diffuse_placeholder_view: rhi::TextureView,
}

impl Application {
    pub fn new(width: u32, height: u32) -> Self {
        let mut device_manager = rhi::DeviceManager::new(true);
        let device = device_manager
            .get_high_perf_device()
            .expect("Failed to fetch high perf gpu");

        let mesh = Mesh::load("./assets/pica_pica_-_mini_diorama_01/scene.gltf");

        let gpu_mesh = GpuMesh::new(GpuMeshBuilder {
            mesh,
            devices: &[&device],
            normal_vb: device.id,
            uv_vb: device.id,
            tangent_vb: device.id,
            materials: device.id,
        });

        let diffuse_placeholder = rhi::DeviceTexture::new(
            &device,
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
        let cmd = device.gfx_queue.get_command_buffer(&device);

        cmd.load_device_texture_from_memory(&diffuse_placeholder, &staging, &[255, 255, 255, 255]);
        cmd.set_device_texture_barrier(
            &diffuse_placeholder,
            dx::ResourceStates::PixelShaderResource,
            None,
        );
        device.gfx_queue.push_cmd_buffer(cmd);
        device.gfx_queue.wait_on_cpu(device.gfx_queue.execute());
        let diffuse_placeholder_view = rhi::TextureView::new(
            &device,
            &diffuse_placeholder,
            rhi::TextureViewType::ShaderResource,
            None,
        );

        let rs = Rc::new(rhi::RootSignature::new(
            &device,
            &[
                rhi::BindingEntry::Cbv,
                rhi::BindingEntry::Cbv,
                rhi::BindingEntry::Srv,
                rhi::BindingEntry::Cbv,
                rhi::BindingEntry::Cbv,
            ],
            false,
        ));

        let vs = rhi::CompiledShader::compile("assets/vert.hlsl", rhi::ShaderType::Vertex);
        let ps = rhi::CompiledShader::compile("assets/pixel.hlsl", rhi::ShaderType::Pixel);

        let pso = rhi::GraphicsPipeline::new(
            &device,
            &rhi::PipelineDesc {
                line: false,
                depth: true,
                depth_format: dx::Format::D24UnormS8Uint,
                op: rhi::DepthOp::Less,
                wireframe: false,
                signature: Some(rs),
                formats: vec![dx::Format::Rgba8Unorm],
                shaders: HashMap::from_iter([
                    (rhi::ShaderType::Vertex, vs),
                    (rhi::ShaderType::Pixel, ps),
                ]),
            },
        );

        let gbuffer = Gbuffer::new(&device, width, height);

        let camera = Camera {
            view: glam::Mat4::IDENTITY,
            far: 1000.0,
            near: 0.1,
            fov: 90.0f32.to_radians(),
            aspect_ratio: width as f32 / height as f32,
        };

        let controller = FpsController::new(0.003, 100.0);

        let camera_buffers =
            rhi::Buffer::constant::<GpuGlobals>(FRAMES_IN_FLIGHT, "Camera Buffers", &[&device]);

        let mut dir_light_buffer = rhi::Buffer::constant::<GpuDirectionalLight>(
            1,
            "Directional Light Buffers",
            &[&device],
        );

        dir_light_buffer.write(
            0,
            &GpuDirectionalLight {
                strength: vec3(1.0, 0.81, 0.16),
                direction: vec3(-1.0, -1.0, -1.0),

                _pad: 0.0,
            },
        );

        let mut ambient_light_buffer =
            rhi::Buffer::constant::<GpuAmbientLight>(1, "Ambient Light Buffers", &[&device]);

        ambient_light_buffer.write(
            0,
            &GpuAmbientLight {
                color: vec4(0.3, 0.3, 0.63, 1.0),
            },
        );

        Self {
            device_manager,
            device,
            camera_buffers,
            dir_light_buffer,
            ambient_light_buffer,
            wnd_ctx: None,
            wnd_title: "Multi-GPU Shadows Sample".to_string(),
            timer: GameTimer::default(),
            app_paused: false,
            curr_frame: 0,
            camera,
            camera_controller: controller,
            pso,
            window_width: width,
            window_height: height,
            keys: HashMap::new(),
            gbuffer,
            gpu_mesh,
            diffuse_placeholder,
            diffuse_placeholder_view,
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
            },
        );
    }

    pub fn render(&mut self) {
        let Some(ctx) = &mut self.wnd_ctx else {
            return;
        };

        let list = self.device.gfx_queue.get_command_buffer(&self.device);
        let (_, texture, view, sync_point) = &mut ctx.swapchain.resources[self.curr_frame];

        list.begin(&self.device);

        list.set_device_texture_barrier(texture, dx::ResourceStates::RenderTarget, None);
        list.clear_render_target(view, 0.301, 0.5607, 0.675);
        list.clear_depth_target(&self.gbuffer.depth_dsv);

        list.set_render_targets(&[view], Some(&self.gbuffer.depth_dsv));
        list.set_viewport(self.window_width, self.window_height);
        list.set_graphics_pipeline(&self.pso);
        list.set_topology(rhi::GeomTopology::Triangles);

        list.set_graphics_cbv(
            &self
                .camera_buffers
                .get_buffer(self.device.id)
                .expect("Not found device")
                .cbv[self.curr_frame],
            0,
        );

        list.set_graphics_cbv(
            &self
                .dir_light_buffer
                .get_buffer(self.device.id)
                .expect("Not found device")
                .cbv[0],
            3,
        );

        list.set_graphics_cbv(
            &self
                .ambient_light_buffer
                .get_buffer(self.device.id)
                .expect("Not found device")
                .cbv[0],
            4,
        );

        list.set_vertex_buffers(&[
            &self.gpu_mesh.pos_vb,
            &self.gpu_mesh.normal_vb,
            &self.gpu_mesh.uv_vb,
        ]);
        list.set_index_buffer(&self.gpu_mesh.ib);

        for submesh in &self.gpu_mesh.sub_meshes {
            list.set_graphics_cbv(
                &self
                    .gpu_mesh
                    .gpu_materials
                    .get_buffer(self.device.id)
                    .expect("Not found device")
                    .cbv[submesh.material_idx],
                1,
            );

            if let Some(map) = self.gpu_mesh.materials[submesh.material_idx].diffuse_map {
                list.set_graphics_srv(&self.gpu_mesh.image_views[map], 2);
            } else {
                list.set_graphics_srv(&self.diffuse_placeholder_view, 2);
            }

            list.draw(
                submesh.index_count,
                submesh.start_index_location,
                submesh.base_vertex_location as i32,
            );
        }
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

                    self.gbuffer = Gbuffer::new(&self.device, size.width, size.height);
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
