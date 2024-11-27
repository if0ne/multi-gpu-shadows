use std::{collections::HashMap, num::NonZero, rc::Rc};

use camera::{Camera, FpsController, GpuCamera};
use glam::{vec2, vec3};
use gltf::Mesh;
use oxidx::dx;
use rhi::{DeviceSettings, FRAMES_IN_FLIGHT};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::Window,
};

mod camera;
mod gltf;
mod rhi;
mod utils;

pub struct WindowContext {
    pub window: Window,
    pub hwnd: NonZero<isize>,
    pub swapchain: rhi::Swapchain,
}

pub struct Application {
    pub device: Rc<rhi::Device>,

    pub keys: HashMap<PhysicalKey, bool>,

    pub cmd_queue: rhi::CommandQueue,

    pub camera_buffers: [rhi::Buffer; FRAMES_IN_FLIGHT],
    pub curr_frame: usize,

    pub wnd_ctx: Option<WindowContext>,

    pub camera: Camera,
    pub camera_controller: FpsController,

    pub pso: rhi::GraphicsPipeline,

    pub model: Mesh,
    pub vertex_buffer: rhi::Buffer,
    pub index_buffer: rhi::Buffer,

    pub window_width: u32,
    pub window_height: u32,
}

impl Application {
    pub fn new(width: u32, height: u32) -> Self {
        let device = Rc::new(
            rhi::Device::fetch(DeviceSettings {
                use_debug: true,
                ty: rhi::AdapterType::Hardware { high_perf: true },
                excluded_adapters: &[],
            })
            .expect("Failed to create device"),
        );

        let model = Mesh::load("./assets/pica_pica_-_mini_diorama_01/scene.gltf");
        let cmd_queue = rhi::CommandQueue::new(&device, dx::CommandListType::Direct);

        let mut vertex_staging = rhi::Buffer::new(
            &device,
            model.positions.len() * std::mem::size_of::<[f32; 3]>(),
            std::mem::size_of::<[f32; 3]>(),
            rhi::BufferType::Copy,
            false,
            format!("{} Vertex Buffer", "Check"),
        );

        {
            let map = vertex_staging.map::<[f32; 3]>(None);
            map.pointer.clone_from_slice(&model.positions);
        }

        let mut index_staging = rhi::Buffer::new(
            &device,
            model.indices.len() * std::mem::size_of::<u32>(),
            std::mem::size_of::<u32>(),
            rhi::BufferType::Copy,
            false,
            format!("{} Index Buffer", "check"),
        );

        {
            let map = index_staging.map::<u32>(None);
            map.pointer.clone_from_slice(&model.indices);
        }

        let vertex_buffer = rhi::Buffer::new(
            &device,
            model.positions.len() * size_of::<[f32; 3]>(),
            size_of::<[f32; 3]>(),
            rhi::BufferType::Vertex,
            false,
            "vertex",
        );

        let index_buffer = rhi::Buffer::new(
            &device,
            model.indices.len() * size_of::<u32>(),
            size_of::<u32>(),
            rhi::BufferType::Index,
            false,
            "index",
        );

        let cmd_list = cmd_queue.get_command_buffer(&device);
        cmd_list.begin(&device);
        cmd_list.copy_buffer_to_buffer(&vertex_buffer, &vertex_staging);
        cmd_list.copy_buffer_to_buffer(&index_buffer, &index_staging);
        cmd_queue.push_cmd_buffer(cmd_list);
        let v = cmd_queue.execute();
        cmd_queue.wait_on_cpu(v);

        let rs = Rc::new(rhi::RootSignature::new(
            &device,
            &[rhi::BindingEntry::Cbv],
            false,
        ));

        let vs = rhi::CompiledShader::compile("assets/vert.hlsl", rhi::ShaderType::Vertex);
        let ps = rhi::CompiledShader::compile("assets/pixel.hlsl", rhi::ShaderType::Pixel);

        let pso = rhi::GraphicsPipeline::new(
            &device,
            &rhi::PipelineDesc {
                line: false,
                depth: false,
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

        let camera = Camera {
            view: glam::Mat4::IDENTITY,
            far: 1000.0,
            near: 0.1,
            fov: 90.0f32.to_radians(),
            aspect_ratio: width as f32 / height as f32,
        };

        let controller = FpsController::new(0.003, 1.0);

        let camera_buffers = std::array::from_fn(|_| {
            let mut buffer = rhi::Buffer::new(
                &device,
                size_of::<GpuCamera>(),
                0,
                rhi::BufferType::Constant,
                false,
                "Camera buffer",
            );
            buffer.build_constant(&device);

            buffer
        });

        Self {
            device,
            cmd_queue,
            camera_buffers,
            wnd_ctx: None,
            curr_frame: 0,
            camera,
            camera_controller: controller,
            pso,
            window_width: width,
            window_height: height,
            keys: HashMap::new(),

            model,
            vertex_buffer,
            index_buffer,
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
            self.camera_controller
                .update_position(0.16, &mut self.camera, direction.normalize());
        }

        let buffer = &mut self.camera_buffers[self.curr_frame];
        let mapped = buffer.map::<GpuCamera>(None);

        mapped.pointer[0] = GpuCamera {
            world: glam::Mat4::from_translation(vec3(0.0, 0.0, 1.0)),
            view: self.camera.view,
            proj: self.camera.proj(),
        };
    }

    pub fn render(&mut self) {
        let Some(ctx) = &mut self.wnd_ctx else {
            return;
        };

        let list = self.cmd_queue.get_command_buffer(&self.device);
        let (_, texture, view, sync_point) = &mut ctx.swapchain.resources[self.curr_frame];

        list.begin(&self.device);

        list.set_image_barrier(texture, dx::ResourceStates::RenderTarget, None);
        list.clear_render_target(view, 0.0, 0.0, 0.0);

        list.set_render_targets(&[view], None);
        list.set_viewport(self.window_width, self.window_height);
        list.set_graphics_pipeline(&self.pso);
        list.set_topology(rhi::GeomTopology::Triangles);

        list.set_graphics_cbv(&self.camera_buffers[self.curr_frame], 0);

        list.set_vertex_buffer(&self.vertex_buffer);
        list.set_index_buffer(&self.index_buffer);
        list.draw(self.model.indices.len() as u32);

        list.set_image_barrier(texture, dx::ResourceStates::Present, None);

        self.cmd_queue.push_cmd_buffer(list);
        *sync_point = self.cmd_queue.execute();
        ctx.swapchain.present(true);
        self.curr_frame = (self.curr_frame + 1) % FRAMES_IN_FLIGHT;
        self.cmd_queue
            .wait_on_cpu(ctx.swapchain.resources[self.curr_frame].3);
    }

    pub fn bind_window(&mut self, window: Window) {
        let Ok(RawWindowHandle::Win32(hwnd)) = window.window_handle().map(|h| h.as_raw()) else {
            unreachable!()
        };
        let hwnd = hwnd.hwnd;

        let swapchain = rhi::Swapchain::new(
            &self.device,
            &self.cmd_queue,
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
}

impl Drop for Application {
    fn drop(&mut self) {
        self.cmd_queue.wait_idle();
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("Mgpu Sample")
            .with_inner_size(PhysicalSize::new(self.window_width, self.window_height));

        let window = event_loop.create_window(window_attributes).unwrap();
        self.bind_window(window);
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::Focused(_focused) => {
                /*if focused {
                    self.base.app_paused = false;
                    self.base.timer.start();
                } else {
                    self.base.app_paused = true;
                    self.base.timer.stop();
                }*/
            }
            WindowEvent::KeyboardInput { event, .. } => match event.state {
                winit::event::ElementState::Pressed => {
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
                self.window_width = size.width;
                self.window_height = size.height;

                self.cmd_queue.wait_idle();
                context.swapchain.resize(
                    &self.device,
                    size.width,
                    size.height,
                    self.cmd_queue.fence.get_current_value(),
                );
                self.curr_frame = 0;

                /*let Some(ref mut context) = self.base.context else {
                    return;
                };

                if context.window.is_minimized().is_some_and(|minized| minized) {
                    self.base.app_paused = true;
                } else {
                    self.base.app_paused = false;
                    self.base.on_resize(size.width, size.height);
                    self.sample
                        .on_resize(&mut self.base, size.width, size.height);
                }*/
            }
            WindowEvent::RedrawRequested => {
                /*if self.base.app_paused {
                    sleep(Duration::from_millis(100));
                    return;
                }
                self.base.calculate_frame_stats();
                self.sample.update(&self.base);
                self.sample.render(&mut self.base);*/
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
