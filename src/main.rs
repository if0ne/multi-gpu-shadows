use std::{collections::HashMap, num::NonZero, rc::Rc};

use camera::{Camera, FpsController, GpuCamera};
use glam::{vec2, vec3};
use gltf::{Model, Node, Vertex};
use oxidx::dx;
use rhi::FRAMES_IN_FLIGHT;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
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
    pub tracker: rhi::GpuResourceTracker,

    pub cmd_queue: rhi::CommandQueue,
    pub fence: rhi::Fence,
    pub cmd_lists: [rhi::CommandBuffer; FRAMES_IN_FLIGHT],
    pub fence_values: [u64; FRAMES_IN_FLIGHT],
    pub camera_buffers: [rhi::Buffer; FRAMES_IN_FLIGHT],
    pub curr_frame: usize,

    pub wnd_ctx: Option<WindowContext>,

    pub model: Model,

    pub camera: Camera,
    pub camera_controller: FpsController,

    pub pso: rhi::GraphicsPipeline,

    pub vertex_buffer: rhi::Buffer,
    pub vertex_staging: rhi::Buffer,

    pub index_buffer: rhi::Buffer,
    pub index_staging: rhi::Buffer,
}

impl Application {
    pub fn new() -> Self {
        let device = Rc::new(rhi::Device::new(true));

        let cmd_queue = rhi::CommandQueue::new(&device, dx::CommandListType::Direct);
        let fence = rhi::Fence::new(&device);

        let tracker = rhi::GpuResourceTracker::new(&device);

        let model = Model::load(
            &device,
            &tracker,
            &cmd_queue,
            &fence,
            "./assets/fantasy_island/scene.gltf",
        );

        let cmd_list = rhi::CommandBuffer::new(&device, dx::CommandListType::Direct, false);

        let vertices = vec![
            Vertex {
                pos: vec3(-0.5, -0.5, 0.0),
                normals: vec3(0.0, 0.0, -1.0),
                uv: vec2(0.0, 1.0),
            },
            Vertex {
                pos: vec3(0.0, 0.5, 0.0),
                normals: vec3(0.0, 0.0, -1.0),
                uv: vec2(0.0, 0.0),
            },
            Vertex {
                pos: vec3(0.5, -0.5, 0.0),
                normals: vec3(0.0, 0.0, -1.0),
                uv: vec2(1.0, 0.0),
            },
        ];

        let indices = vec![0u32, 1, 2];

        let mut vertex_staging = rhi::Buffer::new(
            &tracker,
            vertices.len() * std::mem::size_of::<Vertex>(),
            std::mem::size_of::<Vertex>(),
            rhi::BufferType::Copy,
            false,
            format!("{} Vertex Buffer", "Check"),
        );

        {
            let map = vertex_staging.map::<Vertex>(None);
            map.pointer.clone_from_slice(&vertices);
        }

        let mut index_staging = rhi::Buffer::new(
            &tracker,
            indices.len() * std::mem::size_of::<u32>(),
            std::mem::size_of::<u32>(),
            rhi::BufferType::Copy,
            false,
            format!("{} Index Buffer", "check"),
        );

        {
            let map = index_staging.map::<u32>(None);
            map.pointer.clone_from_slice(&indices);
        }

        let vertex_buffer = rhi::Buffer::new(
            &tracker,
            vertices.len() * std::mem::size_of::<Vertex>(),
            std::mem::size_of::<Vertex>(),
            rhi::BufferType::Vertex,
            false,
            format!("{} Vertex Buffer", "Check"),
        );

        let index_buffer = rhi::Buffer::new(
            &tracker,
            indices.len() * std::mem::size_of::<u32>(),
            std::mem::size_of::<u32>(),
            rhi::BufferType::Index,
            false,
            format!("{} Index Buffer", "Check"),
        );

        cmd_list.begin(&device, false);
        cmd_list.copy_buffer_to_buffer(&vertex_buffer, &vertex_staging);
        cmd_list.copy_buffer_to_buffer(&index_buffer, &index_staging);
        cmd_list.end();
        cmd_queue.submit(&[&cmd_list]);
        let v = cmd_queue.signal(&fence);
        cmd_queue.wait(&fence, v);

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

        let cmd_lists = std::array::from_fn(|_| {
            rhi::CommandBuffer::new(&device, dx::CommandListType::Direct, true)
        });

        let fence_values = std::array::from_fn(|_| v);

        let camera = Camera {
            view: glam::Mat4::IDENTITY,
            far: 1000.0,
            near: 0.1,
            fov: 45.0f32.to_radians(),
            aspect_ratio: 16.0 / 9.0,
        };

        let controller = FpsController::new(1.0, 1.0);

        let camera_buffers = std::array::from_fn(|_| {
            let mut buffer = rhi::Buffer::new(
                &tracker,
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
            tracker,
            cmd_queue,
            fence,
            cmd_lists,
            fence_values,
            camera_buffers,
            wnd_ctx: None,
            model,
            curr_frame: 0,
            camera,
            camera_controller: controller,
            pso,
            vertex_buffer,
            vertex_staging,
            index_buffer,
            index_staging,
        }
    }

    pub fn update(&mut self) {
        self.camera_controller
            .update_position(0.0, &mut self.camera, glam::Vec3::Z);

        let buffer = &mut self.camera_buffers[self.curr_frame];
        let mapped = buffer.map::<GpuCamera>(None);

        mapped.pointer[0] = GpuCamera {
            world: glam::Mat4::from_scale(vec3(0.5, 0.5, 0.5)),
            view: glam::Mat4::look_at_lh(vec3(0.0, 500.0, -500.0), glam::Vec3::ZERO, glam::Vec3::Y),
            proj: self.camera.proj(),
        };
    }

    fn draw_node(&self, cmd_list: &rhi::CommandBuffer, node: &Node) {
        for prim in &node.primitives {
            cmd_list.set_vertex_buffer(&prim.vertex_buffer);
            cmd_list.set_index_buffer(&prim.index_buffer);
            cmd_list.draw(prim.vtx_count);
        }

        for node in &node.children {
            self.draw_node(cmd_list, &*node.borrow());
        }
    }

    pub fn render(&mut self) {
        let Some(ctx) = &self.wnd_ctx else {
            return;
        };

        let list = &self.cmd_lists[self.curr_frame];
        let (_, texture, view) = &ctx.swapchain.resources[self.curr_frame];

        list.begin(&self.device, true);

        list.set_image_barrier(texture, dx::ResourceStates::RenderTarget, None);
        list.clear_render_target(view, 1.0, 0.0, 0.0);

        list.set_render_targets(&[&view], None);
        list.set_viewport(1280, 720);
        list.set_graphics_pipeline(&self.pso);
        list.set_topology(rhi::GeomTopology::Triangles);

        list.set_graphics_cbv(&self.camera_buffers[self.curr_frame], 0);

        self.draw_node(list, &*self.model.root.borrow());
        //list.set_vertex_buffer(&self.vertex_buffer);
        //list.set_index_buffer(&self.index_buffer);
        //list.draw(3);

        list.set_image_barrier(texture, dx::ResourceStates::Present, None);

        list.end();
        self.cmd_queue.submit(&[list]);
        self.fence_values[self.curr_frame] = self.cmd_queue.signal(&self.fence);
        ctx.swapchain.present(true);
        self.curr_frame = (self.curr_frame + 1) % FRAMES_IN_FLIGHT;
        self.cmd_queue
            .wait(&self.fence, self.fence_values[self.curr_frame]);
    }

    pub fn bind_window(&mut self, window: Window) {
        let Ok(RawWindowHandle::Win32(hwnd)) = window.window_handle().map(|h| h.as_raw()) else {
            unreachable!()
        };
        let hwnd = hwnd.hwnd;
        let size = window.inner_size();

        let swapchain =
            rhi::Swapchain::new(&self.device, &self.cmd_queue, size.width, size.height, hwnd);

        self.wnd_ctx = Some(WindowContext {
            window,
            hwnd,
            swapchain,
        });
    }
}

impl Drop for Application {
    fn drop(&mut self) {
        self.cmd_queue
            .wait(&self.fence, self.fence_values[self.curr_frame]);
    }
}

impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = Window::default_attributes()
            .with_title("Mgpu Sample")
            .with_inner_size(PhysicalSize::new(1280, 720));

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
            WindowEvent::Focused(focused) => {
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
                    /*if let PhysicalKey::Code(code) = event.physical_key {
                        self.sample.on_key_down(&self.base, code, event.repeat);
                    }*/
                }
                winit::event::ElementState::Released => {
                    /*if event.physical_key == PhysicalKey::Code(KeyCode::F2) {
                        self.base.set_msaa_4x_state(!self.base.msaa_state);
                    } else if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                        event_loop.exit()
                    }

                    if let PhysicalKey::Code(code) = event.physical_key {
                        self.sample.on_key_up(code);
                    }*/
                }
            },
            WindowEvent::MouseInput { state, button, .. } => match state {
                winit::event::ElementState::Pressed =>
                    /*self.sample.on_mouse_down(button)*/
                    {}
                winit::event::ElementState::Released =>
                    /*self.sample.on_mouse_up(button)*/
                    {}
            },
            WindowEvent::Resized(size) => {
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
            DeviceEvent::MouseMotion { delta } => { /* */ }
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

    let mut app = Application::new();
    event_loop.run_app(&mut app).unwrap();
}
