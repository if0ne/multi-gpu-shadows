use std::{num::NonZero, rc::Rc};

use gltf::Model;
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
    pub curr_frame: usize,

    pub wnd_ctx: Option<WindowContext>,

    pub model: Model,
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

        let cmd_lists = std::array::from_fn(|_| {
            rhi::CommandBuffer::new(&device, dx::CommandListType::Direct, true)
        });

        let fence_values = std::array::from_fn(|_| 0);

        Self {
            device,
            tracker,
            cmd_queue,
            fence,
            cmd_lists,
            fence_values,
            wnd_ctx: None,
            model,
            curr_frame: 0,
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
