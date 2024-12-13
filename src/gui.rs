use std::{path::PathBuf, rc::Rc, sync::Arc};

use glam::{Vec2, Vec4};
use oxidx::dx;
use winit::window::Window;

use crate::rhi::{self, FRAMES_IN_FLIGHT};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Vertex {
    pos: Vec2,
    uv: Vec2,
    color: Vec4
}

struct MeshData {
    vtx: Vec<Vertex>,
    idx: Vec<u16>,
    tex: egui::TextureId,
    clip_rect: egui::Rect,
}

pub struct Gui {
    pub egui_ctx: egui::Context,
    pub egui_winit: egui_winit::State,

    pipeline: rhi::PipelineHandle,

    shapes: Vec<egui::epaint::ClippedShape>,

    vbs: [rhi::DeviceBuffer; FRAMES_IN_FLIGHT],
    ibs: [rhi::DeviceBuffer; FRAMES_IN_FLIGHT],

    staging_vbs: [rhi::DeviceBuffer; FRAMES_IN_FLIGHT],
    staging_ibs: [rhi::DeviceBuffer; FRAMES_IN_FLIGHT],
}

impl Gui {
    pub fn new(
        window: &Window,
        device: &Arc<rhi::Device>,
        pipeline_cache: &mut rhi::RasterPipelineCache,
        shader_cache: &mut rhi::ShaderCache,
    ) -> Gui {
        let egui_ctx: egui::Context = Default::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            window,
            None,
            None,
            None,
        );

        let vs = shader_cache.get_shader_by_desc(rhi::ShaderDesc {
            ty: rhi::ShaderType::Vertex,
            path: PathBuf::from("assets/Gui.hlsl"),
            entry_point: "VSMain".to_string(),
            debug: false,
            defines: vec![],
        });

        let ps = shader_cache.get_shader_by_desc(rhi::ShaderDesc {
            ty: rhi::ShaderType::Pixel,
            path: PathBuf::from("assets/Gui.hlsl"),
            entry_point: "PSMain".to_string(),
            debug: false,
            defines: vec![],
        });

        let rs = Rc::new(rhi::RootSignature::new(
            device,
            rhi::RootSignatureDesc {
                entries: vec![
                    rhi::BindingEntry::Cbv { num: 1, slot: 0 },
                    rhi::BindingEntry::Srv { num: 1, slot: 0 },
                ],
                static_samplers: vec![rhi::StaticSampler {
                    slot: 0,
                    filter: dx::Filter::Point,
                    address_mode: dx::AddressMode::Clamp,
                    comp_func: dx::ComparisonFunc::None,
                    b_color: dx::BorderColor::OpaqueBlack,
                }],
                bindless: false,
            },
        ));

        let pipeline = pipeline_cache.get_pso_by_desc(rhi::RasterPipelineDesc {
            input_elements: vec![
                rhi::InputElementDesc { 
                    semantic: dx::SemanticName::Position(0), 
                    format: dx::Format::Rg32Float, 
                    slot: 0
                },
                rhi::InputElementDesc { 
                    semantic: dx::SemanticName::TexCoord(0), 
                    format: dx::Format::Rg32Float, 
                    slot: 0
                },
                rhi::InputElementDesc { 
                    semantic: dx::SemanticName::Color(0), 
                    format: dx::Format::Rgba32Float, 
                    slot: 0
                }
            ],
            line: false,
            wireframe: false,
            depth_bias: 0,
            slope_bias: 0.0,
            depth: None,
            signature: Some(rs),
            formats: vec![dx::Format::Rgba8Unorm],
            vs,
            shaders: vec![ps],
            cull_mode: rhi::CullMode::None,
        }, shader_cache);

        let vbs = std::array::from_fn(|_| {
            rhi::DeviceBuffer::vertex::<Vertex>(device, 1, "GUI Vertex")
        });

        let ibs = std::array::from_fn(|_| {
            rhi::DeviceBuffer::index_u16(device, 1, "GUI Index")
        });

        let staging_vbs = std::array::from_fn(|_| {
            rhi::DeviceBuffer::copy::<Vertex>(device, 1, "GUI Staging Vertex")
        });

        let staging_ibs = std::array::from_fn(|_| {
            rhi::DeviceBuffer::copy::<u16>(device, 1, "GUI Staging Index")
        });

        Gui {
            egui_ctx,
            egui_winit,
            pipeline,
            vbs,
            ibs,
            staging_vbs,
            staging_ibs,
            shapes: vec![]
        }
    }

    fn pixels_per_point(&self, window: &Window) -> f32 {
        egui_winit::pixels_per_point(&self.egui_ctx, window)
    }

    pub fn update(&mut self, window: &Window, winit_event: &winit::event::WindowEvent) -> bool {
        self.egui_winit
            .on_window_event(window, winit_event)
            .consumed
    }

    pub fn immediate_ui(&mut self, window: &Window, layout_function: impl FnOnce(&mut Self)) {
        let raw_input = self.egui_winit.take_egui_input(window);
        self.egui_ctx.begin_pass(raw_input);
        layout_function(self);
    }

    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.egui_winit.take_egui_input(window);
        self.egui_ctx.begin_pass(raw_input);
    }

    pub fn render(
        &mut self,
        window: &Window,
        device: &Arc<rhi::Device>,
        pipeline_cache: &rhi::RasterPipelineCache,
        view: &rhi::DeviceTextureView,
        width: u32,
        height: u32,
        frame: usize,
    ) {
        let clipped = self.extract_draw_data_at_frame_end(window);

        let list = device.gfx_queue.get_command_buffer(&device);
        list.set_graphics_pipeline(pipeline_cache.get_pso(&self.pipeline));
        list.set_viewport_only(width, height);
        list.set_render_targets(&[view], None);

        for clip in clipped {
            Self::draw_mesh();
        }

        device.gfx_queue.stash_cmd_buffer(list);
    }

    fn draw_mesh(
        
    ) {
       
    }

    fn extract_draw_data_at_frame_end(&mut self, window: &Window) -> Vec<egui::ClippedPrimitive> {
        self.end_frame(window);
        self.egui_ctx
            .tessellate(std::mem::take(&mut self.shapes), self.pixels_per_point(window))
    }

    fn end_frame(&mut self, window: &Window) {
        let egui::FullOutput {
            platform_output,
            textures_delta: _,
            shapes,
            pixels_per_point: _,
            viewport_output: _,
        } = self.egui_ctx.end_pass();

        self.egui_winit
            .handle_platform_output(window, platform_output);
        self.shapes = shapes;
    }

    pub fn context(&self) -> egui::Context {
        self.egui_ctx.clone()
    }
}
