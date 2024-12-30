use std::{path::PathBuf, rc::Rc, sync::Arc};

use glam::{vec3, vec4, Vec3, Vec4};
use oxidx::dx;

use crate::{gbuffer_pass::GbufferPass, rhi};

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

#[derive(Debug)]
pub struct DirectionalLightWithMaskPass {
    pub dir_light_buffer: rhi::DeviceBuffer,
    pub ambient_light_buffer: rhi::DeviceBuffer,
    pub pso: rhi::PipelineHandle,
}

impl DirectionalLightWithMaskPass {
    pub fn new(
        device: &Arc<rhi::Device>,
        shader_cache: &mut rhi::ShaderCache,
        pso_cache: &mut rhi::RasterPipelineCache,
    ) -> Self {
        let mut dir_light_buffer = rhi::DeviceBuffer::constant::<GpuDirectionalLight>(
            &device,
            1,
            "Directional Light Buffers",
        );

        dir_light_buffer.write(
            0,
            GpuDirectionalLight {
                strength: vec3(1.0, 0.81, 0.16),
                direction: vec3(-1.0, -1.0, -1.0),

                _pad: 0.0,
            },
        );

        let mut ambient_light_buffer =
            rhi::DeviceBuffer::constant::<GpuAmbientLight>(&device, 1, "Ambient Light Buffers");

        ambient_light_buffer.write(
            0,
            GpuAmbientLight {
                color: vec4(0.3, 0.3, 0.63, 1.0),
            },
        );

        let rs = Rc::new(rhi::RootSignature::new(
            &device,
            rhi::RootSignatureDesc {
                entries: vec![
                    rhi::BindingEntry::Cbv { num: 1, slot: 0 },
                    rhi::BindingEntry::Cbv { num: 1, slot: 1 },
                    rhi::BindingEntry::Cbv { num: 1, slot: 2 },
                    rhi::BindingEntry::Srv { num: 1, slot: 3 },
                    rhi::BindingEntry::Srv { num: 1, slot: 4 },
                    rhi::BindingEntry::Srv { num: 1, slot: 5 },
                    rhi::BindingEntry::Srv { num: 1, slot: 6 },
                ],
                static_samplers: vec![],
                bindless: false,
            },
        ));

        let vs = shader_cache.get_shader_by_desc(rhi::ShaderDesc {
            ty: rhi::ShaderType::Vertex,
            path: PathBuf::from("assets/FullscreenVS.hlsl"),
            entry_point: "Main".to_string(),
            debug: false,
            defines: vec![],
        });

        let ps = shader_cache.get_shader_by_desc(rhi::ShaderDesc {
            ty: rhi::ShaderType::Pixel,
            path: PathBuf::from("assets/DirectionalLightPassWithMask.hlsl"),
            entry_point: "Main".to_string(),
            debug: true,
            defines: vec![],
        });

        let pso = pso_cache.get_pso_by_desc(
            rhi::RasterPipelineDesc {
                input_elements: vec![
                    rhi::InputElementDesc {
                        semantic: dx::SemanticName::Position(0),
                        format: dx::Format::Rgb32Float,
                        slot: 0,
                    },
                    rhi::InputElementDesc {
                        semantic: dx::SemanticName::TexCoord(0),
                        format: dx::Format::Rg32Float,
                        slot: 1,
                    },
                ],
                vs: vs,
                line: false,
                depth: None,
                wireframe: false,
                signature: Some(rs),
                formats: vec![dx::Format::Rgba32Float],
                shaders: vec![ps],
                depth_bias: 0,
                slope_bias: 0.0,
                cull_mode: rhi::CullMode::None,
                depth_clip: false,
            },
            &shader_cache,
        );

        Self {
            pso,
            dir_light_buffer,
            ambient_light_buffer,
        }
    }

    pub fn render(
        &self,
        device: &Arc<rhi::Device>,
        camera_buffer: &rhi::DeviceBuffer,
        pso_cache: &rhi::RasterPipelineCache,
        frame_idx: usize,
        gbuffer: &GbufferPass,
        shadow_mask: (&rhi::DeviceTexture, &rhi::DeviceTextureView),
    ) {
        let list = device.gfx_queue.get_command_buffer(&device);

        list.set_viewport(gbuffer.width, gbuffer.height);
        list.set_graphics_pipeline(pso_cache.get_pso(&self.pso));
        list.set_topology(rhi::GeomTopology::Triangles);
        list.set_device_texture_barriers(&[
            (&gbuffer.diffuse, dx::ResourceStates::PixelShaderResource),
            (&gbuffer.normal, dx::ResourceStates::PixelShaderResource),
            (&gbuffer.material, dx::ResourceStates::PixelShaderResource),
            (&shadow_mask.0, dx::ResourceStates::PixelShaderResource),
        ]);

        list.clear_render_target(&gbuffer.accum_rtv, 0.0, 0.0, 0.0);
        list.set_render_targets(&[&gbuffer.accum_rtv], None);
        list.set_graphics_cbv(&camera_buffer.cbv[frame_idx], 0);

        list.set_graphics_cbv(&self.dir_light_buffer.cbv[0], 1);
        list.set_graphics_cbv(&self.ambient_light_buffer.cbv[0], 2);

        list.set_graphics_srv(&gbuffer.diffuse_srv, 3);
        list.set_graphics_srv(&gbuffer.normal_srv, 4);
        list.set_graphics_srv(&gbuffer.material_srv, 5);
        list.set_graphics_srv(&shadow_mask.1, 6);
        list.draw(3);

        list.set_device_texture_barrier(shadow_mask.0, dx::ResourceStates::Common, None);
        device.gfx_queue.stash_cmd_buffer(list);
    }
}
