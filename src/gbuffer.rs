use std::sync::Arc;

use oxidx::dx;

use crate::rhi;

#[derive(Debug)]
pub struct Gbuffer {
    pub diffuse: rhi::DeviceTexture,
    pub diffuse_rtv: rhi::TextureView,
    pub diffuse_srv: rhi::TextureView,

    pub normal: rhi::DeviceTexture,
    pub normal_rtv: rhi::TextureView,
    pub normal_srv: rhi::TextureView,

    pub material: rhi::DeviceTexture,
    pub material_rtv: rhi::TextureView,
    pub material_srv: rhi::TextureView,

    pub accum: rhi::DeviceTexture,
    pub accum_rtv: rhi::TextureView,
    pub accum_srv: rhi::TextureView,

    pub depth: rhi::DeviceTexture,
    pub depth_dsv: rhi::TextureView,
}

impl Gbuffer {
    pub fn new(device: &Arc<rhi::Device>, width: u32, height: u32) -> Self {
        let diffuse = rhi::DeviceTexture::new(
            device,
            width,
            height,
            dx::Format::Rgba32Float,
            1,
            dx::ResourceFlags::AllowRenderTarget,
            dx::ResourceStates::RenderTarget,
            Some(dx::ClearValue::color(
                dx::Format::Rgba32Float,
                [0.0, 0.0, 0.0, 1.0],
            )),
            "Diffuse Texture",
        );

        let diffuse_rtv =
            rhi::TextureView::new(device, &diffuse, rhi::TextureViewType::RenderTarget, None);

        let diffuse_srv =
            rhi::TextureView::new(device, &diffuse, rhi::TextureViewType::ShaderResource, None);

        let normal = rhi::DeviceTexture::new(
            device,
            width,
            height,
            dx::Format::Rgba32Float,
            1,
            dx::ResourceFlags::AllowRenderTarget,
            dx::ResourceStates::RenderTarget,
            Some(dx::ClearValue::color(
                dx::Format::Rgba32Float,
                [0.0, 0.0, 0.0, 1.0],
            )),
            "Diffuse Texture",
        );

        let normal_rtv =
            rhi::TextureView::new(device, &normal, rhi::TextureViewType::RenderTarget, None);

        let normal_srv =
            rhi::TextureView::new(device, &normal, rhi::TextureViewType::ShaderResource, None);

        let material = rhi::DeviceTexture::new(
            device,
            width,
            height,
            dx::Format::Rgba32Float,
            1,
            dx::ResourceFlags::AllowRenderTarget,
            dx::ResourceStates::RenderTarget,
            Some(dx::ClearValue::color(
                dx::Format::Rgba32Float,
                [0.0, 0.0, 0.0, 1.0],
            )),
            "Diffuse Texture",
        );

        let material_rtv =
            rhi::TextureView::new(device, &material, rhi::TextureViewType::RenderTarget, None);

        let material_srv = rhi::TextureView::new(
            device,
            &material,
            rhi::TextureViewType::ShaderResource,
            None,
        );

        let accum = rhi::DeviceTexture::new(
            device,
            width,
            height,
            dx::Format::Rgba32Float,
            1,
            dx::ResourceFlags::AllowRenderTarget,
            dx::ResourceStates::RenderTarget,
            Some(dx::ClearValue::color(
                dx::Format::Rgba32Float,
                [0.0, 0.0, 0.0, 1.0],
            )),
            "Diffuse Texture",
        );

        let accum_rtv =
            rhi::TextureView::new(device, &accum, rhi::TextureViewType::RenderTarget, None);

        let accum_srv =
            rhi::TextureView::new(device, &accum, rhi::TextureViewType::ShaderResource, None);

        let depth = rhi::DeviceTexture::new(
            &device,
            width,
            height,
            dx::Format::D24UnormS8Uint,
            1,
            dx::ResourceFlags::AllowDepthStencil,
            dx::ResourceStates::DepthWrite,
            Some(dx::ClearValue::depth(dx::Format::D24UnormS8Uint, 1.0, 0)),
            "Depth Buffer",
        );

        let depth_dsv =
            rhi::TextureView::new(&device, &depth, rhi::TextureViewType::DepthTarget, None);

        Self {
            diffuse,
            diffuse_rtv,
            diffuse_srv,
            normal,
            normal_rtv,
            normal_srv,
            material,
            material_rtv,
            material_srv,
            accum,
            accum_rtv,
            accum_srv,
            depth,
            depth_dsv,
        }
    }
}
