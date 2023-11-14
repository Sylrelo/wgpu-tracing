use wgpu::{
    AddressMode, Device, FilterMode, Sampler, SamplerDescriptor, Texture, TextureDimension,
    TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

use crate::structs::{INTERNAL_H, INTERNAL_W};

pub struct RenderTexture {
    pub render: Texture,
    pub render_view: TextureView,
    pub render_sanpler: Sampler,

    pub color: Texture,
    pub color_view: TextureView,

    pub normal: Texture,
    pub normal_view: TextureView,

    pub depth: Texture,
    pub depth_view: TextureView,
}

impl RenderTexture {
    pub fn new(device: &Device) -> Self {
        let render =
            Self::create_texture_helper(device, INTERNAL_W, INTERNAL_H, TextureFormat::Rgba8Unorm);
        let render_view = render.create_view(&TextureViewDescriptor::default());
        let render_sanpler = Self::create_sampler_helper(device);

        let color =
            Self::create_texture_helper(device, INTERNAL_W, INTERNAL_H, TextureFormat::Rgba8Unorm);
        let color_view = color.create_view(&TextureViewDescriptor::default());

        let normal =
            Self::create_texture_helper(device, INTERNAL_W, INTERNAL_H, TextureFormat::Rgba8Snorm);
        let normal_view = normal.create_view(&TextureViewDescriptor {
            format: Some(TextureFormat::Rgba8Snorm),
            ..TextureViewDescriptor::default()
        });

        let depth =
            Self::create_texture_helper(device, INTERNAL_W, INTERNAL_H, TextureFormat::Rgba8Unorm);
        let depth_view = depth.create_view(&TextureViewDescriptor {
            format: Some(TextureFormat::Rgba8Unorm),
            ..TextureViewDescriptor::default()
        });

        Self {
            render,
            render_view,
            render_sanpler,

            color,
            color_view,

            normal,
            normal_view,

            depth,
            depth_view,
        }
    }

    fn create_sampler_helper(device: &Device) -> Sampler {
        device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
            ..Default::default()
        })
    }

    fn create_texture_helper(
        device: &Device,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Texture {
        let texture_size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };

        device.create_texture(&wgpu::TextureDescriptor {
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING
                | TextureUsages::COPY_DST
                | TextureUsages::STORAGE_BINDING,
            label: Some("diffuse_texture"),
            view_formats: &[],
        })
    }
}
