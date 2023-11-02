use wgpu::{Device, Texture, TextureDimension, TextureFormat, TextureUsages, TextureView};

use crate::structs::{INTERNAL_H, INTERNAL_W};

pub struct RenderTexture {
    pub render: Texture,
    pub render_view: TextureView,

    // pub color: Texture,
    // pub color_view: TextureView,

    // pub normal: Texture,
    // pub normal_view: TextureView,

    // pub depth: Texture,
    // pub depth_view: TextureView,
}

impl RenderTexture {
    pub fn new(device: &Device) -> Self {
        let render_texture =
            Self::create_texture_helper(device, INTERNAL_W, INTERNAL_H, TextureFormat::Rgba8Unorm);
        let render_texture_view = render_texture.create_view(&wgpu::TextureViewDescriptor::default());


        Self {
          render: render_texture,
          render_view: render_texture_view,
        }
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
