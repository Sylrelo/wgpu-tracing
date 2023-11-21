#![allow(dead_code)]

use wgpu::{Buffer, Device, ShaderStages, StorageTextureAccess, TextureFormat};

use crate::{
    init_textures::RenderTexture,
    utils::wgpu_binding_utils::{BindGroups, BindingGeneratorBuilder},
};

pub struct PipelineBindGroups {
    tracing_textures: Option<BindGroups>,
    tracing_uniforms: Option<BindGroups>,
    tracing_buffers: Option<BindGroups>,
}

impl PipelineBindGroups {
    pub fn new() -> Self {
        Self {
            tracing_buffers: None,
            tracing_textures: None,
            tracing_uniforms: None,
        }
    }
}

pub fn create_tracing_textures_bg(device: &Device, textures: &RenderTexture) -> BindGroups {
    BindingGeneratorBuilder::new(device)
        .with_storage_texture(
            &textures.normal_view,
            TextureFormat::Rgba8Snorm,
            StorageTextureAccess::WriteOnly,
        )
        .visibility(ShaderStages::COMPUTE)
        .done()
        .with_storage_texture(
            &textures.color_view,
            TextureFormat::Rgba8Unorm,
            StorageTextureAccess::ReadWrite,
        )
        .visibility(ShaderStages::COMPUTE)
        .done()
        .with_storage_texture(
            &textures.depth_view,
            TextureFormat::Rgba32Float,
            StorageTextureAccess::ReadWrite,
        )
        .visibility(ShaderStages::COMPUTE)
        .done()
        .with_storage_texture(
            &textures.velocity_view,
            TextureFormat::Rg32Float,
            StorageTextureAccess::WriteOnly,
        )
        .visibility(ShaderStages::COMPUTE)
        .done()
        .build()
}

pub fn create_tracing_uniforms_bg(
    device: &Device,
    settings_buffer: &Buffer,
    camera_buffer: &Buffer,
) -> BindGroups {
    BindingGeneratorBuilder::new(device)
        .with_default_buffer_uniform(ShaderStages::COMPUTE, settings_buffer)
        .done()
        .with_default_buffer_uniform(ShaderStages::COMPUTE, camera_buffer)
        .done()
        .build()
}

pub fn create_tracing_buffers_bg(
    device: &Device,
    chunk_content: &Buffer,
    chunk_grid: &Buffer,
) -> BindGroups {
    BindingGeneratorBuilder::new(device)
        .with_default_buffer_storage(ShaderStages::COMPUTE, &chunk_content, true)
        .done()
        .with_default_buffer_storage(ShaderStages::COMPUTE, &chunk_grid, true)
        .done()
        .build()
}
