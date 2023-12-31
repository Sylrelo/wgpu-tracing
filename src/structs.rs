use rust_embed::RustEmbed;
use wgpu::{PipelineLayout, RenderPipeline, SurfaceCapabilities, TextureFormat};
use winit::dpi::PhysicalSize;
use winit::window::Window;

// pub const INTERNAL_W: u32 = (1920.0f32 / 1.0) as u32;
// pub const INTERNAL_H: u32 = (1080.0f32 / 1.0) as u32;
pub const INTERNAL_W: u32 = (1280.0 * 0.5) as u32;
pub const INTERNAL_H: u32 = (720.0 * 0.5) as u32;

// EMBEDS
#[derive(RustEmbed)]
#[folder = "shaders/"]
pub struct ShaderAssets;

pub struct SwapchainData {
    pub capabilities: SurfaceCapabilities,
    pub format: TextureFormat,
}

pub struct RenderContext {
    pub pipeline: RenderPipeline,
    pub layout: PipelineLayout,
}

pub struct App {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    pub window: Window,
    pub swapchain_config: SwapchainData,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct RenderUniform {
    pub position_offset: [f32; 4],
}
