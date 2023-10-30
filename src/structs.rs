use wgpu::{PipelineLayout, RenderPipeline, SurfaceCapabilities, TextureFormat};
use winit::dpi::PhysicalSize;
use winit::window::Window;

pub const INTERNAL_W: u32 = (1920.0f32 / 2.0) as u32;
pub const INTERNAL_H: u32 = (1080.0f32 / 2.0) as u32;

pub struct SwapchainData {
    pub capabilities: SurfaceCapabilities,
    pub format: TextureFormat,
}

pub struct RenderContext {
    pub pipeline: RenderPipeline,
    pub layout: PipelineLayout,
    // pub bind_groups: BindGroupLayout,
}

// pub struct Pipelines {
//     pub tracing: TracingPipeline,
//     pub render: RenderContext,
// }

pub struct App {
    pub surface: wgpu::Surface,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    pub window: Window,
    pub swapchain_config: SwapchainData,
}

// UNIFORMS

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct ComputeUniform {
    pub view_proj: [[f32; 4]; 4],
    pub test: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct Triangle {
    pub p0: [f32; 4],
    pub p1: [f32; 4],
    pub p2: [f32; 4],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct Voxel {
    pub min: [f32; 4],
    pub max: [f32; 4],
    pub pos: [f32; 4],
}

// #[repr(C)]
// #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
// pub struct TriangleBinding {
//     pub triangles: Vec<Triangle>,
// }
