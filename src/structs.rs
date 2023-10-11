use wgpu::{
    BindGroup, BindGroupLayout, Buffer, ComputePipeline, PipelineLayout, RenderPipeline,
    SurfaceCapabilities, TextureFormat,
};
use winit::dpi::PhysicalSize;
use winit::window::Window;

pub struct SwapchainData {
    pub capabilities: SurfaceCapabilities,
    pub format: TextureFormat,
}

pub struct ComputeContext {
    pub pipeline: ComputePipeline,
    pub bind_group: BindGroup,
    pub bind_group_layout: BindGroupLayout,

    pub uniform: ComputeUniform,
    pub uniform_buffer: Buffer,
    // pub uniform_group_layout: BindGroupLayout,
    pub uniform_bind_group: BindGroup,
}

pub struct RenderContext {
    pub pipeline: RenderPipeline,
    pub layout: PipelineLayout,
    // pub bind_groups: BindGroupLayout,
}

pub struct Pipelines {
    pub tracing: ComputeContext,
    pub render: RenderContext,
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

// #[repr(C)]
// #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
// pub struct TriangleBinding {
//     pub triangles: Vec<Triangle>,
// }
