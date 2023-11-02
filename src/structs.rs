use wgpu::{PipelineLayout, RenderPipeline, SurfaceCapabilities, TextureFormat, Texture, TextureView};
use winit::dpi::PhysicalSize;
use winit::window::Window;

// pub const INTERNAL_W: u32 = (1920.0f32 / 1.0) as u32;
// pub const INTERNAL_H: u32 = (1080.0f32 / 1.0) as u32;
pub const INTERNAL_W: u32 = (1280) as u32;
pub const INTERNAL_H: u32 = (720) as u32;

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


#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct Camera {
    pub position: [f32; 4],
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
    pub node_index: usize,
    pub _padding: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct VoxelWorldTest {
    // pub voxel: [f32; 4],
    pub voxel: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct BvhNodeGpu {
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],
    pub entry_index: u32,
    pub exit_index: u32,
    pub shape_index: u32,
    pub _padding: u32,
}

impl BvhNodeGpu {
    pub fn new(
        aabb: &bvh::aabb::AABB,
        entry_index: u32,
        exit_index: u32,
        shape_index: u32,
    ) -> BvhNodeGpu {
        BvhNodeGpu {
            aabb_min: [aabb.min.x, aabb.min.y, aabb.min.z, 0.0],
            aabb_max: [aabb.max.x, aabb.max.y, aabb.max.z, 0.0],
            entry_index,
            exit_index,
            shape_index,
            _padding: 0,
        }
    }
}

// #[repr(C)]
// #[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
// pub struct TriangleBinding {
//     pub triangles: Vec<Triangle>,
// }
