use std::borrow::Cow;

use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, Label, PipelineLayoutDescriptor, Queue, ShaderModule,
    ShaderSource, ShaderStages, StorageTextureAccess, TextureFormat,
};

use crate::{
    init_textures::RenderTexture,
    structs::{INTERNAL_H, INTERNAL_W},
    utils::wgpu_binding_utils::{BindGroups, BindingGeneratorBuilder},
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct TracingPipelineSettings {
    pub player_position: [f32; 4],
    pub chunk_count: u32,
    pub _padding: u32,
}

pub struct TracingPipelineBindGroups {
    pub textures: BindGroups,
    pub uniforms: BindGroups,
    pub buffers: BindGroups,
}

pub struct TracingPipelineBuffers {
    pub chunk_content: Buffer,
    pub chunk_content_size: u32,

    pub uniform: Buffer,

    pub root_grid: Buffer,
}

pub struct TracingPipelineTest {
    pub pipeline: ComputePipeline,
    pub shader_module: ShaderModule,

    pub bind_groups: TracingPipelineBindGroups,

    pub buffers: TracingPipelineBuffers,
}

#[allow(dead_code)]
impl TracingPipelineTest {
    pub fn new(device: &Device, textures: &RenderTexture) -> Self {
        println!("Init TracingPipelineTest");

        let buffers = Self::create_buffers(device);

        let bind_groups = TracingPipelineBindGroups {
            textures: Self::create_textures_bind_groups(device, textures),
            uniforms: Self::uniform_create_bind_groups(device, &buffers.uniform),
            buffers: Self::buffers_create_bind_groups(device, &buffers),
        };
        println!("Buffers done.");

        let shader_module = Self::get_shader_module(device);
        println!("Shader done.");
        let pipeline = Self::init_pipeline(device, &bind_groups, &shader_module);
        println!("Init done.");
        Self {
            shader_module,
            pipeline,

            bind_groups,
            buffers,
        }
    }

    pub fn recreate_pipeline(&mut self, device: &Device, shader_module: ShaderModule) {
        self.pipeline = Self::init_pipeline(device, &self.bind_groups, &self.shader_module);
        self.shader_module = shader_module;
    }

    pub fn exec_pass(&self, encoder: &mut CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            // timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_groups.uniforms.bind_group, &[]);
        compute_pass.set_bind_group(1, &self.bind_groups.buffers.bind_group, &[]);
        compute_pass.set_bind_group(2, &self.bind_groups.textures.bind_group, &[]);
        compute_pass.dispatch_workgroups(INTERNAL_W / 16, INTERNAL_H / 16, 1);
    }

    // ===============================

    pub fn buffer_chunk_content_update(&self, queue: &Queue, content: &Vec<u32>) {
        queue.write_buffer(
            &self.buffers.chunk_content,
            0,
            bytemuck::cast_slice(content.as_slice()),
        );
    }

    pub fn buffer_root_grid_update(&self, queue: &Queue, grid: &Vec<[i32; 4]>) {
        queue.write_buffer(
            &self.buffers.root_grid,
            0,
            bytemuck::cast_slice(grid.as_slice()),
        );
    }

    pub fn uniform_settings_update(&self, queue: &Queue, settings: TracingPipelineSettings) {
        queue.write_buffer(&self.buffers.uniform, 0, bytemuck::cast_slice(&[settings]));
    }

    // ===============================

    fn get_shader_module(device: &Device) -> ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Label::from("Tracing Shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/simple_raytracer_tests.wgsl"
            ))),
        })
    }

    fn init_pipeline(
        device: &Device,
        bind_groups: &TracingPipelineBindGroups,
        shader_module: &ShaderModule,
    ) -> ComputePipeline {
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from("Tracing Layout"),
            bind_group_layouts: &[
                &bind_groups.uniforms.bind_group_layout,
                &bind_groups.buffers.bind_group_layout,
                &bind_groups.textures.bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Label::from("Tracing Pipeline New"),
            layout: Some(&pipeline_layout),
            module: shader_module,
            entry_point: "main",
        })
    }

    fn create_textures_bind_groups(device: &Device, textures: &RenderTexture) -> BindGroups {
        BindingGeneratorBuilder::new(device)
            .with_storage_texture(
                &textures.render_view,
                TextureFormat::Rgba8Unorm,
                StorageTextureAccess::WriteOnly,
            )
            .visibility(ShaderStages::COMPUTE)
            .done()
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
                StorageTextureAccess::WriteOnly,
            )
            .visibility(ShaderStages::COMPUTE)
            .done()
            .build()
    }

    fn uniform_create_bind_groups(device: &Device, buffer: &Buffer) -> BindGroups {
        BindingGeneratorBuilder::new(device)
            .with_default_buffer_uniform(ShaderStages::COMPUTE, buffer)
            .done()
            .build()
    }

    fn buffers_create_bind_groups(device: &Device, buffers: &TracingPipelineBuffers) -> BindGroups {
        BindingGeneratorBuilder::new(device)
            .with_default_buffer_storage(ShaderStages::COMPUTE, &buffers.chunk_content, true)
            .done()
            .with_default_buffer_storage(ShaderStages::COMPUTE, &buffers.root_grid, true)
            .done()
            .build()
    }

    fn create_buffers(device: &Device) -> TracingPipelineBuffers {
        let uniform = device.create_buffer(&BufferDescriptor {
            label: Label::from("Tracing Pipeline : Uniform Buffer"),
            mapped_at_creation: false,
            size: 32,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let chunk_content = device.create_buffer(&BufferDescriptor {
            label: Label::from("Tracing Pipeline : Chunk Content Buffer"),
            mapped_at_creation: false,
            size: 2147483640,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let root_grid = device.create_buffer(&BufferDescriptor {
            label: Label::from("Tracing Pipeline : ROOT GRID"),
            mapped_at_creation: false,
            size: 30 * 30 * 16,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        TracingPipelineBuffers {
            chunk_content,
            chunk_content_size: 900 * 4,
            uniform,

            root_grid,
        }
    }
}
