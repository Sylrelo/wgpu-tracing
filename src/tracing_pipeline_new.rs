use std::borrow::Cow;

use wgpu::{
    Buffer, CommandEncoder, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Label, PipelineLayoutDescriptor, ShaderModule, ShaderSource, ShaderStages,
    StorageTextureAccess, TextureFormat,
};
use winit::window::Window;

use crate::{
    init_buffers::Buffers,
    init_textures::RenderTexture,
    structs::{INTERNAL_H, INTERNAL_W},
    utils::wgpu_binding_utils::{BindGroups, BindingGeneratorBuilder},
    wgpu_utils::live_shader_compilation,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct TracingPipelineSettings {
    pub player_position: [f32; 4],
    pub chunk_count: u32,
    pub frame_random_number: u32,
}

pub struct TracingPipelineBindGroups {
    pub textures: BindGroups,
    pub uniforms: BindGroups,
    pub buffers: BindGroups,
}

pub struct TracingPipelineTest {
    pub pipeline: ComputePipeline,
    pub shader_module: ShaderModule,

    pub bind_groups: TracingPipelineBindGroups,
}

#[allow(dead_code)]
impl TracingPipelineTest {
    pub fn new(
        device: &Device,
        textures: &RenderTexture,
        camera_buffer: &Buffer,
        buffers: &Buffers,
    ) -> Self {
        println!("Init TracingPipelineTest");

        let bind_groups = TracingPipelineBindGroups {
            textures: Self::create_textures_bind_groups(device, textures),
            uniforms: Self::uniform_create_bind_groups(
                device,
                buffers.rt_unidata.get_buffer(),
                &camera_buffer,
            ),
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
        }
    }

    pub fn recreate_pipeline(&mut self, device: &Device, shader_module: ShaderModule) {
        self.pipeline = Self::init_pipeline(device, &self.bind_groups, &shader_module);
        self.shader_module = shader_module;
    }

    pub fn exec_pass(&self, encoder: &mut CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_groups.uniforms.bind_group, &[]);
        compute_pass.set_bind_group(1, &self.bind_groups.buffers.bind_group, &[]);
        compute_pass.set_bind_group(2, &self.bind_groups.textures.bind_group, &[]);
        compute_pass.dispatch_workgroups(INTERNAL_W / 16, INTERNAL_H / 16, 1);
    }

    fn get_shader_module(device: &Device) -> ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Label::from("Tracing Shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/simple_raytracer_tests.wgsl"
            ))),
        })
    }

    pub fn shader_realtime_compilation(&mut self, device: &Device, window: &Window) {
        const SHADER_PATH: &str = "shaders/simple_raytracer_tests.wgsl";

        let shader = live_shader_compilation(device, SHADER_PATH.to_string());

        if shader.is_some() {
            self.recreate_pipeline(device, shader.unwrap());
            window.request_redraw();
        }
    }

    // ===============================

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
            // .with_storage_texture(
            //     &textures.render_view,
            //     TextureFormat::Rgba8Unorm,
            //     StorageTextureAccess::WriteOnly,
            // )
            // .visibility(ShaderStages::COMPUTE)
            // .done()
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
            // .with_texture_only(ShaderStages::COMPUTE, &textures.velocity_view)
            // .visibility(ShaderStages::COMPUTE)
            // .done()
            .build()
    }

    fn uniform_create_bind_groups(
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

    fn buffers_create_bind_groups(device: &Device, buffers: &Buffers) -> BindGroups {
        BindingGeneratorBuilder::new(device)
            .with_default_buffer_storage(
                ShaderStages::COMPUTE,
                &buffers.rt_chunk_content.get_buffer(),
                true,
            )
            .done()
            .with_default_buffer_storage(
                ShaderStages::COMPUTE,
                &buffers.rt_chunk_grid.get_buffer(),
                true,
            )
            .done()
            .build()
    }
}
