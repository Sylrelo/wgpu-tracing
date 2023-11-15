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

pub struct UpscalerPipelineBindGroups {
    pub textures: BindGroups,
}

// pub struct TracingPipelineBuffers {
//     pub chunk_content: Buffer,
//     pub chunk_content_size: u32,

//     pub uniform: Buffer,

//     pub root_grid: Buffer,
// }

pub struct UpscalerPipeline {
    pub pipeline: ComputePipeline,
    pub shader_module: ShaderModule,

    pub bind_groups: UpscalerPipelineBindGroups,
}

#[allow(dead_code)]
impl UpscalerPipeline {
    pub fn new(device: &Device, textures: &RenderTexture) -> Self {
        println!("Init UpscalerPipeline");

        let bind_groups = UpscalerPipelineBindGroups {
            textures: Self::create_textures_bind_groups(device, textures),
        };

        let shader_module = Self::get_shader_module(device);
        let pipeline = Self::init_pipeline(device, &bind_groups, &shader_module);

        Self {
            shader_module,
            pipeline,

            bind_groups,
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
        compute_pass.set_bind_group(0, &self.bind_groups.textures.bind_group, &[]);
        compute_pass.dispatch_workgroups(1920 / 16, 1080 / 16, 1);
    }

    // ===============================

    fn get_shader_module(device: &Device) -> ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Label::from("Tracing Shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("./shader.wgsl"))),
        })
    }

    fn init_pipeline(
        device: &Device,
        bind_groups: &UpscalerPipelineBindGroups,
        shader_module: &ShaderModule,
    ) -> ComputePipeline {
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from("Upscaler Layout"),
            bind_group_layouts: &[
                &bind_groups.textures.bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Label::from("Upscaler Pipeline New"),
            layout: Some(&pipeline_layout),
            module: shader_module,
            entry_point: "main",
        })
    }

    fn create_textures_bind_groups(device: &Device, textures: &RenderTexture) -> BindGroups {
        BindingGeneratorBuilder::new(device)
            .with_storage_texture(
                &textures.color_view, // TODO : Change to RenderView (denoised)
                TextureFormat::Rgba8Unorm,
                StorageTextureAccess::ReadOnly,
            )
            .visibility(ShaderStages::COMPUTE)
            .done()
            .with_storage_texture(
                &textures.final_render_view,
                TextureFormat::Rgba8Unorm,
                StorageTextureAccess::WriteOnly,
            )
            .visibility(ShaderStages::COMPUTE)
            .done()
            .build()
    }
}
