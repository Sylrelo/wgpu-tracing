use std::borrow::Cow;

use wgpu::{
    CommandEncoder, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    Label, PipelineLayoutDescriptor, ShaderModule, ShaderStages, StorageTextureAccess,
    TextureFormat,
};

use crate::{
    init_textures::RenderTexture,
    structs::{INTERNAL_H, INTERNAL_W},
    utils::wgpu_binding_utils::{BindGroups, BindingGeneratorBuilder},
};

pub struct DenoiserBindGroups {
    pub textures: BindGroups,
}

pub struct DenoiserPipeline {
    pub bind_groups: DenoiserBindGroups,
    pub pipeline: ComputePipeline,
    pub shader_module: ShaderModule,
}

#[allow(dead_code)]
impl DenoiserPipeline {
    pub fn new(device: &Device, textures: &RenderTexture) -> Self {
        let texture_bind_groups: BindGroups = Self::create_textures_bind_groups(device, textures);

        let bind_groups = DenoiserBindGroups {
            textures: texture_bind_groups,
        };

        let shader_module = Self::get_shader_module(device);
        let pipeline = Self::init_pipeline(device, &bind_groups, &shader_module);

        Self {
            shader_module,
            pipeline,
            bind_groups,
        }
    }

    fn create_textures_bind_groups(device: &Device, textures: &RenderTexture) -> BindGroups {
        BindingGeneratorBuilder::new(device)
            // .with_storage_texture(
            //     &textures.render_view,
            //     TextureFormat::Rgba8Unorm,
            //     StorageTextureAccess::ReadOnly,
            // )
            // .visibility(ShaderStages::COMPUTE)
            .with_texture_only(ShaderStages::COMPUTE, &textures.color_view)
            .done()
            .with_texture_only(ShaderStages::COMPUTE, &textures.normal_view)
            .done()
            .with_storage_texture(
                &textures.render_view,
                TextureFormat::Rgba8Unorm,
                StorageTextureAccess::WriteOnly,
            )
            .visibility(ShaderStages::COMPUTE)
            .done()
            .build()
    }

    pub fn recreate_pipeline(&mut self, device: &Device, shader_module: ShaderModule) {
        self.pipeline = Self::init_pipeline(device, &self.bind_groups, &self.shader_module);
        self.shader_module = shader_module;
    }

    pub fn exec_pass(&self, encoder: &mut CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_groups.textures.bind_group, &[]);
        // compute_pass.set_bind_group(1, &self.storage_binds.bind_group, &[]);
        // compute_pass.set_bind_group(2, &self.uniform_binds.bind_group, &[]);
        compute_pass.dispatch_workgroups(INTERNAL_W / 16, INTERNAL_H / 16, 1);
    }

    fn get_shader_module(device: &Device) -> ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Label::from("Denoiser Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/denoiser.wgsl"
            ))),
        })
    }

    fn init_pipeline(
        device: &Device,
        bind_groups: &DenoiserBindGroups,
        shader_module: &ShaderModule,
    ) -> ComputePipeline {
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from("Denoiser Layout"),
            bind_group_layouts: &[&bind_groups.textures.bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Label::from("Denoiser Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader_module,
            entry_point: "main",
        })
    }
}
