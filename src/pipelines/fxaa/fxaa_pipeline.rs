use std::borrow::Cow;

use wgpu::{
    CommandEncoder, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    Label, PipelineLayoutDescriptor, ShaderModule, ShaderSource, ShaderStages,
    StorageTextureAccess, TextureFormat,
};

use crate::{
    init_textures::RenderTexture,
    structs::{INTERNAL_H, INTERNAL_W},
    utils::wgpu_binding_utils::{BindGroups, BindingGeneratorBuilder},
};

pub struct FXAAPipeline {
    pipeline: ComputePipeline,
    shader_module: ShaderModule,

    bind_groups: BindGroups,
}

#[allow(dead_code)]
impl FXAAPipeline {
    pub fn new(device: &Device, textures: &RenderTexture) -> Self {
        println!("Init FXAA Pipeline");

        let bind_groups = Self::create_bind_groups(device, textures);

        let shader_module = Self::get_shader_modules(device);

        let fxaa_pipeline = Self::init_pipeline(device, &bind_groups, &shader_module);

        Self {
            pipeline: fxaa_pipeline,
            shader_module,
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
        compute_pass.set_bind_group(0, &self.bind_groups.bind_group, &[]);
        compute_pass.dispatch_workgroups(INTERNAL_W / 16, INTERNAL_H / 16, 1);
    }

    // ===============================

    fn get_shader_modules(device: &Device) -> ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Label::from("FXAA Shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("./fxaa.comp.wgsl"))),
        })
    }

    // ===============================

    fn init_pipeline(
        device: &Device,
        bind_groups: &BindGroups,
        shader_module: &ShaderModule,
    ) -> ComputePipeline {
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from("FXAA Layout"),
            bind_group_layouts: &[&bind_groups.bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Label::from("FXAA Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader_module,
            entry_point: "main",
        })
    }

    fn create_bind_groups(device: &Device, textures: &RenderTexture) -> BindGroups {
        BindingGeneratorBuilder::new(device)
            .with_storage_texture(
                &textures.render_view,
                TextureFormat::Rgba8Unorm,
                StorageTextureAccess::ReadWrite,
            )
            .visibility(ShaderStages::COMPUTE)
            .done()
            .build()
    }
}
