use std::borrow::Cow;

use wgpu::{
    Buffer, CommandEncoder, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, Label, PipelineLayoutDescriptor, ShaderModule, ShaderSource, ShaderStages,
    StorageTextureAccess, TextureFormat,
};
use winit::window::Window;

use crate::{
    init_textures::RenderTexture,
    structs::{INTERNAL_H, INTERNAL_W},
    utils::wgpu_binding_utils::{BindGroups, BindingGeneratorBuilder},
    wgpu_utils::live_shader_compilation,
};

struct TemporalReprojectionBindings {
    textures: BindGroups,
    uniforms: BindGroups,
}

pub struct TemporalReprojection {
    pipeline: ComputePipeline,
    shader_module: ShaderModule,
    bind_groups: TemporalReprojectionBindings,
}

#[allow(dead_code)]
impl TemporalReprojection {
    pub fn new(device: &Device, textures: &RenderTexture, camera_buffer: &Buffer) -> Self {
        println!("Init Temporal Reprojection Pipeline");

        let uniform_bindingds = BindingGeneratorBuilder::new(device)
            .with_default_buffer_uniform(ShaderStages::COMPUTE, camera_buffer)
            .done()
            .build();

        let textures_bindings = Self::create_bind_groups(device, textures);

        let shader_module = Self::get_shader_modules(device);

        let bind_groups = TemporalReprojectionBindings {
            textures: textures_bindings,
            uniforms: uniform_bindingds,
        };

        let pipeline = Self::init_pipeline(device, &bind_groups, &shader_module);

        Self {
            pipeline: pipeline,
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
        compute_pass.set_bind_group(0, &self.bind_groups.uniforms.bind_group, &[]);
        compute_pass.set_bind_group(1, &self.bind_groups.textures.bind_group, &[]);
        compute_pass.dispatch_workgroups(INTERNAL_W / 16, INTERNAL_H / 16, 1);
    }

    // ===============================

    fn get_shader_modules(device: &Device) -> ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Label::from("Temporal Reprojection Shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("./temporal_reprojection.wgsl"))),
        })
    }

    pub fn shader_realtime_compilation(&mut self, device: &Device, window: &Window) {
        const SHADER_PATH: &str = "src/pipelines/temporal_reprojection/temporal_reprojection.wgsl";

        let shader = live_shader_compilation(device, SHADER_PATH.to_string());

        if shader.is_some() {
            self.recreate_pipeline(device, shader.unwrap());
            window.request_redraw();
        }
    }

    // ===============================

    fn init_pipeline(
        device: &Device,
        bind_groups: &TemporalReprojectionBindings,
        shader_module: &ShaderModule,
    ) -> ComputePipeline {
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from("Temporal Reprojection Layout"),
            bind_group_layouts: &[
                &bind_groups.uniforms.bind_group_layout,
                &bind_groups.textures.bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Label::from("Temporal Reprojection Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader_module,
            entry_point: "main",
        })
    }

    fn create_bind_groups(device: &Device, textures: &RenderTexture) -> BindGroups {
        BindingGeneratorBuilder::new(device)
            .with_texture_only(ShaderStages::COMPUTE, &textures.color_view)
            .done()
            .with_texture_only(ShaderStages::COMPUTE, &textures.velocity_view)
            .done()
            .with_texture_only(ShaderStages::COMPUTE, &textures.depth_view)
            .done()
            .with_storage_texture(
                &textures.accumulated_view,
                TextureFormat::Rgba8Unorm,
                StorageTextureAccess::ReadWrite,
            )
            .visibility(ShaderStages::COMPUTE)
            .done()
            .with_storage_texture(
                &textures.accumulated_history_view,
                TextureFormat::Rgba8Unorm,
                StorageTextureAccess::WriteOnly,
            )
            .visibility(ShaderStages::COMPUTE)
            .done()
            .build()
    }
}
