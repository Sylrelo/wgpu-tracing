use std::borrow::Cow;

use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, Label, PipelineLayoutDescriptor, Queue, ShaderModule,
    ShaderStages, StorageTextureAccess, TextureFormat,
};
use winit::window::Window;

use crate::{
    init_textures::RenderTexture,
    structs::{INTERNAL_H, INTERNAL_W},
    utils::wgpu_binding_utils::{BindGroups, BindingGeneratorBuilder},
    wgpu_utils::live_shader_compilation,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct ATrousSettingsUniform {
    pub c_phi: f32,
    pub n_phi: f32,
    pub p_phi: f32,
    pub step_width: f32,
}

pub struct DenoiserBindGroups {
    pub textures: BindGroups,
    pub setting_uniform: BindGroups,
}

pub struct DenoiserPipeline {
    bind_groups: DenoiserBindGroups,
    pipeline: ComputePipeline,
    shader_module: ShaderModule,

    setting_uniform_buffer: Buffer,
    pub settings: ATrousSettingsUniform,
}

#[allow(dead_code)]
impl DenoiserPipeline {
    pub fn new(device: &Device, textures: &RenderTexture) -> Self {
        let setting_uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Label::from("Uniform Denoise Buffer"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let texture_bind_groups: BindGroups = Self::create_textures_bind_groups(device, textures);

        let bind_groups = DenoiserBindGroups {
            textures: texture_bind_groups,
            setting_uniform: BindingGeneratorBuilder::new(device)
                .with_default_buffer_uniform(ShaderStages::COMPUTE, &setting_uniform_buffer)
                .done()
                .build(),
        };

        let shader_module = Self::get_shader_module(device);

        let pipeline = Self::init_pipeline(device, &bind_groups, &shader_module);

        Self {
            shader_module,
            pipeline,
            bind_groups,

            setting_uniform_buffer,

            settings: ATrousSettingsUniform {
                c_phi: 1.0,
                n_phi: 0.5,
                p_phi: 0.1,
                step_width: 1.0,
            },
        }
    }

    pub fn update_uniform_settings(&mut self, queue: &Queue) {
        queue.write_buffer(
            &self.setting_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.settings]),
        )
    }

    fn create_textures_bind_groups(device: &Device, textures: &RenderTexture) -> BindGroups {
        BindingGeneratorBuilder::new(device)
            // .with_texture_only(ShaderStages::COMPUTE, &textures.color_view)
            // .done()
            .with_texture_only(ShaderStages::COMPUTE, &textures.normal_view)
            .done()
            .with_texture_only(ShaderStages::COMPUTE, &textures.depth_view)
            .done()
            .with_storage_texture(
                &textures.color_view,
                TextureFormat::Rgba8Unorm,
                StorageTextureAccess::ReadWrite,
            )
            .visibility(ShaderStages::COMPUTE)
            .done()
            // .with_storage_texture(
            //     &textures.render_view,
            //     TextureFormat::Rgba8Unorm,
            //     StorageTextureAccess::WriteOnly,
            // )
            // .visibility(ShaderStages::COMPUTE)
            // .done()
            .build()
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
        compute_pass.set_bind_group(0, &self.bind_groups.textures.bind_group, &[]);
        compute_pass.set_bind_group(1, &self.bind_groups.setting_uniform.bind_group, &[]);
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
            bind_group_layouts: &[
                &bind_groups.textures.bind_group_layout,
                &bind_groups.setting_uniform.bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Label::from("Denoiser Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader_module,
            entry_point: "main",
        })
    }

    pub fn shader_realtime_compilation(&mut self, device: &Device, window: &Window) {
        const SHADER_PATH: &str = "shaders/denoiser.wgsl";

        let shader = live_shader_compilation(device, SHADER_PATH.to_string());

        if shader.is_some() {
            self.recreate_pipeline(device, shader.unwrap());
            window.request_redraw();
        }
    }
}
