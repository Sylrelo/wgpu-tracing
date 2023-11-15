use std::{borrow::Cow, fs, time::UNIX_EPOCH};

use wgpu::{
    CommandEncoder, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device,
    Label, PipelineLayoutDescriptor, ShaderModule, ShaderSource, ShaderStages,
    StorageTextureAccess, TextureFormat,
};
use winit::window::Window;

use crate::{
    init_textures::RenderTexture,
    utils::wgpu_binding_utils::{BindGroups, BindingGeneratorBuilder},
    wgpu_utils::compile_shader,
};

pub struct UpscalerPipeline {
    upscale: ComputePipeline,
    upscale_shader_module: ShaderModule,
    // pub upscale_bind_groups: UpscalerPipelineBindGroups,
    sharpen: ComputePipeline,
    sharpen_shader_module: ShaderModule,

    bind_groups_easu: BindGroups,
    bind_groups_rcas: BindGroups,
    // pub sharpen_bind_groups: UpscalerPipelineBindGroups,
}

#[allow(dead_code)]
impl UpscalerPipeline {
    pub fn new(device: &Device, textures: &RenderTexture) -> Self {
        println!("Init UpscalerPipeline");

        let bind_groups_easu = Self::create_easu_bind_groups(device, textures);
        let bind_groups_rcas = Self::create_rcas_bind_groups(device, textures);

        let (easu_module, rcas_module) = Self::get_shader_modules(device);

        let upscale_pipeline = Self::init_pipeline(device, &bind_groups_easu, &easu_module);
        let sharpen_pipeline = Self::init_pipeline(device, &bind_groups_rcas, &rcas_module);

        Self {
            upscale_shader_module: easu_module,
            upscale: upscale_pipeline,
            bind_groups_easu: bind_groups_easu,

            sharpen_shader_module: rcas_module,
            sharpen: sharpen_pipeline,
            bind_groups_rcas: bind_groups_rcas,
            // sharpen: pipeline,
            // sharpen_bind_groups: bind_groups,
        }
    }

    pub fn recreate_pipelines(&mut self, device: &Device, shader_module: ShaderModule) {
        self.upscale =
            Self::init_pipeline(device, &self.bind_groups_easu, &self.upscale_shader_module);
        self.upscale_shader_module = shader_module;
    }

    pub fn exec_passes(&self, encoder: &mut CommandEncoder) {
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.upscale);
            compute_pass.set_bind_group(0, &self.bind_groups_easu.bind_group, &[]);
            compute_pass.dispatch_workgroups(1920 / 16, 1080 / 16, 1);
        }
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.sharpen);
            compute_pass.set_bind_group(0, &self.bind_groups_rcas.bind_group, &[]);
            compute_pass.dispatch_workgroups(1920 / 16, 1080 / 16, 1);
        }
    }

    // ===============================

    pub fn shader_realtime_compilation(&mut self, device: &Device, window: &Window) {
        const SHADER_PATH: &str = "src/pipelines/upscaler/amd_fsr_easu.wgsl";
        static mut LAST_TIME: u64 = 0;

        let modification_time = fs::metadata(SHADER_PATH);

        if modification_time.is_err() {
            return;
        }

        let modification_time = modification_time
            .unwrap()
            .modified()
            .unwrap()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        unsafe {
            if LAST_TIME == 0 {
                LAST_TIME = modification_time;
            }

            if modification_time - LAST_TIME == 0 {
                return;
            }

            LAST_TIME = modification_time;
        }

        let compiled_shader = compile_shader(device, SHADER_PATH);
        if compiled_shader.is_some() {
            self.recreate_pipelines(device, compiled_shader.unwrap());
            window.request_redraw();
        }
    }

    fn get_shader_modules(device: &Device) -> (ShaderModule, ShaderModule) {
        let easu = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Label::from("UPSCALER Shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("./amd_fsr_easu.wgsl"))),
        });

        let rcas = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Label::from("UPSCALER Shader"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("./amd_fsr_rcas.wgsl"))),
        });

        return (easu, rcas);
    }

    // ===============================

    fn init_pipeline(
        device: &Device,
        bind_groups: &BindGroups,
        shader_module: &ShaderModule,
    ) -> ComputePipeline {
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from("Upscaler Layout"),
            bind_group_layouts: &[&bind_groups.bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Label::from("Upscaler Pipeline New"),
            layout: Some(&pipeline_layout),
            module: shader_module,
            entry_point: "main",
        })
    }

    fn create_easu_bind_groups(device: &Device, textures: &RenderTexture) -> BindGroups {
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

    fn create_rcas_bind_groups(device: &Device, textures: &RenderTexture) -> BindGroups {
        BindingGeneratorBuilder::new(device)
            .with_storage_texture(
                &textures.final_render_view,
                TextureFormat::Rgba8Unorm,
                StorageTextureAccess::ReadWrite,
            )
            .visibility(ShaderStages::COMPUTE)
            .done()
            .build()
    }
}
