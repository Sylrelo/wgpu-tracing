use std::borrow::Cow;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType,
    Buffer, BufferBindingType, BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device,
    Label, PipelineLayoutDescriptor, Queue, ShaderStages, TextureView,
};

use crate::structs::{ComputeContext, ComputeUniform};
use crate::utils::wgpu_binding_utils::{BindGroups, BindingGeneratorBuilder};

pub fn init_tracing_binding_render_texture(
    device: &Device,
    texture_view: &TextureView,
) -> BindGroups {
    BindingGeneratorBuilder::new(device)
        .with_default_storage_texture(texture_view)
        .visibility(ShaderStages::COMPUTE)
        .done()
        .build()
}

pub fn init_tracing_pipeline(
    device: &Device,
    bind_group_layouts: &[&BindGroupLayout],
) -> ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Label::from("Tracing Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("compute.wgsl"))),
    });

    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Label::from("Tracing Layout"),
        bind_group_layouts,
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Label::from("Tracing Pipeline"),
        layout: Some(&layout),
        module: &module,
        entry_point: "main",
    });

    pipeline
}

impl ComputeContext {
    pub fn uniform_init(device: &Device, uniform: ComputeUniform) -> Buffer {
        device.create_buffer_init(&BufferInitDescriptor {
            label: Some("[Compute Uniform] Buffer"),
            contents: bytemuck::cast_slice(&[uniform]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        })
    }

    #[allow(dead_code)]
    pub fn uniform_update(&self, queue: &Queue) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.uniform]),
        );
    }

    pub fn uniform_create_binds(device: &Device, buffer: &Buffer) -> (BindGroupLayout, BindGroup) {
        let layout = BindingGeneratorBuilder::new(device)
            .with_default_buffer_uniform(ShaderStages::COMPUTE, buffer)
            .done()
            .build();
        (layout.bind_group_layout, layout.bind_group)
    }

    // pub fn buffers_init(device: &Device) -> Buffer {
    //     let empty_vec: Vec<Triangle> = vec![];
    //
    //     device.create_buffer_init(
    //         &BufferInitDescriptor {
    //             label: Some("[Compute Buffers] Init Buffer"),
    //             contents: bytemuck::cast_slice(empty_vec.as_slice()),
    //             usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    //         }
    //     )
    // }

    // pub fn buffers_create_binds(device: &Device, buffer: &Buffer) -> (BindGroupLayout, BindGroup) {
    //     gen_bindings(
    //         device,
    //         vec![
    //             GenBindings {
    //                 visibility: wgpu::ShaderStages::COMPUTE,
    //                 ty: GenBindingType::Buffer,
    //                 ty_buffer: GenBindingBufferType::Uniform,
    //                 resource: buffer,
    //             }
    //         ])
    // }
}
