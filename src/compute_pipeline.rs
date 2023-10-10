use std::borrow::Cow;

use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout, BindingResource, BindingType, Buffer, BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device, Label, PipelineLayoutDescriptor, Queue, ShaderStages, TextureView};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::structs::{ComputeContext, ComputeUniform};
use crate::wgpu_binding_utils::BindingGeneratorBuilder;

pub fn init_tracing_pipeline_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("My fancy compute bindings"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    view_dimension: wgpu::TextureViewDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    access: wgpu::StorageTextureAccess::WriteOnly,
                },
                count: None,
            },
        ],
    })
}

pub fn init_tracing_pipeline(device: &Device, uniform_group_layout: BindGroupLayout, texture: &TextureView) -> (BindGroupLayout, BindGroup, ComputePipeline) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Label::from("Tracing Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("compute.wgsl"))),
    });

    let bind_group_layout = init_tracing_pipeline_layout(device);

    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Label::from("Tracing Layout"),
        bind_group_layouts: &[
            &bind_group_layout,
            &uniform_group_layout,
        ],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Label::from("Tracing Pipeline"),
        layout: Some(&layout),
        module: &module,
        entry_point: "main",
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(texture),
        }],
    });

    return (bind_group_layout, bind_group, pipeline);
}

impl ComputeContext {
    pub fn uniform_init(device: &Device, uniform: ComputeUniform) -> Buffer {
        device.create_buffer_init(
            &BufferInitDescriptor {
                label: Some("[Compute Uniform] Buffer"),
                contents: bytemuck::cast_slice(&[uniform]),
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            }
        )
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
            .build()
            ;
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

