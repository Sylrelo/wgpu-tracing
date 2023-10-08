use std::borrow::Cow;

use wgpu::{BindGroup, BindGroupEntry, BindGroupLayout, BindingResource, ComputePipeline, ComputePipelineDescriptor, Device, Label, PipelineLayoutDescriptor, TextureView};

pub fn init_tracing_pipeline_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("My fancy compute bindings"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    view_dimension: wgpu::TextureViewDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    access: wgpu::StorageTextureAccess::WriteOnly,
                },
                count: None,
            },
        ],
    })
}

pub fn init_tracing_pipeline(device: &Device, texture: &TextureView) -> (BindGroupLayout, BindGroup, ComputePipeline) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Label::from("Tracing Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("compute.wgsl"))),
    });

    let bind_group_layout = init_tracing_pipeline_layout(device);

    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Label::from("Tracing Layout"),
        bind_group_layouts: &[&bind_group_layout],
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
