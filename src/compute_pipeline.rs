use std::borrow::Cow;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupLayout, Buffer, BufferUsages, ComputePipeline, ComputePipelineDescriptor, Device,
    Label, PipelineLayoutDescriptor, ShaderStages, TextureView,
};

use crate::structs::Triangle;
use crate::utils::wgpu_binding_utils::{BindGroups, BindingGeneratorBuilder};

pub struct TracingPipeline {
    pub pipeline: ComputePipeline,

    pub render_texture_binds: BindGroups,
    // pub uniform_binds: BindGroups,
    pub storage_binds: BindGroups,

    // pub uniform_buffer: Buffer,
    pub triangles_buffer: Buffer,
}

fn init_pipeline(device: &Device, bind_group_layouts: &[&BindGroupLayout]) -> ComputePipeline {
    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Label::from("Tracing Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("compute.wgsl"))),
    });

    let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Label::from("Tracing Layout"),
        bind_group_layouts,
        push_constant_ranges: &[],
    });

    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Label::from("Tracing Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_module,
        entry_point: "main",
    })
}

fn init_bind_render_texture(device: &Device, texture_view: &TextureView) -> BindGroups {
    BindingGeneratorBuilder::new(device)
        .with_default_storage_texture(texture_view)
        .visibility(ShaderStages::COMPUTE)
        .done()
        .build()
}

fn init_bind_storage(device: &Device, triangle_buffer: &Buffer) -> BindGroups {
    BindingGeneratorBuilder::new(device)
        .with_default_buffer_storage(ShaderStages::COMPUTE, triangle_buffer, true)
        .done()
        .build()
}

fn create_triangle_buffer_tmp_todo_remove(device: &Device) -> Buffer {
    let test_triangles_list = vec![
        Triangle {
            p0: [0.0, 0.0, 0.0, 0.0],
            p1: [0.5, 0.0, 0.0, 0.0],
            p2: [0.5, 0.5, 0.0, 0.0],
        },
        Triangle {
            p0: [0.0, 0.5, 0.0, 0.0],
            p1: [0.5, 0.5, 0.0, 0.0],
            p2: [0.5, 1.0, 0.0, 0.0],
        },
        Triangle {
            p0: [0.0, -0.5, 0.0, 0.0],
            p1: [-0.5, -0.5, 0.0, 0.0],
            p2: [-0.5, -1.0, 0.0, 0.0],
        },
    ];

    device.create_buffer_init(&BufferInitDescriptor {
        label: Some("[Compute Uniform] Buffer"),
        contents: bytemuck::cast_slice(test_triangles_list.as_slice()),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    })
}

impl TracingPipeline {
    pub fn new(device: &Device, render_texture: &TextureView) -> TracingPipeline {
        let triangles_buffer = create_triangle_buffer_tmp_todo_remove(device);
        let storage_binds = init_bind_storage(device, &triangles_buffer);

        let render_texture_binds = init_bind_render_texture(device, render_texture);

        let pipeline = init_pipeline(
            device,
            &[
                &render_texture_binds.bind_group_layout,
                &storage_binds.bind_group_layout,
            ],
        );

        TracingPipeline {
            pipeline,
            render_texture_binds,
            // uniform_binds: BindGroups {},
            storage_binds,
            // uniform_buffer: (),
            triangles_buffer,
        }
    }

    // pub fn uniform_init(device: &Device, uniform: ComputeUniform) -> Buffer {
    //     device.create_buffer_init(&BufferInitDescriptor {
    //         label: Some("[Compute Uniform] Buffer"),
    //         contents: bytemuck::cast_slice(&[uniform]),
    //         usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    //     })
    // }

    // #[allow(dead_code)]
    // pub fn uniform_update(&self, queue: &Queue) {
    //     queue.write_buffer(
    //         &self.uniform_buffer,
    //         0,
    //         bytemuck::cast_slice(&[self.uniform]),
    //     );
    // }

    // pub fn uniform_create_binds(device: &Device, buffer: &Buffer) -> (BindGroupLayout, BindGroup) {
    //     let layout = BindingGeneratorBuilder::new(device)
    //         .with_default_buffer_uniform(ShaderStages::COMPUTE, buffer)
    //         .done()
    //         .build();
    //     (layout.bind_group_layout, layout.bind_group)
    // }

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
