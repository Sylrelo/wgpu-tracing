use std::borrow::Cow;

use rand::Rng;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupLayout, Buffer, BufferUsages, CommandEncoder, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, Label, PipelineLayoutDescriptor, ShaderModule, ShaderStages,
    TextureView,
};

use crate::structs::{Triangle, Voxel, INTERNAL_H, INTERNAL_W};
use crate::utils::wgpu_binding_utils::{BindGroups, BindingGeneratorBuilder};

pub struct TracingPipeline {
    pub pipeline: ComputePipeline,

    pub render_texture_binds: BindGroups,
    // pub uniform_binds: BindGroups,
    pub storage_binds: BindGroups,

    // pub uniform_buffer: Buffer,
    pub triangles_buffer: Buffer,

    pub voxels_buffer: Buffer,
}

impl TracingPipeline {
    pub fn new(device: &Device, render_texture: &TextureView) -> TracingPipeline {
        let (triangles_buffer, voxels_buffer) =
            Self::create_triangle_buffer_tmp_todo_remove(device);

        // let storage_binds = Self::init_bind_storage(device, &triangles_buffer);

        let storage_binds = BindingGeneratorBuilder::new(device)
            .with_default_buffer_storage(ShaderStages::COMPUTE, &triangles_buffer, true)
            .done()
            .with_default_buffer_storage(ShaderStages::COMPUTE, &voxels_buffer, true)
            .done()
            .build();

        let render_texture_binds = Self::init_bind_render_texture(device, render_texture);

        let pipeline = Self::init_pipeline(
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
            voxels_buffer,
        }
    }

    pub fn compute_pass(&self, encoder: &mut CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.render_texture_binds.bind_group, &[]);
        compute_pass.set_bind_group(1, &self.storage_binds.bind_group, &[]);
        // compute_pass.set_bind_group(2, &triangle_buffer_binding.bind_group, &[]);
        compute_pass.dispatch_workgroups(INTERNAL_W / 16, INTERNAL_H / 9, 1);
    }

    //TODO: Somehow merge or half-merge init_pipeline and recreate_pipeline
    pub fn recreate_pipeline(&mut self, device: &Device, shader_module: ShaderModule) {
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from("Tracing Layout"),
            bind_group_layouts: &[
                &self.render_texture_binds.bind_group_layout,
                &self.storage_binds.bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        self.pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Label::from("Tracing Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });
    }

    fn init_pipeline(device: &Device, bind_group_layouts: &[&BindGroupLayout]) -> ComputePipeline {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Label::from("Tracing Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!(
                "../shaders/compute.wgsl"
            ))),
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

    fn create_triangle_buffer_tmp_todo_remove(device: &Device) -> (Buffer, Buffer) {
        let mut test_triangles_list = vec![
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

        let mut test_voxels_list: Vec<Voxel> = Vec::new();

        for _i in 0..500 {
            let mut rng = rand::thread_rng();
            let pt = (1000 - rng.gen_range(0..2000)) as f32 * 0.01;
            let pt2 = (1000 - rng.gen_range(0..2000)) as f32 * 0.01;
            let pt3 = (1000 - rng.gen_range(0..2000)) as f32 * 0.008;

            test_voxels_list.push(Voxel {
                min: [-0.5, -0.5, -0.5, 0.0],
                max: [0.5, 0.5, 0.5, 0.0],
                pos: [pt, pt2, pt3, 0.0],
            });

            test_triangles_list.push(Triangle {
                p0: [0.0 + pt, 0.0 + pt2, pt3, 0.0],
                p1: [0.5 + pt, 0.0 + pt2, pt3, 0.0],
                p2: [0.5 + pt, 0.5 + pt2, pt3, 0.0],
            });
        }

        let triangle_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("[Compute Uniform] Buffer"),
            contents: bytemuck::cast_slice(test_triangles_list.as_slice()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        let voxel_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("[Compute Uniform] Buffer Voxels"),
            contents: bytemuck::cast_slice(test_voxels_list.as_slice()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });

        return (triangle_buffer, voxel_buffer);
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
