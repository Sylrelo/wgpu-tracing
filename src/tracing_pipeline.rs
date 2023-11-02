use std::borrow::Cow;
use std::process::exit;
use std::time::Instant;

use bvh::aabb::AABB;
use bvh::bvh::BVH;
use image::flat;
use rand::Rng;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupLayout, Buffer, BufferUsages, CommandEncoder, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, Label, PipelineLayoutDescriptor, ShaderModule, ShaderStages,
    TextureView,
};

use crate::structs::{BvhNodeGpu, Camera, Triangle, Voxel, VoxelWorldTest, INTERNAL_H, INTERNAL_W};
use crate::utils::wgpu_binding_utils::{BindGroups, BindingGeneratorBuilder};

pub struct TracingPipeline {
    pub pipeline: ComputePipeline,

    pub render_texture_binds: BindGroups,
    pub uniform_binds: BindGroups,
    pub storage_binds: BindGroups,

    pub uniform_buffer: Buffer,

    pub grid_buffer: Buffer,
    pub test_voxels_array_dda: Vec<VoxelWorldTest>, // pub triangles_buffer: Buffer,

                                                    // pub voxels_buffer: Buffer,
}

impl TracingPipeline {
    pub fn new(device: &Device, render_texture: &TextureView, camera: Camera) -> TracingPipeline {
        let (dda_buffer, test_voxels_array_dda) =
            Self::create_triangle_buffer_tmp_todo_remove(device);

        // let storage_binds = Self::init_bind_storage(device, &triangles_buffer);

        let cameraUniformBuffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("[Compute Uniform] Buffer"),
            contents: bytemuck::cast_slice(&[camera]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let uniform_binds = BindingGeneratorBuilder::new(device)
            .with_default_buffer_uniform(ShaderStages::COMPUTE, &cameraUniformBuffer)
            .done()
            .build();

        let storage_binds = BindingGeneratorBuilder::new(device)
            // .with_default_buffer_storage(ShaderStages::COMPUTE, &triangles_buffer, true)
            // .done()
            // .with_default_buffer_storage(ShaderStages::COMPUTE, &voxels_buffer, true)
            // .done()
            // .with_default_buffer_storage(ShaderStages::COMPUTE, &bvh_buffer, true)
            // .done()
            .with_default_buffer_storage(ShaderStages::COMPUTE, &dda_buffer, true)
            .done()
            .build();

        let render_texture_binds = Self::init_bind_render_texture(device, render_texture);

        let pipeline = Self::init_pipeline(
            device,
            &[
                &render_texture_binds.bind_group_layout,
                &storage_binds.bind_group_layout,
                &uniform_binds.bind_group_layout,
            ],
        );

        TracingPipeline {
            pipeline,
            render_texture_binds,
            uniform_binds: uniform_binds,
            storage_binds,
            uniform_buffer: cameraUniformBuffer,
            grid_buffer: dda_buffer,
            test_voxels_array_dda: test_voxels_array_dda, // triangles_buffer,
                                                          // voxels_buffer,
        }
    }

    pub fn compute_pass(&self, encoder: &mut CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.render_texture_binds.bind_group, &[]);
        compute_pass.set_bind_group(1, &self.storage_binds.bind_group, &[]);
        compute_pass.set_bind_group(2, &self.uniform_binds.bind_group, &[]);
        compute_pass.dispatch_workgroups(INTERNAL_W / 16, INTERNAL_H / 16, 1);
    }

    //TODO: Somehow merge or half-merge init_pipeline and recreate_pipeline
    pub fn recreate_pipeline(&mut self, device: &Device, shader_module: ShaderModule) {
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Label::from("Tracing Layout"),
            bind_group_layouts: &[
                &self.render_texture_binds.bind_group_layout,
                &self.storage_binds.bind_group_layout,
                &self.uniform_binds.bind_group_layout,
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

    fn create_triangle_buffer_tmp_todo_remove(device: &Device) -> (Buffer, Vec<VoxelWorldTest>) {
        // let mut test_triangles_list = vec![
        //     Triangle {
        //         p0: [0.0, 0.0, 0.0, 0.0],
        //         p1: [0.5, 0.0, 0.0, 0.0],
        //         p2: [0.5, 0.5, 0.0, 0.0],
        //     },
        //     Triangle {
        //         p0: [0.0, 0.5, 0.0, 0.0],
        //         p1: [0.5, 0.5, 0.0, 0.0],
        //         p2: [0.5, 1.0, 0.0, 0.0],
        //     },
        //     Triangle {
        //         p0: [0.0, -0.5, 0.0, 0.0],
        //         p1: [-0.5, -0.5, 0.0, 0.0],
        //         p2: [-0.5, -1.0, 0.0, 0.0],
        //     },
        // ];

        // let mut test_voxels_list: Vec<Voxel> = Vec::new();

        let capa = 386 * 256 * 386;

        let mut test_voxels_array_dda: Vec<VoxelWorldTest> = Vec::with_capacity(capa);

        let mut rng = rand::thread_rng();

        let now = Instant::now();
        for _i in 0..capa {
            let generate = rng.gen_bool(0.05);
            test_voxels_array_dda.push(VoxelWorldTest {
                voxel: generate as u32,
            });
        }
        println!(
            "====> Generating random blocks : {}",
            now.elapsed().as_millis()
        );
        // for x in 0..386 {
        //     for y in 0..256 {
        //         for z in 0..386 {
        //             let generate = rng.gen_bool(0.03);

        //             let r: f32 = (rng.gen_range(0..255)) as f32 / 255.0;
        //             let g = (rng.gen_range(0..255)) as f32 / 255.0;
        //             let b = (rng.gen_range(0..255)) as f32 / 255.0;

        //             test_voxels_array_dda.push(VoxelWorldTest {
        //                 voxel: generate as u32,
        //             });
        //         }
        //     }
        // }
        // let then  = Instant::now();
        // println!("{:?}", test_voxels_array_dda);

        // let bvh = BVH::build(&mut test_voxels_list);

        // println!("{}", test_voxels_list[4].node_index);

        // let flatten = bvh.flatten();

        // let custom_constructor =
        //     |aabb: &AABB, entry, exit, shape_index| BvhNodeGpu::new(aabb, entry, exit, shape_index);

        // let flatten = bvh.flatten_custom(&custom_constructor);

        // for (index, flat) in flatten.iter().enumerate() {
        //     println!(
        //         "{:>10} - entry: {:<10} | exit: {:<10} | {:<10} - {:?} {:?}",
        //         index,
        //         flat.entry_index,
        //         flat.exit_index,
        //         flat.shape_index,
        //         flat.aabb_min,
        //         flat.aabb_max,
        //     );

        //     if (flat.shape_index != u32::MAX) {
        //         println!("{}", test_voxels_list[flat.shape_index as usize].node_index);
        //     }
        // }

        // exit(0);

        // let triangle_buffer = device.create_buffer_init(&BufferInitDescriptor {
        //     label: Some("[Compute Uniform] Buffer"),
        //     contents: bytemuck::cast_slice(test_triangles_list.as_slice()),
        //     usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        // });

        // let voxel_buffer = device.create_buffer_init(&BufferInitDescriptor {
        //     label: Some("[Compute Uniform] Buffer Voxels"),
        //     contents: bytemuck::cast_slice(test_voxels_list.as_slice()),
        //     usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        // });

        // let bvh_buffer = device.create_buffer_init(&BufferInitDescriptor {
        //     label: Some("[Compute Uniform] Buffer BVH"),
        //     contents: bytemuck::cast_slice(flatten.as_slice()),
        //     usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        // });

        let start = Instant::now();
        let dda_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("[Compute Uniform] Buffer DDA"),
            contents: bytemuck::cast_slice(test_voxels_array_dda.as_slice()),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        println!("Buffer transfer : {}", start.elapsed().as_millis());

        return (dda_buffer, test_voxels_array_dda);
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
