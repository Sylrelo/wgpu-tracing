use std::borrow::Cow;

use wgpu::{
    BindGroupLayout, CommandEncoder, ComputePassDescriptor, ComputePipeline,
    ComputePipelineDescriptor, Device, Label, PipelineLayout, PipelineLayoutDescriptor, Queue,
    RenderPipeline, ShaderModule, ShaderModuleDescriptor, ShaderSource,
};

use crate::{structs::ShaderAssets, utils::wgpu_binding_utils::BindGroups};

pub struct PipelineBuilder<'a> {
    device: &'a Device,
    queue: &'a Queue,
}

impl<'a> PipelineBuilder<'a> {
    pub fn new(device: &'a Device, queue: &'a Queue) -> Self {
        Self {
            device: device,
            queue: queue,
        }
    }

    pub fn create_pipeline(&self, pipeline_type: PipelineType, shader_path: &str) -> Pipeline {
        let mut pipeline = Pipeline::new(self.device, self.queue);

        pipeline.set_type(pipeline_type);
        pipeline.set_shader(shader_path);
        pipeline.recreate_pipeline();

        pipeline
    }
}

enum PipelineTypeReturn {
    None,
    Compute(ComputePipeline),
    Render(RenderPipeline),
}

pub enum PipelineType {
    Compute,
    Render,
}

pub struct Pipeline<'a> {
    device: &'a Device,
    queue: &'a Queue,
    bind_groups: Vec<BindGroups>,
    workgroup_dispatch_size: [u32; 2],

    shader_module: Option<ShaderModule>,

    label: &'a str,

    shader_path: String,

    pipeline_type: PipelineType,

    // pipeline_layout: Option<PipelineLayout>,
    pipeline: PipelineTypeReturn,
}

impl<'a> Pipeline<'a> {
    pub fn new(device: &'a Device, queue: &'a Queue) -> Self {
        Self {
            device: device,
            queue: queue,
            bind_groups: Vec::new(),
            workgroup_dispatch_size: [0, 0],

            pipeline_type: PipelineType::Compute,
            // pipeline_layout: None,
            label: "Patate",

            shader_module: None,
            shader_path: String::new(),

            pipeline: PipelineTypeReturn::None,
        }
    }

    pub fn set_type(&mut self, pipeline_type: PipelineType) {
        self.pipeline_type = pipeline_type;
    }

    pub fn set_shader(&mut self, shader_path: &str) {
        let shader_content = ShaderAssets::get(shader_path);

        if shader_content.is_none() {
            panic!("{} not found.", shader_path);
        }

        let shader_cotent = &shader_content.unwrap().data;
        let shader_content = std::str::from_utf8(&shader_cotent).unwrap();

        let shader_module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Label::from(self.label),
            source: ShaderSource::Wgsl(Cow::Borrowed(&shader_content)),
        });

        self.shader_module = Some(shader_module);
        self.shader_path = shader_path.to_string();
    }

    pub fn set_workgroup_dispatch_size(&mut self, x: u32, y: u32) {
        self.workgroup_dispatch_size = [x, y];
    }

    pub fn exec(&self, command_encoder: &mut CommandEncoder) {
        match &self.pipeline {
            PipelineTypeReturn::None => {}
            PipelineTypeReturn::Compute(pipeline) => {
                self.exec_compute_pipeline(pipeline, command_encoder);
            }
            PipelineTypeReturn::Render(_) => todo!(),
        }
    }

    pub fn recreate_pipeline(&mut self) {
        self.create_pipeline();
    }

    //

    fn exec_compute_pipeline(
        &self,
        pipeline: &ComputePipeline,
        command_encoder: &mut CommandEncoder,
    ) {
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Label::from(self.label),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);

        for (index, bind_group) in self.bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(index as u32, &bind_group.bind_group, &[]);
        }

        compute_pass.dispatch_workgroups(
            self.workgroup_dispatch_size[0] / 16,
            self.workgroup_dispatch_size[1] / 16,
            1,
        );
    }

    //

    fn create_pipeline_layout(&self) -> PipelineLayout {
        let bind_group_layouts = self
            .bind_groups
            .iter()
            .map(|group| &group.bind_group_layout)
            .collect::<Vec<&BindGroupLayout>>();

        self.device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Label::from(self.label),
                bind_group_layouts: &bind_group_layouts.as_slice(),
                push_constant_ranges: &[],
            })
    }

    fn create_pipeline(&mut self) {
        if self.shader_module.is_none() {}

        let pipeline_layout = self.create_pipeline_layout();

        match self.pipeline_type {
            PipelineType::Compute => {
                let compute_pipeline =
                    self.device
                        .create_compute_pipeline(&ComputePipelineDescriptor {
                            label: Label::from(self.label),
                            layout: Some(&pipeline_layout),
                            module: &self.shader_module.as_ref().unwrap(),
                            entry_point: "main",
                        });

                self.pipeline = PipelineTypeReturn::Compute(compute_pipeline);
            }
            PipelineType::Render => todo!(),
        }
    }
}
