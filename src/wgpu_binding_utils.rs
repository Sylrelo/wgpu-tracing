use std::cmp::min;
use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferSize, Device, ShaderStages};
use winit::event::VirtualKeyCode::P;


pub struct BindingGeneratorBuilder<'a> {
    binding_type: BindingType,
    buffer_binding_type: BufferBindingType,
    resource: Option<&'a Buffer>,
}

impl BindingGeneratorBuilder {
    pub fn new() -> BindingGeneratorBuilder {

        BindingGeneratorBuilder {
            binding_type: BindingType::Buffer {
                ty: Default::default(),
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            buffer_binding_type: BufferBindingType::Uniform,
            resource: None,
        }
    }

    pub fn visibility(mut self, shader_stage: ShaderStages) -> BindingGeneratorBuilder {
        self
    }

    pub fn buffer_type(mut self, buffer_type: BindingType) -> BindingGeneratorBuilder {
        self
    }

    pub fn with_buffer_type(
        mut self,
        has_dynamic_offset: bool,
        min_binding_size: Option<BufferSize>,
    ) -> BindingGeneratorBuilder {
        self.binding_type = BindingType::Buffer {
            ty: self.buffer_binding_type,
            has_dynamic_offset,
            min_binding_size,
        };
        self
    }

    pub fn with_uniform_binding(mut self) -> BindingGeneratorBuilder {
        self.buffer_binding_type = BufferBindingType::Uniform;
        self
    }

    pub fn with_compute_binding(mut self, read_only: bool) -> BindingGeneratorBuilder {
        self.buffer_binding_type = BufferBindingType::Storage {
            read_only
        };
        self
    }

    pub fn resource(mut self, buffer: &Buffer) -> BindingGeneratorBuilder {
        self.resource = Some(buffer);
        self
    }

    pub fn build(mut self) {
        self.binding_type = match self.binding_type {
            BindingType::Buffer {
                has_dynamic_offset,
                min_binding_size,
                ..
            } =>  BindingType::Buffer {
                ty: self.buffer_binding_type,
                has_dynamic_offset,
                min_binding_size,
            },
            _ => {
                panic!("Binding type not handled.");
            }
            // _ => self.binding_type,
            // BindingType::Sampler(sampler_binding_type) => {}
            // BindingType::Texture { .. } => {}
            // BindingType::StorageTexture { .. } => {}
        };

        println!("{:?}", self.binding_type);
    }
}


pub enum GenBindingType {
    Buffer,
}

pub enum GenBindingBufferType {
    Storage,
    Uniform,
}

pub struct GenBindings<'a> {
    pub visibility: ShaderStages,
    pub ty: GenBindingType,
    pub ty_buffer: GenBindingBufferType,
    pub resource: &'a Buffer,
}

pub fn gen_binding_entries(
    entries: Vec<GenBindings>,
) -> (Vec<BindGroupLayoutEntry>, Vec<BindGroupEntry>)
{
    let mut group_layout_entries: Vec<BindGroupLayoutEntry> = vec![];
    let mut bind_group_entries: Vec<BindGroupEntry> = vec![];

    for (index, genbinding) in entries.iter().enumerate() {
        let type_buffer = match genbinding.ty_buffer {
            GenBindingBufferType::Storage => BufferBindingType::Storage {
                read_only: true,
            },
            GenBindingBufferType::Uniform => BufferBindingType::Uniform,
        };

        let ty = match genbinding.ty {
            GenBindingType::Buffer => BindingType::Buffer {
                ty: type_buffer,
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        };

        group_layout_entries.push(
            BindGroupLayoutEntry {
                binding: index as u32,
                visibility: genbinding.visibility,
                ty,
                count: None,
            }
        );
        
        bind_group_entries.push(
            BindGroupEntry {
                binding: index as u32,
                resource: genbinding.resource.as_entire_binding(),
            }
        )
    }

    return (group_layout_entries, bind_group_entries);
}

pub fn gen_bindings(
    device: &Device,
    entries: Vec<GenBindings>,
) -> (BindGroupLayout, BindGroup)
{
    let (layout_entries, group_entries) = gen_binding_entries(entries);

    let group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        entries: layout_entries.as_slice(),
        label: Some("Group Layout"),
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        layout: &group_layout,
        entries: group_entries.as_slice(),
        label: Some("Bind Group"),
    });

    return (group_layout, bind_group);
}