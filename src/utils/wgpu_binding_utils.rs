use wgpu::{BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType, BufferSize, Device, Sampler, ShaderStages, TextureView};

#[derive(Debug)]
pub (super) struct Context<'a> {
    pub (super) binding_type: BindingType,
    pub (super) buffer_binding_type: BufferBindingType,
    pub (super) visibility: ShaderStages,

    pub (super) binding_resource: Option<BindingResource<'a>>,
    
    // pub (super) resource_buffer: Option<&'a Buffer>,
    // pub (super) resource_texture_view: Option<&'a TextureView>,
    // pub (super) resource_sampler: Option<&'a Sampler>,
}

#[derive(Debug)]
pub struct BindGroups {
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
}

impl<'a> Context<'a> {
    pub fn new() -> Context<'a> {
        Context {
            binding_type: BindingType::Buffer {
                ty: Default::default(),
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            buffer_binding_type: BufferBindingType::Uniform,
            // resource_buffer: None,
            // resource_texture_view: None,
            visibility: ShaderStages::FRAGMENT,
            // resource_sampler: None,
            binding_resource: None,
        }
    }

    pub fn set_buffer(&mut self, buffer: &'a Buffer) {
        // match self.binding_type {
        //     BindingType::Buffer { .. } => self.resource_buffer.unwrap().as_entire_binding(),
        //     BindingType::Sampler(_) => BindingResource::Sampler(self.resource_sampler.unwrap()),
        //     BindingType::Texture { .. } => BindingResource::TextureView(self.resource_texture_view.unwrap()),
        //     BindingType::StorageTexture { .. } => {
        //         panic!("Binding Type Resource not handled yet !")
        //     }
        // }
        self.binding_resource = Some(buffer.as_entire_binding());
    }

    pub fn set_sampler() {}
    pub fn set_texture_view() {}

    pub fn set_storage_texture() {}
}

#[derive(Debug)]
pub struct BindingGeneratorBuilder<'a> {
    pub (super) context: Context<'a>,
    device: &'a Device,

    group_layout_entries: Vec<BindGroupLayoutEntry>,
    group_entries: Vec<BindGroupEntry<'a>>,
}

impl<'a> BindingGeneratorBuilder<'a> {
    pub fn new(device: &'a Device) -> BindingGeneratorBuilder<'a> {
        BindingGeneratorBuilder {
            context: Context::new(),
            device,
            group_layout_entries: vec![],
            group_entries: vec![],
        }
    }

    pub fn with_default_buffer_storage(self, visibility: ShaderStages, buffer: &'a Buffer, read_only: bool) -> BindingGeneratorBuilder<'a> {
        self
            .with_buffer_type(false, None)
            .with_storage_binding(read_only)
            .visibility(visibility)
            .resource(buffer)
    }

    pub fn with_default_buffer_uniform(self, visibility: ShaderStages, buffer: &'a Buffer) -> BindingGeneratorBuilder<'a> {
        self
            .with_buffer_type(false, None)
            .with_uniform_binding()
            .visibility(visibility)
            .resource(buffer)
    }

    pub fn with_buffer_type(
        mut self,
        has_dynamic_offset: bool,
        min_binding_size: Option<BufferSize>,
    ) -> BindingGeneratorBuilder<'a> {
        self.context.binding_type = BindingType::Buffer {
            ty: self.context.buffer_binding_type,
            has_dynamic_offset,
            min_binding_size,
        };
        self
    }

    pub fn visibility(mut self, shader_stage: ShaderStages) -> BindingGeneratorBuilder<'a> {
        self.context.visibility = shader_stage;
        self
    }

    pub fn with_uniform_binding(mut self) -> BindingGeneratorBuilder<'a> {
        self.context.buffer_binding_type = BufferBindingType::Uniform;
        self
    }

    pub fn with_storage_binding(mut self, read_only: bool) -> BindingGeneratorBuilder<'a> {
        self.context.buffer_binding_type = BufferBindingType::Storage {
            read_only
        };
        self
    }

    pub fn resource(mut self, buffer: &'a Buffer) -> BindingGeneratorBuilder<'a> {
        // self.context.resource_buffer = Some(buffer.as_entire_binding());
        self.context.set_buffer(buffer);
        self
    }

    pub(super) fn context_done(&mut self) {
        self.context.binding_type = match self.context.binding_type {
            BindingType::Buffer {
                has_dynamic_offset,
                min_binding_size,
                ..
            } => BindingType::Buffer {
                ty: self.context.buffer_binding_type,
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

        self.create_entries();
        self.context = Context::new();
    }

    pub fn done(mut self) -> BindingGeneratorBuilder<'a> {
        self.context_done();
        self
    }

    // fn get_binding_resource(&self) -> BindingResource{
    //     match self.context.binding_type {
    //         BindingType::Buffer { .. } => self.context.resource_buffer.unwrap().as_entire_binding(),
    //         BindingType::Sampler(_) => BindingResource::Sampler(self.context.resource_sampler.unwrap()),
    //         BindingType::Texture { .. } => BindingResource::TextureView(self.context.resource_texture_view.unwrap()),
    //         BindingType::StorageTexture { .. } => {
    //             panic!("Binding Type Resource not handled yet !")
    //         }
    //     }
    // }

    fn create_entries(&mut self) {
        let index = self.group_layout_entries.iter().count();

        self.group_layout_entries.push(
            BindGroupLayoutEntry {
                binding: index as u32,
                visibility: self.context.visibility,
                ty: self.context.binding_type,
                count: None,
            }
        );

        self.group_entries.push(
            BindGroupEntry {
                binding: index as u32,
                resource: self.context.binding_resource.clone().unwrap(),
            }
        );
    }

    fn generate_bindings(&self) -> BindGroups {
        let bind_group_layout = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: self.group_layout_entries.as_slice(),
            label: Some("Group Layout"),
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: self.group_entries.as_slice(),
            label: Some("Bind Group"),
        });

        BindGroups {
            bind_group_layout,
            bind_group,
        }
    }

    pub fn build(mut self) -> BindGroups {
        let bind_groups = self.generate_bindings();

        self.group_layout_entries.clear();
        self.group_entries.clear();

        bind_groups
    }
}