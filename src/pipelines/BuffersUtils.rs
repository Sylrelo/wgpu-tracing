use bytemuck::Pod;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Label, Queue};

pub struct SynBuffer<'a> {
    device: &'a Device,
    queue: &'a Queue,

    buffer: Option<Buffer>,
    size: u64,
    usage: BufferUsages,

    label: String,
}

impl<'a> SynBuffer<'a> {
    pub fn new(device: &'a Device, queue: &'a Queue) -> Self {
        Self {
            device: device,
            queue: queue,

            buffer: None,
            size: 0,
            usage: BufferUsages::empty(),

            label: String::new(),
        }
    }

    pub fn set_label(&mut self, label: &str) -> &mut Self {
        self.label = label.to_string();

        self
    }

    pub fn set_usage(&mut self, usage: BufferUsages) -> &mut Self {
        self.usage = usage;

        self
    }

    pub fn set_size(&mut self, size: u64) -> &mut Self {
        if size != self.size && self.buffer.is_some() {
            self.buffer.as_ref().unwrap().destroy();
        }

        self.size = size;
        self.create_buffer();

        return self;
    }

    pub fn get_buffer(&self) -> &Buffer {
        return self.buffer.as_ref().unwrap();
    }

    // Update buffers
    pub fn update_buffer_vec<T: Pod>(&self, data: &Vec<T>) {
        self.queue.write_buffer(
            &self.buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(data.as_slice()),
        );
    }

    pub fn update_buffer_struct<T: Pod>(&self, data: T) {
        self.queue.write_buffer(
            &self.buffer.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&[data]),
        );
    }
    //

    fn create_buffer(&mut self) {
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Label::from(self.label.as_str()),
            size: self.size,
            usage: self.usage,
            mapped_at_creation: false,
        });

        self.buffer = Some(buffer);
    }
}
