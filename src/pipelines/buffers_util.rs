#![allow(dead_code)]

use bytemuck::Pod;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Label};

use crate::Context;

pub struct SynBuffer {
    ctx: Context,
    buffer: Buffer,
    size: u64,
}

impl SynBuffer {
    pub fn get_buffer(&self) -> &Buffer {
        return &self.buffer;
    }

    pub fn update_buffer_struct<T: Pod>(&self, data: T) {
        self.ctx
            .queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&[data]));
    }

    pub fn update_buffer_vec<T: Pod>(&self, data: &Vec<T>) {
        self.ctx
            .queue
            .write_buffer(&self.buffer, 0, bytemuck::cast_slice(data.as_slice()));
    }
}

pub fn create_buffer(ctx: &Context, label: &str, usage: BufferUsages, size: u64) -> SynBuffer {
    let formated_number = if size >= 1000000 {
        format!("{:.2} MB", size as f32 * 0.000001)
    } else if size >= 1000 {
        format!("{:.2} KB", size as f32 * 0.001)
    } else {
        format!("{:.2} B", size as f32)
    };

    println!(
        "- Creating '{}' buffer with a size of '{}'.",
        label, formated_number
    );

    SynBuffer {
        buffer: ctx.device.create_buffer(&BufferDescriptor {
            label: Label::from(label),
            size: size,
            usage: usage,
            mapped_at_creation: false,
        }),
        size: size,
        ctx: Context {
            device: ctx.device.clone(),
            queue: ctx.queue.clone(),
        },
    }
}

pub fn create_uniform_buffer(ctx: &Context, label: &str, size: u64) -> SynBuffer {
    create_buffer(
        ctx,
        label,
        BufferUsages::COPY_DST | BufferUsages::UNIFORM,
        size,
    )
}

pub fn create_storage_buffer(ctx: &Context, label: &str, size: u64) -> SynBuffer {
    create_buffer(
        ctx,
        label,
        BufferUsages::COPY_DST | BufferUsages::STORAGE,
        size,
    )
}
