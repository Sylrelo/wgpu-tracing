use crate::structs::App;

impl App {
    // pub fn create_uniform_slice(&self, content: T) -> Buffer {
    //     self.device.create_buffer_init(
    //         &wgpu::util::BufferInitDescriptor {
    //             label: Some("Uniform Buffer"),
    //             contents: bytemuck::cast_slice(&[content]),
    //             usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    //         }
    //     )
    // }
    //
    // pub fn update_uniform_slice(&self, buffer: &Buffer, content: T) {
    //     self.queue.write_buffer(
    //         buffer,
    //         0,
    //         bytemuck::cast_slice(&[content]),
    //     );
    // }
}