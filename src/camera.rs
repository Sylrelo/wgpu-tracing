use cgmath::{Deg, Matrix4, SquareMatrix};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct CameraUniform {
    pub position: [f32; 4],

    pub perspective: [[f32; 4]; 4],
    pub perspective_inverse: [[f32; 4]; 4],

    pub view: [[f32; 4]; 4],
}

pub struct Camera {
    pub position: [f32; 3],

    pub perspective: Matrix4<f32>,
    pub perspective_inverse: Matrix4<f32>,

    pub view: Matrix4<f32>,

    pub mvp_matrix: Matrix4<f32>,

    pub as_uniform: CameraUniform,
    pub uniform_buffer: Option<wgpu::Buffer>,
}

impl Camera {
    pub fn new() -> Self {
        println!(
            "[Debug] SizeOf CameraUniform {}",
            std::mem::size_of::<CameraUniform>()
        );
        Camera {
            position: [189.0, 40.0, 339.0],
            perspective: Matrix4::identity(),
            perspective_inverse: Matrix4::identity(),
            view: Matrix4::identity(),
            mvp_matrix: Matrix4::identity(),
            as_uniform: CameraUniform::default(),
            uniform_buffer: None,
        }
    }

    pub fn set_perspective(&mut self, fov: f32) {
        self.perspective = cgmath::perspective(Deg(fov), 1.7777777778, 0.1, 1000.0);
        self.perspective_inverse = self.perspective.invert().unwrap();

        self.as_uniform.perspective = self.perspective.into();
        self.as_uniform.perspective_inverse = self.perspective_inverse.into();

        self.update_mvp();
    }

    pub fn move_origin_by(&mut self, x: f32, y: f32, z: f32) {
        self.position[0] += x;
        self.position[1] += y;
        self.position[2] += z;

        self.as_uniform.position = [self.position[0], self.position[1], self.position[2], 1.0];

        self.update_view_matrix();
    }

    fn update_view_matrix(&mut self) {
        self.view = Matrix4::from_translation(self.position.into());

        self.as_uniform.view = self.view.into();

        self.update_mvp();
    }

    fn update_mvp(&mut self) {
        self.mvp_matrix = self.view * self.perspective;
    }

    // ===========================

    pub fn create_uniform_buffer(&mut self, device: &wgpu::Device) {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: wgpu::Label::from("Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        self.uniform_buffer = Some(buffer);
    }

    pub fn update_uniform_buffer(&self, queue: &wgpu::Queue) {
        match &self.uniform_buffer {
            Some(buffer) => {
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[self.as_uniform]));
            }
            None => {
                println!("Cannot update Camera Uniform Buffer");
                return;
            }
        }
    }
}
