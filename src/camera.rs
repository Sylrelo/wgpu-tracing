use cgmath::{Deg, Matrix4, SquareMatrix, Vector4};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct CameraUniform {
    pub position: [f32; 4],

    pub perspective: [[f32; 4]; 4],
    pub perspective_inverse: [[f32; 4]; 4],

    pub view: [[f32; 4]; 4],
    pub view_inverse: [[f32; 4]; 4],

    pub old_vp_matrix: [[f32; 4]; 4],
}

pub struct Camera {
    position: [f32; 3],
    rotation: [f32; 3],

    old_position: [f32; 3],
    old_rotation: [f32; 3],

    pub perspective: Matrix4<f32>,
    pub perspective_inverse: Matrix4<f32>,

    rotation_matrix: Matrix4<f32>,
    // old_rotation_matrix: Matrix4<f32>,
    pub view: Matrix4<f32>,
    pub view_inverse: Matrix4<f32>,

    pub vp_matrix: Matrix4<f32>,
    pub old_vp_matrix: Matrix4<f32>,

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
            old_position: [189.0, 40.0, 339.0],

            rotation: [0.0, 0.0, 0.0],
            old_rotation: [0.0, 0.0, 0.0],
            rotation_matrix: Matrix4::identity(),
            // old_rotation_matrix: Matrix4::identity(),
            perspective: Matrix4::identity(),
            perspective_inverse: Matrix4::identity(),

            view: Matrix4::identity(),
            view_inverse: Matrix4::identity(),

            vp_matrix: Matrix4::identity(),
            old_vp_matrix: Matrix4::identity(),

            as_uniform: CameraUniform::default(),
            uniform_buffer: None,
        }
    }

    pub fn set_perspective(&mut self, fov: f32) {
        self.perspective = cgmath::perspective(Deg(fov), 1.7777777778, 0.1, 1000.0);
        self.perspective_inverse = self.perspective.invert().unwrap();

        self.as_uniform.perspective = self.perspective.into();
        self.as_uniform.perspective_inverse = self.perspective_inverse.into();

        self.update_vpmat();
    }

    pub fn rotate_origin_by(&mut self, x: f32, y: f32, z: f32) {
        self.old_rotation = self.rotation;
        // self.old_rotation_matrix = self.old_rotation_matrix;

        self.rotation[0] += x;
        self.rotation[1] += y;
        self.rotation[2] += z;

        let mat_x = Matrix4::from_angle_x(Deg(self.rotation[0]));
        let mat_y = Matrix4::from_angle_y(Deg(self.rotation[1]));
        let mat_z = Matrix4::from_angle_z(Deg(self.rotation[2]));
        self.rotation_matrix = mat_x * mat_y * mat_z;

        self.update_view_matrix();
    }

    pub fn move_origin_by(&mut self, x: f32, y: f32, z: f32) {
        self.old_position = self.position;

        let new_position = self.rotation_matrix * Vector4::new(x, y, z, 1.0);

        self.position[0] += new_position.x;
        self.position[1] += y;
        self.position[2] += new_position.z;

        self.as_uniform.position = [self.position[0], self.position[1], self.position[2], 1.0];

        self.update_view_matrix();
    }

    //

    fn update_view_matrix(&mut self) {
        self.view = Matrix4::from_translation(self.position.into()) * self.rotation_matrix;
        self.view_inverse = self.view.invert().unwrap();

        self.as_uniform.view = self.view.into();
        self.as_uniform.view_inverse = self.view_inverse.into();

        self.update_vpmat();
    }

    fn update_vpmat(&mut self) {
        self.vp_matrix = self.view * self.perspective;
        self.old_vp_matrix = self.perspective;

        self.update_old_vpmat();
    }

    fn update_old_vpmat(&mut self) {
        let mat_x = Matrix4::from_angle_x(Deg(self.old_rotation[0]));
        let mat_y = Matrix4::from_angle_y(Deg(self.old_rotation[1]));
        let mat_z = Matrix4::from_angle_z(Deg(self.old_rotation[2]));
        let rotation_matrix = mat_x * mat_y * mat_z;

        let view = Matrix4::from_translation(self.old_position.into()) * rotation_matrix;

        self.old_vp_matrix = view * self.perspective;
        self.as_uniform.old_vp_matrix = self.old_vp_matrix.into();
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
