use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::exit;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{thread, time};

use bvh::aabb::Bounded;
use bvh::bounding_hierarchy::BHShape;
use bvh::Point3;
use naga::valid::{Capabilities, ValidationFlags};
use notify::{RecursiveMode, Watcher};
use wgpu::{Device, Label, ShaderModule, TextureFormat};
use winit::dpi::{PhysicalSize, Size};
use winit::event::{ElementState, ScanCode, VirtualKeyCode};
use winit::window::WindowBuilder;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use log::{error, info};
use structs::{App, SwapchainData, Voxel};

use crate::init_wgpu::InitWgpu;
use crate::structs::{Camera, RenderContext, INTERNAL_H, INTERNAL_W};
use crate::tracing_pipeline::TracingPipeline;
use crate::utils::wgpu_binding_utils::BindingGeneratorBuilder;

mod init_render_pipeline;
mod init_wgpu;
mod structs;
mod tracing_pipeline;
mod utils;

impl App {
    pub async fn new(window: Window) -> App {
        let (instance, surface) = InitWgpu::create_instance(&window);
        let (adapter, device, queue) = InitWgpu::get_device_and_queue(&instance, &surface).await;
        let swapchain_config = InitWgpu::get_swapchain_config(&surface, &adapter);
        let config = InitWgpu::init_config(&swapchain_config, &window.inner_size());

        App {
            size: window.inner_size(),
            surface,
            device,
            queue,
            window,
            swapchain_config,
            config,
        }
    }
}

fn compile_shader(device: &Device) -> Option<ShaderModule> {
    let file = File::open("shaders/compute.wgsl");
    let mut buff: String = String::new();
    file.unwrap()
        .read_to_string(&mut buff)
        .expect("TODO: panic message");

    let shader = naga::front::wgsl::parse_str(&buff);

    return if let Ok(shader) = shader {
        let validator =
            naga::valid::Validator::new(ValidationFlags::all(), Capabilities::default())
                .validate(&shader);

        if validator.is_err() {
            error!(target: "compile_shader", "{}", validator.err().unwrap());
            return None;
        }

        Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Label::from("Reloaded Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&*buff)),
        }))
    } else {
        error!(target: "compile_shader", "{}", shader.err().unwrap());
        None
    };
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let app = App::new(window).await;
    let mut camera = Camera {
        position: [25.0, 125.0, 105.5, 0.0],
    };

    // TEX TEST
    let diffuse_bytes = include_bytes!("teddy.jpg");
    let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
    // let diffuse_rgba = diffuse_image.to_rgba8();

    use image::GenericImageView;
    // let dimensions = diffuse_image.dimensions();

    let texture_size = wgpu::Extent3d {
        width: INTERNAL_W,
        height: INTERNAL_H,
        depth_or_array_layers: 1,
    };

    let diffuse_texture = app.device.create_texture(&wgpu::TextureDescriptor {
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::STORAGE_BINDING,
        label: Some("diffuse_texture"),
        view_formats: &[],
    });
    // app.queue.write_texture(
    //     wgpu::ImageCopyTexture {
    //         texture: &diffuse_texture,
    //         mip_level: 0,
    //         origin: wgpu::Origin3d::ZERO,
    //         aspect: wgpu::TextureAspect::All,
    //     },
    //     &diffuse_rgba,
    //     wgpu::ImageDataLayout {
    //         offset: 0,
    //         bytes_per_row: Some(4 * dimensions.0),
    //         rows_per_image: Some(dimensions.1),
    //     },
    //     texture_size,
    // );
    let diffuse_texture_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let diffuse_sampler = app.device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    ///////////////////////////////////////////////////////////
    // let default_uniform = ComputeUniform {
    //     test: [0.3, 0.2, 0.9, 1.0],
    //     ..Default::default()
    // };
    // default_uniform.view_proj = (OPENGL_TO_WGPU_MATRIX * perspective_projection).invert().unwrap().into();
    // println!("{:?}", default_uniform.view_proj);
    // let tray_stor_buffer = ComputeContext::buffers_init(&app.device);
    // let tray_uni_buffer = TracingPipeline::uniform_init(&app.device, default_uniform);
    // let (tray_uni_layout, tray_uni_group) =
    //     TracingPipeline::uniform_create_binds(&app.device, &tray_uni_buffer);
    // pipeline_tracing.uniform_update(&app.queue);
    ///////////////////////////////////////////////////////////

    let tracing_pipeline = TracingPipeline::new(&app.device, &diffuse_texture_view, camera);

    //////////

    let shader = app
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/shader.wgsl"))),
        });

    let render_texture_bindgroups = BindingGeneratorBuilder::new(&app.device)
        .with_texture_and_sampler(&diffuse_texture_view, &diffuse_sampler)
        .build();

    let pipeline_layout = app
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&render_texture_bindgroups.bind_group_layout],
            push_constant_ranges: &[],
        });

    let ren_pipeline = app
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(app.swapchain_config.format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

    app.surface.configure(&app.device, &app.config);

    let render_pipeline = RenderContext {
        pipeline: ren_pipeline,
        layout: pipeline_layout,
    };

    let mut last_modified = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    app.window.set_visible(true);

    let app_arc = Arc::new(Mutex::new(app));

    let tracing_pipeline_arc = Arc::new(Mutex::new(tracing_pipeline));

    let tracing_pipeline = tracing_pipeline_arc.clone();
    let app = app_arc.clone();
    let mut watcher = notify::recommended_watcher(move |res| match res {
        Ok(event) => {
            let ev = event as notify::event::Event;

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();

            if now - last_modified >= 1 && ev.kind.is_modify() {
                thread::sleep(time::Duration::from_millis(50));

                let shader_module = compile_shader(&app.lock().unwrap().device);
                if let Some(shader_module) = shader_module {
                    tracing_pipeline
                        .lock()
                        .unwrap()
                        .recreate_pipeline(&app.lock().unwrap().device, shader_module);
                    app.lock().unwrap().window.request_redraw();
                    info!("Shader reloaded !");
                }

                last_modified = now;
            }
        }
        Err(e) => println!("watch error: {:?}", e),
    })
    .unwrap();

    watcher
        .watch(Path::new("shaders/"), RecursiveMode::NonRecursive)
        .expect("TODO: panic message");

    let app = app_arc.clone();
    let tracing_pipeline = tracing_pipeline_arc.clone();

    let mut fps = 0;
    let mut last_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        let mut app = app.lock().unwrap();
        let tracing_pipeline = tracing_pipeline.lock().unwrap();
        match event {
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => {
                if input.state != ElementState::Pressed {
                    return;
                }

                match input.virtual_keycode.unwrap() {
                    VirtualKeyCode::W => {
                        camera.position[2] -= 0.35;
                    }
                    VirtualKeyCode::S => {
                        camera.position[2] += 0.35;
                    }
                    VirtualKeyCode::A => {
                        camera.position[0] -= 0.35;
                    }
                    VirtualKeyCode::D => {
                        camera.position[0] += 0.35;
                    }
                    VirtualKeyCode::R => {
                        camera.position[1] += 0.35;
                    }
                    VirtualKeyCode::F => {
                        camera.position[1] -= 0.35;
                    }
                    _ => (),
                }

                app.queue.write_buffer(
                    &tracing_pipeline.uniform_buffer,
                    0,
                    bytemuck::cast_slice(&[camera]),
                );

                // println!("{} Hello mofo", input.scancode);
            }

            Event::WindowEvent {
                event: WindowEvent::Resized(..),
                ..
            } => {
                let new_size = app.window.inner_size();
                app.config.width = new_size.width;
                app.config.height = new_size.height;
                app.surface.configure(&app.device, &app.config);
                // On macos the window needs to be redrawn manually after resizing
                app.window.request_redraw();
            }

            Event::MainEventsCleared => {
                let curr = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                fps += 1;

                if curr - last_time >= 1 {
                    println!("FPS: {}", fps);
                    fps = 0;
                    last_time = curr;
                }
                // println!("Done !");
                app.window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let frame = app
                    .surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder = app
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                tracing_pipeline.compute_pass(&mut encoder);

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_bind_group(0, &render_texture_bindgroups.bind_group, &[]); // NEW!
                    rpass.set_pipeline(&render_pipeline.pipeline);
                    rpass.draw(0..3, 0..1);
                }

                app.queue.submit(Some(encoder.finish()));

                frame.present();
            }

            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

impl Bounded for Voxel {
    fn aabb(&self) -> bvh::aabb::AABB {
        bvh::aabb::AABB::with_bounds(
            Point3::new(
                self.min[0] - self.pos[0],
                self.min[1] - self.pos[1],
                self.min[2] - self.pos[2],
            ),
            Point3::new(
                self.max[0] - self.pos[0],
                self.max[1] - self.pos[1],
                self.max[2] - self.pos[2],
            ),
        )
    }
}

impl BHShape for Voxel {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

// impl Bounded<f32, 3> for Voxel {
//     fn aabb(&self) -> Aabb<f32, 3> {
//         // let half_size = SVector::<f32, 3>::new(self.radius, self.radius, self.radius);
//         // let min = self.position - half_size;
//         // let max = self.position + half_size;
//         Aabb::with_bounds(min, max)
//     }
// }

fn main() {
    env_logger::init();

    println!("Internal resolution : {} x {}", INTERNAL_W, INTERNAL_H);

    // let input_img: Vec<f32> = Vec::new();
    // let mut filter_output = vec![0.0f32; input_img.len()];

    unsafe {
        let device = oidn2_sys::oidnNewDevice(oidn2_sys::OIDNDeviceType_OIDN_DEVICE_TYPE_HIP);
        oidn2_sys::oidnCommitDevice(device);

        let color_buffer = oidn2_sys::oidnNewBuffer(device, 1280 * 720 * 3 * 4);
        let output_buffer = oidn2_sys::oidnNewBuffer(device, 1280 * 720 * 3 * 4);

        let filter = oidn2_sys::oidnNewFilter(device, CString::new("RT").unwrap().into_raw());
        oidn2_sys::oidnSetFilterImage(
            filter,
            CString::new("output").unwrap().into_raw(),
            output_buffer,
            oidn2_sys::OIDNFormat_OIDN_FORMAT_FLOAT3,
            1280,
            720,
            0,
            0,
            0,
        );
        oidn2_sys::oidnSetFilterImage(
            filter,
            CString::new("color").unwrap().into_raw(),
            color_buffer,
            oidn2_sys::OIDNFormat_OIDN_FORMAT_FLOAT3,
            1280,
            720,
            0,
            0,
            0,
        );

        oidn2_sys::oidnCommitFilter(filter);

        let pute = oidn2_sys::oidnGetBufferData(color_buffer);
        let slice = std::slice::from_raw_parts_mut(pute as *mut f32, 1280 * 720 * 3 * 4);

        oidn2_sys::oidnReleaseBuffer(color_buffer);
        oidn2_sys::oidnExecuteFilter(filter);
        let mut caca = CString::new("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000").unwrap();
        oidn2_sys::oidnGetDeviceError(device, &mut caca.as_ptr());
        println!("=> {:?}", caca.as_bytes());
    }

    exit(1);

    // let device = oidn::Device::new();
    // oidn::RayTracing::new(&device)
    //     // Optionally add float3 normal and albedo buffers as well.
    //     .srgb(true)
    //     .image_dimensions(1280 as usize, 720 as usize)
    //     .filter(&input_img[..], &mut filter_output[..])
    //     .expect("Filter config error!");

    // if let Err(e) = device.get_error() {
    //     println!("Error denosing image: {}", e.1);
    // }

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_visible(false)
        .with_inner_size(Size::from(PhysicalSize::new(
            1920f32 * 0.75,
            1080f32 * 0.75,
        )))
        .build(&event_loop)
        .unwrap();

    pollster::block_on(run(event_loop, window));
}
