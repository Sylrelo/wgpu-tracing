use std::borrow::Cow;

use std::fs::{self, File};
use std::io::Read;

use std::path::Path;

use std::process::exit;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::sleep;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::{thread, time};

use bvh::aabb::Bounded;
use bvh::bounding_hierarchy::BHShape;
use bvh::Point3;
use denoiser_pipeline::DenoiserPipeline;
use naga::valid::{Capabilities, ValidationFlags};
use notify::{RecursiveMode, Watcher};
use tracing_pipeline_new::TracingPipelineTest;
use wgpu::{Device, Label, ShaderModule};
use winit::dpi::{PhysicalSize, Size};
use winit::event::{ElementState, VirtualKeyCode};
use winit::window::WindowBuilder;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use log::{error, info};
use structs::{App, SwapchainData, Voxel};

use crate::chunk_generator::Chunk;
use crate::init_textures::RenderTexture;
use crate::init_wgpu::InitWgpu;
use crate::structs::{Camera, RenderContext, INTERNAL_H, INTERNAL_W};
use crate::tracing_pipeline::TracingPipeline;
use crate::tracing_pipeline_new::TracingPipelineSettings;
use crate::utils::wgpu_binding_utils::BindingGeneratorBuilder;

mod chunk_generator;
mod denoiser_pipeline;
mod init_render_pipeline;
mod init_textures;
mod init_wgpu;
mod structs;
mod tracing_pipeline;
mod tracing_pipeline_new;
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

fn compile_shader(device: &Device, shader_path: &String) -> Option<ShaderModule> {
    let file = File::open(shader_path);
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
        position: [0.0, 285.0, 0.0, 0.0],
        // position: [0.0, 265.0, 0.0, 0.0],
    };
    let textures = RenderTexture::new(&app.device);

    // TEX TEST
    // let diffuse_bytes = include_bytes!("teddy.jpg");
    // let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
    // let diffuse_rgba = diffuse_image.to_rgba8();

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

    // let tracing_pipeline = Arc::new(Mutex::new(TracingPipeline::new(
    //     &app.device,
    //     &textures,
    //     camera,
    // )));
    let denoiser_pipeline = Arc::new(Mutex::new(DenoiserPipeline::new(&app.device, &textures)));
    let tracing_pipeline_new =
        Arc::new(Mutex::new(TracingPipelineTest::new(&app.device, &textures)));

    let mut chunks = Chunk::init();
    //////////

    let shader = app
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("../shaders/shader.wgsl"))),
        });

    let render_texture_bindgroups = BindingGeneratorBuilder::new(&app.device)
        .with_texture_and_sampler(&textures.render_view, &textures.render_sanpler)
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

    // let tracing_pipeline_arc = Arc::new(Mutex::new(tracing_pipeline));

    // let tracing_pipeline1 = tracing_pipeline.clone();
    let denoiser_pipeline1 = denoiser_pipeline.clone();
    let tracing_pipeline1 = tracing_pipeline_new.clone();

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
                let shader_path = ev.paths[0].clone().to_str().unwrap().to_string();

                let shader_module = compile_shader(&app.lock().unwrap().device, &shader_path);
                if let Some(shader_module) = shader_module {
                    println!("Hey {}", shader_path);
                    if shader_path.contains("simple_raytracer_tests.wgsl") {
                        tracing_pipeline1
                            .lock()
                            .unwrap()
                            .recreate_pipeline(&app.lock().unwrap().device, shader_module);
                        app.lock().unwrap().window.request_redraw();
                        info!("Shader reloaded !");
                    }
                    // if shader_path.contains("compute.wgsl") {
                    //     tracing_pipeline1
                    //         .lock()
                    //         .unwrap()
                    //         .recreate_pipeline(&app.lock().unwrap().device, shader_module);
                    //     app.lock().unwrap().window.request_redraw();
                    //     info!("Shader reloaded !");
                    // } else if shader_path.contains("denoiser.wgsl") {
                    //     denoiser_pipeline1
                    //         .lock()
                    //         .unwrap()
                    //         .recreate_pipeline(&app.lock().unwrap().device, shader_module);

                    //     app.lock().unwrap().window.request_redraw();
                    //     info!("Shader reloaded !");
                    // }
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
    // let tracing_pipeline = tracing_pipeline.clone();
    let denoiser_pipeline = denoiser_pipeline.clone();

    let mut fps = 0;
    let mut last_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // let mut last_shader_update: u64 = 0;

    chunks.generate_around(camera.position);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        let mut app = app.lock().unwrap();
        // let tracing_pipeline = tracing_pipeline.lock().unwrap();
        let _denoiser_pipeline = denoiser_pipeline.lock().unwrap();
        let tracing_pipeline_new = tracing_pipeline_new.lock().unwrap();
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
                        camera.position[2] -= 1.0;
                    }
                    VirtualKeyCode::S => {
                        camera.position[2] += 1.0;
                    }
                    VirtualKeyCode::A => {
                        camera.position[0] -= 1.0;
                    }
                    VirtualKeyCode::D => {
                        camera.position[0] += 1.0;
                    }
                    VirtualKeyCode::R => {
                        camera.position[1] += 5.0;
                    }
                    VirtualKeyCode::F => {
                        camera.position[1] -= 5.0;
                    }
                    _ => (),
                }

                println!("{:?}", camera.position);

                // app.queue.write_buffer(
                //     &tracing_pipeline.uniform_buffer,
                //     0,
                //     bytemuck::cast_slice(&[camera]),
                // );

                // let test = fs::metadata("E:\\Dev\\test-ray\\shaders\\simple_raytracer_tests.wgsl")
                //     .unwrap()
                //     .modified()
                //     .unwrap()
                //     .duration_since(UNIX_EPOCH)
                //     .unwrap()
                //     .as_secs();

                // if last_shader_update == 0 {
                //     last_shader_update = test;
                // }
                // if test > last_shader_update {
                //     last_shader_update = test;

                //     let shader_module = compile_shader(
                //         &app.device,
                //         &"E:\\Dev\\test-ray\\shaders\\simple_raytracer_tests.wgsl".to_string(),
                //     );
                //     if shader_module.is_some() {
                //         println!("Recreate");
                //         tracing_pipeline_new.recreate_pipeline(&app.device, shader_module.unwrap());
                //     }
                // }

                // tracing_pipeline_new.chunks_buffer_update(&app.queue, &chunks.generated_chunks_gpu);
                // tracing_pipeline_new
                //     .chunk_grid_buffer_update(&app.queue, &chunks.chunks_uniform_grod);

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

                tracing_pipeline_new.buffer_root_chunk_update(&app.queue, &chunks.root_chunks);

                tracing_pipeline_new.buffer_root_grid_update(&app.queue, &chunks.root_grid);

                app.window.request_redraw();
            }

            Event::MainEventsCleared => {
                let curr = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                fps += 1;

                sleep(Duration::from_millis(33));

                if curr - last_time >= 1 {
                    // tracing_pipeline_new
                    //     .buffer_chunk_content_update(&app.queue, &chunks.chunks_mem);

                    // println!("Camera {:?}", camera.position);

                    // tracing_pipeline_new.uniform_settings_update(
                    //     &app.queue,
                    //     TracingPipelineSettings{
                    //        chunk_count: chunks.generated_chunks_gpu.len() as u32,
                    //        player_position: camera.position,
                    //        _padding: 0,
                    // });

                    // tracing_pipeline_new.chunks_buffer_update(&app.queue, &chunks.generated_chunks_gpu);

                    // app.queue.submit();

                    app.window
                        .set_title(format!("{:3} FPS - {:3} ms", fps, fps / 1000).as_str());
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

                tracing_pipeline_new.uniform_settings_update(
                    &app.queue,
                    TracingPipelineSettings {
                        chunk_count: chunks.chunks_mem.len() as u32,
                        // chunk_count: chunks.generated_chunks_gpu.len() as u32,
                        player_position: camera.position,
                        // _padding: 0,
                        _padding: chunks.root_chunks.len() as u32,
                    },
                );

                // let mut caca = app
                //     .queue
                //     .write_buffer_with(
                //         &tracing_pipeline.grid_buffer,
                //         0,
                //         NonZeroU64::new(400).unwrap(),
                //     )
                //     .unwrap();

                // caca.sort();

                // app.queue.write_buffer(
                //     &tracing_pipeline.grid_buffer,
                //     0,
                //     bytemuck::cast_slice(tracing_pipeline.test_voxels_array_dda.as_slice()),
                // );

                let mut encoder = app
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                // tracing_pipeline.compute_pass(&mut encoder);
                tracing_pipeline_new.exec_pass(&mut encoder);

                // denoiser_pipeline.exec_pass(&mut encoder);

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        occlusion_query_set: None,
                        timestamp_writes: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,

                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_bind_group(0, &render_texture_bindgroups.bind_group, &[]); // NEW!
                    rpass.set_pipeline(&render_pipeline.pipeline);
                    rpass.draw(0..3, 0..1);
                }

                // tracing_pipeline_new.chunks_buffer_update(&app.queue, &chunks.generated_chunks_gpu);

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
            Point3::new(0.0 - self.pos[0], 0.0 - self.pos[1], 0.0 - self.pos[2]),
            Point3::new(1.0 - self.pos[0], 1.0 - self.pos[1], 1.0 - self.pos[2]),
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

    // let mut chunks = Chunk::init();

    // chunks.new([0, 0, 0, 0]);

    // chunks.generate_around([0.0, 0.0, 0.0, 0.0]);

    // chunks.generate_around([0.0, 0.0, 0.0, 0.0]);

    // exit(1);

    // let input_img: Vec<f32> = Vec::new();
    // let mut filter_output = vec![0.0f32; input_img.len()];

    // unsafe {
    //     let allo = oidn2_sys::oidnGetNumPhysicalDevices();

    //     println!("{}", allo);

    //     let device = oidn2_sys::oidnNewDevice(oidn2_sys::OIDNDeviceType_OIDN_DEVICE_TYPE_DEFAULT);
    //     oidn2_sys::oidnCommitDevice(device);

    //     let color_buffer = oidn2_sys::oidnNewBuffer(device, 500 * 500 * 3 * 4);
    //     let output_buffer = oidn2_sys::oidnNewBuffer(device, 500 * 500 * 3 * 4);

    //     let filter = oidn2_sys::oidnNewFilter(device, CString::new("RT").unwrap().into_raw());

    //     // oidn2_sys::OIDNQuality_OIDN_QUALITY_BALANCED
    //     // oidn2_sys::oidnSetFilterInt(filter, name, value)

    //     oidn2_sys::oidnSetFilterImage(
    //         filter,
    //         CString::new("color").unwrap().into_raw(),
    //         color_buffer,
    //         oidn2_sys::OIDNFormat_OIDN_FORMAT_FLOAT3,
    //         500,
    //         500,
    //         0,
    //         0,
    //         0,
    //     );
    //     oidn2_sys::oidnSetFilterImage(
    //         filter,
    //         CString::new("output").unwrap().into_raw(),
    //         output_buffer,
    //         oidn2_sys::OIDNFormat_OIDN_FORMAT_FLOAT3,
    //         500,
    //         500,
    //         0,
    //         0,
    //         0,
    //     );
    //     oidn2_sys::oidnSetFilterInt(filter, CString::new("quality").unwrap().into_raw(), 0);

    //     oidn2_sys::oidnCommitFilter(filter);

    //     let pute = oidn2_sys::oidnGetBufferData(color_buffer);
    //     let slice = std::slice::from_raw_parts_mut(pute as *mut f32, 500 * 500 * 3 * 4);

    //     let diffuse_bytes = include_bytes!("tracing.png");
    //     let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();

    //     let mut i = 0;

    //     for img in diffuse_image.pixels() {
    //         // println!("{} {} {:?} ", img.0, img.1, img.2);

    //         slice[i] = img.2 .0[0] as f32 / 255.0;
    //         slice[i + 1] = img.2 .0[1] as f32 / 255.0;
    //         slice[i + 2] = img.2 .0[2] as f32 / 255.0;

    //         i += 3;
    //     }

    //     println!("{:?} {:?}", pute, color_buffer);
    //     oidn2_sys::oidnReleaseBuffer(color_buffer);

    //     // let img2 = image::ImageReader::new(Cursor::new(slice))
    //     //     .with_guessed_format()?
    //     //     .decode()?;

    //     // img2.write_to(&mut Cursor::new(&mut bytes), image::ImageOutputFormat::Png)?;

    //     let start = Instant::now();
    //     oidn2_sys::oidnExecuteFilter(filter);
    //     println!("Denoising time : {} ms", start.elapsed().as_millis());

    //     let output_buffer_data = oidn2_sys::oidnGetBufferData(output_buffer);
    //     let output_buffer_ptr =
    //         std::slice::from_raw_parts(output_buffer_data as *mut f32, 500 * 500 * 3 * 4);

    //     // let mut imgbuf: RgbImage = image::ImageBuffer::new(500, 500);

    //     let mut img_buff: RgbImage = ImageBuffer::new(500, 500);

    //     // let mut img = ImageBuffer::from_fn(512, 512, |x, y| image::Rgb([1.0, 1.0, 1.0]));

    //     for (x, y, pixel) in img_buff.enumerate_pixels_mut() {
    //         *pixel = image::Rgb([
    //             (output_buffer_ptr[(y * 500 + x) as usize * 3] * 255.0) as u8,
    //             (output_buffer_ptr[(y * 500 + x) as usize * 3 + 1] * 255.0) as u8,
    //             (output_buffer_ptr[(y * 500 + x) as usize * 3 + 2] * 255.0) as u8,
    //         ]);
    //     }

    //     img_buff.save("patate.jpg");

    //     oidn2_sys::oidnReleaseBuffer(output_buffer);

    //     let mut caca = CString::new("000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000").unwrap();
    //     oidn2_sys::oidnGetDeviceError(device, &mut caca.as_ptr());
    //     println!("=> {:?}", caca.as_bytes());
    // }

    // exit(1);

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
        .with_inner_size(Size::from(PhysicalSize::new(1280, 720)))
        .with_inner_size(Size::from(PhysicalSize::new(1920, 1080)))
        .build(&event_loop)
        .unwrap();

    pollster::block_on(run(event_loop, window));
}
