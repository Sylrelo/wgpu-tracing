use std::borrow::Cow;

use std::fs::File;
use std::io::Read;

use std::path::Path;

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{thread, time};

use denoiser_pipeline::DenoiserPipeline;
use naga::valid::{Capabilities, ValidationFlags};
use notify::{RecursiveMode, Watcher};
use tracing_pipeline_new::TracingPipelineTest;
use wgpu::{Device, Label, ShaderModule, ShaderStages};
use winit::dpi::{PhysicalSize, Size};
use winit::event::{ElementState, VirtualKeyCode};
use winit::window::WindowBuilder;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use log::{error, info};
use structs::{App, RenderUniform, SwapchainData};

use crate::chunk_generator::Chunk;
use crate::init_textures::RenderTexture;
use crate::init_wgpu::InitWgpu;
use crate::structs::{Camera, RenderContext, INTERNAL_H, INTERNAL_W};
use crate::tracing_pipeline_new::TracingPipelineSettings;
use crate::utils::wgpu_binding_utils::BindingGeneratorBuilder;

mod chunk_generator;
mod denoiser_pipeline;
mod init_render_pipeline;
mod init_textures;
mod init_wgpu;
mod structs;
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

fn tmp_update_render_uniform(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset_x: f32,
    offset_y: f32,
) {
    queue.write_buffer(
        buffer,
        0,
        bytemuck::cast_slice(&[RenderUniform {
            position_offset: [offset_x, offset_y, 0.0, 0.0],
        }]),
    );
}

fn tmp_exec_render(
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    uniform_bind_groups: &wgpu::BindGroup,
    texture_bind_groups: &wgpu::BindGroup,
) {
    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        // occlusion_query_set: None,
        // timestamp_writes: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: view,
            resolve_target: None,

            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: true,
                // store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
    });
    rpass.set_bind_group(0, &texture_bind_groups, &[]);
    rpass.set_bind_group(1, &uniform_bind_groups, &[]);
    rpass.set_pipeline(pipeline);
    rpass.draw(0..3, 0..1);
}

async fn run(event_loop: EventLoop<()>, window: Window) {
    let app = App::new(window).await;
    let mut camera = Camera {
        position: [192.0, 156.0, 381.0, 0.0],
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
        // .with_texture_and_sampler(&textures.render_view, &textures.render_sanpler)
        .with_texture_only(ShaderStages::FRAGMENT, &textures.render_view)
        .done()
        .build();

    let render_texture_normal_debug_bindgroups = BindingGeneratorBuilder::new(&app.device)
        .with_texture_only(ShaderStages::FRAGMENT, &textures.normal_view)
        .done()
        .build();

    let render_texture_color_debug_bindgroups = BindingGeneratorBuilder::new(&app.device)
        .with_texture_only(ShaderStages::FRAGMENT, &textures.color_view)
        .done()
        .build();

    let render_texture_depth_debug_bindgroups = BindingGeneratorBuilder::new(&app.device)
        .with_texture_only(ShaderStages::FRAGMENT, &textures.depth_view)
        .done()
        .build();

    let render_uniform = app.device.create_buffer(&wgpu::BufferDescriptor {
        label: Label::from("RENDER Pipeline : RENDER UNIFORM"),
        mapped_at_creation: false,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let render_uniform_binds = BindingGeneratorBuilder::new(&app.device)
        .with_default_buffer_uniform(wgpu::ShaderStages::VERTEX, &render_uniform)
        .done()
        .build();

    //tmp
    let render_texture_bind_group_layout =
        app.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        },
                        count: None,
                    },
                    // wgpu::BindGroupLayoutEntry {
                    //     binding: 1,
                    //     visibility: wgpu::ShaderStages::FRAGMENT,
                    //     ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    //     count: None,
                    // },
                ],
                label: Some("texture_bind_group_layout"),
            });

    let pipeline_layout = app
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                &render_texture_bind_group_layout,
                &render_uniform_binds.bind_group_layout,
            ],
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
        .watch(Path::new("./shaders/"), RecursiveMode::NonRecursive)
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

    chunks.generate_around([0.0, 0.0, 0.0, 0.0]);
    let mut already_uploaded_tmp = false;
    let mut tmp_displayed_texture = 1;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        let mut app = app.lock().unwrap();
        // let tracing_pipeline = tracing_pipeline.lock().unwrap();
        let denoiser_pipeline = denoiser_pipeline.lock().unwrap();
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
                        camera.position[1] += 1.0;
                    }
                    VirtualKeyCode::F => {
                        camera.position[1] -= 1.0;
                    }
                    VirtualKeyCode::T => {
                        tmp_displayed_texture += 1;
                        tmp_displayed_texture = if tmp_displayed_texture > 3 {
                            0
                        } else {
                            tmp_displayed_texture
                        }
                    }
                    _ => (),
                }

                println!("{:?}", camera.position);
            }

            Event::WindowEvent {
                event: WindowEvent::Resized(..),
                ..
            } => {
                let new_size = app.window.inner_size();
                app.config.width = new_size.width;
                app.config.height = new_size.height;
                app.surface.configure(&app.device, &app.config);

                if already_uploaded_tmp == false {
                    tracing_pipeline_new.buffer_root_grid_update(&app.queue, &chunks.root_grid);
                    tracing_pipeline_new
                        .buffer_chunk_content_update(&app.queue, &chunks.chunks_mem);

                    already_uploaded_tmp = true
                }

                app.window.request_redraw();
            }

            Event::MainEventsCleared => {
                let curr = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                fps += 1;

                if curr - last_time >= 1 {
                    app.window.set_title(
                        format!("{:3} FPS - {:3} ms", fps, 1000.0 / fps as f32).as_str(),
                    );
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
                        player_position: camera.position,
                        _padding: 0,
                    },
                );

                let mut encoder = app
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                // tracing_pipeline.compute_pass(&mut encoder);
                tracing_pipeline_new.exec_pass(&mut encoder);
                denoiser_pipeline.exec_pass(&mut encoder);

                let texture_group = match tmp_displayed_texture {
                    1 => &render_texture_color_debug_bindgroups.bind_group,
                    2 => &render_texture_normal_debug_bindgroups.bind_group,
                    3 => &render_texture_depth_debug_bindgroups.bind_group,
                    _ => &render_texture_bindgroups.bind_group,
                };

                tmp_exec_render(
                    &mut encoder,
                    &view,
                    &render_pipeline.pipeline,
                    &render_uniform_binds.bind_group,
                    &texture_group,
                );

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

fn main() {
    env_logger::init();

    println!("Internal resolution : {} x {}", INTERNAL_W, INTERNAL_H);

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_visible(false)
        .with_inner_size(Size::from(PhysicalSize::new(1280, 720)))
        .with_inner_size(Size::from(PhysicalSize::new(1920, 1080)))
        .build(&event_loop)
        .unwrap();

    pollster::block_on(run(event_loop, window));
}
