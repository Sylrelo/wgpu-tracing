use std::borrow::Cow;

use std::collections::HashSet;

use std::thread;
use std::time::{SystemTime, UNIX_EPOCH, Duration};

use camera::Camera;
use denoiser_pipeline::DenoiserPipeline;
use pipelines::fxaa::fxaa_pipeline::FXAAPipeline;
use pipelines::temporal_reprojection::temporal_reprojection::TemporalReprojection;
use pipelines::upscaler::pipeline::UpscalerPipeline;
use rand::Rng;
use tracing_pipeline_new::TracingPipelineTest;
use wgpu::{BufferUsages, Label, ShaderStages};
use winit::dpi::{PhysicalSize, Size};
use winit::event::{ElementState, VirtualKeyCode};
use winit::window::WindowBuilder;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use structs::{App, SwapchainData};

use crate::chunk_generator::Chunk;
use crate::init_textures::RenderTexture;
use crate::init_wgpu::InitWgpu;
use crate::structs::{RenderContext, INTERNAL_H, INTERNAL_W};
use crate::tracing_pipeline_new::TracingPipelineSettings;
use crate::utils::wgpu_binding_utils::BindingGeneratorBuilder;

mod camera;
mod chunk_generator;
mod denoiser_pipeline;
mod init_render_pipeline;
mod init_textures;
mod init_wgpu;
mod pipelines;
mod structs;
mod tracing_pipeline_new;
mod utils;
mod wgpu_utils;

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

fn handle_keypressed(pressed_keys: &HashSet<VirtualKeyCode>, camera: &mut Camera) {
    let fast_modifier = if pressed_keys.contains(&VirtualKeyCode::LShift) {
        3.0
    } else if pressed_keys.contains(&VirtualKeyCode::LAlt) {
        5.0
    } else {
        1.0
    };

    let mut move_by = [0.0, 0.0, 0.0];

    if pressed_keys.contains(&VirtualKeyCode::W) {
        move_by[2] -= 0.25 * fast_modifier;
    }
    if pressed_keys.contains(&VirtualKeyCode::S) {
        move_by[2] += 0.25 * fast_modifier;
    }
    if pressed_keys.contains(&VirtualKeyCode::A) {
        move_by[0] += 0.25 * fast_modifier;
    }
    if pressed_keys.contains(&VirtualKeyCode::D) {
        move_by[0] -= 0.25 * fast_modifier;
    }
    if pressed_keys.contains(&VirtualKeyCode::R) {
        move_by[1] += 0.25 * fast_modifier;
    }
    if pressed_keys.contains(&VirtualKeyCode::F) {
        move_by[1] -= 0.25 * fast_modifier;
    }

    if pressed_keys.contains(&VirtualKeyCode::Up) {
        camera.rotate_origin_by(1.50 * fast_modifier, 0.0, 0.0);
    } else if pressed_keys.contains(&VirtualKeyCode::Down) {
        camera.rotate_origin_by(-1.50 * fast_modifier, 0.0, 0.0);
    } else if pressed_keys.contains(&VirtualKeyCode::Left) {
        camera.rotate_origin_by(0.0, -1.50 * fast_modifier, 0.0);
    } else if pressed_keys.contains(&VirtualKeyCode::Right) {
        camera.rotate_origin_by(0.0, 1.50 * fast_modifier, 0.0);
    }

    if move_by[0] != 0.0 || move_by[1] != 0.0 || move_by[2] != 0.0 {
        camera.move_origin_by(move_by[0], move_by[1], move_by[2]);
    }
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
        occlusion_query_set: None,
        timestamp_writes: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: view,
            resolve_target: None,

            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                // store: true,
                store: wgpu::StoreOp::Store,
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
    let mut app = App::new(window).await;
    let textures = RenderTexture::new(&app.device);
    let mut pressed_keys: HashSet<VirtualKeyCode> = HashSet::new();

    let mut camera = Camera::new();

    camera.set_perspective(90.0);
    camera.move_origin_by(0.0, 0.0, 0.0);
    camera.rotate_origin_by(-30.0, 0.0, 0.0);
    camera.create_uniform_buffer(&app.device);
    camera.update_uniform_buffer(&app.queue);

    let mut denoiser_pipeline = DenoiserPipeline::new(&app.device, &textures);
    let mut tracing_pipeline_new = TracingPipelineTest::new(
        &app.device,
        &textures,
        &camera.uniform_buffer.as_ref().unwrap(),
    );
    let mut upscaler_pipeline = UpscalerPipeline::new(&app.device, &textures);
    let mut fxaa_pipeline = FXAAPipeline::new(&app.device, &textures);
    let mut temporal_reprojection = TemporalReprojection::new(
        &app.device,
        &textures,
        &camera.uniform_buffer.as_ref().unwrap(),
    );

    // let upscaler_pipeline = Upsca
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

    let render_texture_velocity_debug_bindgroups = BindingGeneratorBuilder::new(&app.device)
        .with_texture_only(ShaderStages::FRAGMENT, &textures.velocity_view)
        .done()
        .build();

    let render_texture_final_view_bindgroups = BindingGeneratorBuilder::new(&app.device)
        .with_texture_only(ShaderStages::FRAGMENT, &textures.final_render_view)
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

    // let mut last_modified = SystemTime::now()
    //     .duration_since(UNIX_EPOCH)
    //     .unwrap()
    //     .as_secs();

    app.window.set_visible(true);

    let mut fps = 0;
    let mut last_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // let mut last_shader_update: u64 = 0;

    chunks.generate_around([0.0, 0.0, 0.0, 0.0]);
    let mut already_uploaded_tmp = false;
    let mut tmp_displayed_texture = 0;
    let mut rng = rand::thread_rng();
    let mut tmp_sample_count: f32 = 1.0;

    //

    let timestamp_buffer = app.device.create_buffer(&wgpu::BufferDescriptor {
        label: Label::from("Timestamp Buffer"),
        size: 8 * 3,
        usage: BufferUsages::QUERY_RESOLVE
            | BufferUsages::STORAGE
            | BufferUsages::COPY_DST
            | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let timestamp_buffer_read = app.device.create_buffer(&wgpu::BufferDescriptor {
        label: Label::from("Timestamp Buffer Read"),
        size: 8 * 3,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let query_set = app.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Label::from("Timestamp QuerySet"),
        ty: wgpu::QueryType::Timestamp,
        count: 3,
    });

    //

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { input, .. },
                ..
            } => {
                if input.virtual_keycode.is_none() {
                    return;
                }

                let current_key = input.virtual_keycode.unwrap();

                tmp_sample_count = 1.0;
                if input.state == ElementState::Pressed {
                    pressed_keys.insert(current_key);
                    match current_key {
                        VirtualKeyCode::T => {
                            tmp_displayed_texture += 1;
                            tmp_displayed_texture = if tmp_displayed_texture > 5 {
                                0
                            } else {
                                tmp_displayed_texture
                            }
                        }
                        _ => (),
                    }
                    return;
                }

                if input.state == ElementState::Released {
                    pressed_keys.remove(&current_key);
                    // println!("{:?}", camera.position);
                    return;
                }
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
                    upscaler_pipeline.shader_realtime_compilation(&app.device, &app.window);
                    denoiser_pipeline.shader_realtime_compilation(&app.device, &app.window);
                    tracing_pipeline_new.shader_realtime_compilation(&app.device, &app.window);
                    temporal_reprojection.shader_realtime_compilation(&app.device, &app.window);

                    app.window.set_title(
                        format!("{:3} FPS - {:3} ms", fps, 1000.0 / fps as f32).as_str(),
                    );
                    fps = 0;
                    last_time = curr;
                }
                // println!("Done !");
                thread::sleep(Duration::from_millis(80));
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

                handle_keypressed(&pressed_keys, &mut camera);
                camera.update_uniform_buffer(&app.queue);

                let rnd_number: u16 = rng.gen::<u16>();

                tmp_sample_count = 1.0 / (1.0 + (1.0 / tmp_sample_count));

                let setting_uniform = TracingPipelineSettings {
                    chunk_count: chunks.chunks_mem.len() as u32,
                    player_position: [0.0, 0.0, 0.0, 0.0],
                    frame_random_number: ((rnd_number as u32) << 16)
                        | (tmp_sample_count * 65535.0) as u32,
                };

                tracing_pipeline_new.uniform_settings_update(&app.queue, setting_uniform);

                // println!("{}", ((setting_uniform.frame_random_number) & 65535) as f32 / 65535.0);

                let mut encoder = app
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                // tracing_pipeline.compute_pass(&mut encoder);

                encoder.write_timestamp(&query_set, 0);

                tracing_pipeline_new.exec_pass(&mut encoder);

                encoder.write_timestamp(&query_set, 1);

                temporal_reprojection.exec_pass(&mut encoder);
                app.queue.submit(Some(encoder.finish()));

                // for _ in 0..3 {
                //     let mut encoder = app
                //         .device
                //         .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                //     // denoiser_pipeline.settings.step_width *= 1.1;
                //     denoiser_pipeline.update_uniform_settings(&app.queue);
                //     denoiser_pipeline.exec_pass(&mut encoder);
                //     app.queue.submit(Some(encoder.finish()));
                // }

                let mut encoder = app
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                // denoiser_pipeline.settings.step_width *= 1.5;

                // denoiser_pipeline.update_uniform_settings(&app.queue);

                // denoiser_pipeline.exec_pass(&mut encoder);
                // fxaa_pipeline.exec_pass(&mut encoder);

                upscaler_pipeline.exec_passes(&mut encoder);

                let texture_group = match tmp_displayed_texture {
                    1 => &render_texture_bindgroups.bind_group,
                    2 => &render_texture_color_debug_bindgroups.bind_group,
                    3 => &render_texture_normal_debug_bindgroups.bind_group,
                    4 => &render_texture_depth_debug_bindgroups.bind_group,
                    5 => &render_texture_velocity_debug_bindgroups.bind_group,
                    _ => &render_texture_final_view_bindgroups.bind_group,
                };

                tmp_exec_render(
                    &mut encoder,
                    &view,
                    &render_pipeline.pipeline,
                    &render_uniform_binds.bind_group,
                    &texture_group,
                );

                encoder.resolve_query_set(&query_set, 0..3, &timestamp_buffer, 0);
                app.queue.submit(Some(encoder.finish()));

                // {
                //     let mut encoder = app
                //         .device
                //         .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                //     encoder.copy_buffer_to_buffer(
                //         &timestamp_buffer,
                //         0,
                //         &timestamp_buffer_read,
                //         0,
                //         8 * 3,
                //     );
                //     app.queue.submit(Some(encoder.finish()));

                //     timestamp_buffer_read
                //         .slice(0..8)
                //         .map_async(wgpu::MapMode::Read, |e| match e {
                //             Ok(k) => {
                //                 println!("Ok {:?}", k);
                //                 // timestamp_buffer_read.unmap();
                //             }
                //             Err(e) => {
                //                 println!("Hello {:?}", e);
                //             }
                //         });
                //     app.device.poll(wgpu::Maintain::Poll);
                //     // app.queue.submit(None);
                //     // println!("{:?}",);
                // }

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
