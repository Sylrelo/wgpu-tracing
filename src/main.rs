use std::borrow::Cow;

use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::PrimitiveTopology::TriangleList;
use wgpu::{BufferUsages, ShaderStages, TextureFormat};
use winit::dpi::{PhysicalSize, Size};
use winit::window::WindowBuilder;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use structs::{App, SwapchainData};

use crate::compute_pipeline::{init_tracing_pipeline, init_tracing_pipeline_layout};
use crate::init_wgpu::InitWgpu;
use crate::structs::{ComputeContext, ComputeUniform, Pipelines, RenderContext, Triangle};
use crate::utils::wgpu_binding_utils::BindingGeneratorBuilder;

mod compute_pipeline;
mod init_render_pipeline;
mod init_wgpu;
mod structs;
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

async fn run(event_loop: EventLoop<()>, window: Window) {
    let mut app = App::new(window).await;

    // TEX TEST
    let diffuse_bytes = include_bytes!("teddy.jpg");
    let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
    let diffuse_rgba = diffuse_image.to_rgba8();

    use image::GenericImageView;
    let dimensions = diffuse_image.dimensions();

    let texture_size = wgpu::Extent3d {
        width: 1920,
        height: 1080,
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
    app.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &diffuse_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &diffuse_rgba,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * dimensions.0),
            rows_per_image: Some(dimensions.1),
        },
        texture_size,
    );
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

    let render_texture_bindgroups = BindingGeneratorBuilder::new(&app.device)
        .with_texture_and_sampler(&diffuse_texture_view, &diffuse_sampler)
        .build();

    ///////////////////////////////////////////////////////////

    let default_uniform = ComputeUniform {
        test: [0.3, 0.2, 0.9, 1.0],
        ..Default::default()
    };

    let test_triangles_list = vec![
        Triangle {
            p0: [0.0, 0.0, 0.0, 0.0],
            p1: [0.5, 0.0, 0.0, 0.0],
            p2: [0.5, 0.5, 0.0, 0.0],
        },
        Triangle {
            p0: [0.0, 0.5, 0.0, 0.0],
            p1: [0.5, 0.5, 0.0, 0.0],
            p2: [0.5, 1.0, 0.0, 0.0],
        },
        Triangle {
            p0: [0.0, -0.5, 0.0, 0.0],
            p1: [-0.5, -0.5, 0.0, 0.0],
            p2: [-0.5, -1.0, 0.0, 0.0],
        },
    ];

    // let grosse_pute = TriangleBinding {
    //     triangles: test_triangles_list,
    // };

    let triangle_buffer = app.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("[Compute Uniform] Buffer"),
        contents: bytemuck::cast_slice(test_triangles_list.as_slice()),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });

    println!("=========================");
    let triangle_buffer_binding = BindingGeneratorBuilder::new(&app.device)
        .with_default_buffer_storage(ShaderStages::COMPUTE, &triangle_buffer, true)
        .done()
        .build();
    println!("=========================");

    // default_uniform.view_proj = (OPENGL_TO_WGPU_MATRIX * perspective_projection).invert().unwrap().into();

    println!("{:?}", default_uniform.view_proj);

    // let tray_stor_buffer = ComputeContext::buffers_init(&app.device);

    let tray_uni_buffer = ComputeContext::uniform_init(&app.device, default_uniform);

    let (tray_uni_layout, tray_uni_group) =
        ComputeContext::uniform_create_binds(&app.device, &tray_uni_buffer);

    let render_texture_bind_groups =
        init_tracing_pipeline_layout(&app.device, &diffuse_texture_view);

    let tracing_pipeline = init_tracing_pipeline(
        &app.device,
        &[
            &render_texture_bind_groups.bind_group_layout,
            &tray_uni_layout,
            &triangle_buffer_binding.bind_group_layout,
        ],
    );

    let pipeline_tracing = ComputeContext {
        pipeline: tracing_pipeline,
        bind_group: render_texture_bind_groups.bind_group,
        bind_group_layout: render_texture_bind_groups.bind_group_layout,
        uniform: default_uniform,
        uniform_buffer: tray_uni_buffer,
        uniform_bind_group: tray_uni_group,
    };

    // pipeline_tracing.uniform_update(&app.queue);

    ///////////////////////////////////////////////////////////

    let shader = app
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

    let pipeline_layout = app
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&render_texture_bindgroups.bind_group_layout],
            push_constant_ranges: &[],
        });

    let render_pipeline = app
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

    let pipeline_render = RenderContext {
        pipeline: render_pipeline,
        layout: pipeline_layout,
    };

    let pipelines = Pipelines {
        render: pipeline_render,
        tracing: pipeline_tracing,
    };

    // init_tracing_pipeline_layout(&app.device);

    app.window.set_visible(true);

    event_loop.run(move |event, _, control_flow| {
        // let _ = (&instance, &adapter, &shader, &pipeline_layout);

        *control_flow = ControlFlow::Wait;
        match event {
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
            Event::RedrawRequested(_) => {
                let frame = app
                    .surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");

                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                println!("Redraw");
                let mut encoder = app
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                {
                    let mut compute_pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

                    compute_pass.set_pipeline(&pipelines.tracing.pipeline);
                    compute_pass.set_bind_group(0, &pipelines.tracing.bind_group, &[]);
                    compute_pass.set_bind_group(1, &pipelines.tracing.uniform_bind_group, &[]);
                    compute_pass.set_bind_group(2, &triangle_buffer_binding.bind_group, &[]);
                    compute_pass.dispatch_workgroups(1920, 1080, 1);
                }

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_bind_group(0, &render_texture_bindgroups.bind_group, &[]); // NEW!
                    rpass.set_pipeline(&pipelines.render.pipeline);
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

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_visible(false)
        .with_inner_size(Size::from(PhysicalSize::new(1280, 720)))
        .build(&event_loop)
        .unwrap();

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
