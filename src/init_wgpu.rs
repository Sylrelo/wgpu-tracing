use wgpu::{
    Adapter, Backends, Device, Instance, InstanceDescriptor, Limits, Queue, Surface,
    SurfaceConfiguration,
};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::SwapchainData;

pub struct InitWgpu {}

impl InitWgpu {
    pub fn create_instance(window: &Window) -> (Instance, Surface) {
        let backends = Backends::DX12 | Backends::METAL | Backends::DX11;

        #[cfg(target_os = "linux")]
        let backends = Backends::VULKAN;

        let instance = Instance::new(InstanceDescriptor {
            backends,
            dx12_shader_compiler: Default::default(),
            ..Default::default()
        });

        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        (instance, surface)
    }

    pub async fn get_device_and_queue(
        instance: &Instance,
        surface: &Surface,
    ) -> (Adapter, Device, Queue) {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(surface),
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: Limits {
                        max_storage_buffer_binding_size: 536870912,
                        max_buffer_size: 536870912,
                        ..wgpu::Limits::default()
                    },
                },
                None,
            )
            .await
            .expect("Failed to create device");

        (adapter, device, queue)
    }

    pub fn get_swapchain_config(surface: &Surface, adapter: &Adapter) -> SwapchainData {
        let swapchain_capabilities = surface.get_capabilities(adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        SwapchainData {
            capabilities: swapchain_capabilities,
            format: swapchain_format,
        }
    }

    pub fn init_config(
        swapchain_config: &SwapchainData,
        size: &PhysicalSize<u32>,
    ) -> SurfaceConfiguration {
        SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: swapchain_config.format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Immediate,
            // present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: swapchain_config.capabilities.alpha_modes[0],
            view_formats: vec![],
        }
    }
}
