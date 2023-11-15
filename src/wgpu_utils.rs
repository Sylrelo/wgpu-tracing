use log::error;
use std::{borrow::Cow, fs};

use naga::valid::{Capabilities, ValidationFlags};
use wgpu::{Device, Label, ShaderModule};

pub fn compile_shader(device: &Device, shader_path: &str) -> Option<ShaderModule> {
    let shader_raw_content = fs::read_to_string(shader_path);

    if shader_raw_content.is_err() {
        error!(target: "compile_shader", "{}", shader_raw_content.err().unwrap());
        return None;
    }

    let shader_raw_str = shader_raw_content.unwrap();
    let shader = naga::front::wgsl::parse_str(&shader_raw_str);

    return if let Ok(shader) = shader {
        let validator =
            naga::valid::Validator::new(ValidationFlags::all(), Capabilities::default())
                .validate(&shader);

        if validator.is_err() {
            println!("{:?}", validator.err());
            // error!(target: "compile_shader", "{}", validator.err().unwrap());
            return None;
        }

        Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Label::from("Reloaded Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&*shader_raw_str)),
        }))
    } else {
        println!("{:?}", shader.err());
        // error!(target: "compile_shader", "{}", shader.err().unwrap());
        None
    };
}
