use log::error;
use std::{borrow::Cow, fs, collections::{HashSet, HashMap}, time::UNIX_EPOCH};

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


pub fn live_shader_compilation(device: &Device, shader_path: String ) -> Option<ShaderModule> {
    static mut LAST_TIME_GLOB: Option<HashMap<String, u64>> = None ;

    let modification_time = fs::metadata(&shader_path);
    if modification_time.is_err() {
        println!("{:?}", modification_time.err());
        return None;
    }

    let modification_time = modification_time
        .unwrap()
        .modified()
        .unwrap()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    unsafe {
        if LAST_TIME_GLOB.is_none() {
            LAST_TIME_GLOB = Some(HashMap::new())
        }

        let last_time_glob = LAST_TIME_GLOB.as_mut().unwrap();

        let old_modification_time = last_time_glob.insert(shader_path.clone(), modification_time);

        if old_modification_time.is_none() || modification_time - old_modification_time.unwrap() == 0 {
            return None;
        }

        return compile_shader(device, &shader_path);
    }

}
