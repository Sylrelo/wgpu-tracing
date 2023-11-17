//  https://jo.dreggn.org/home/2010_atrous.pdf
// https://gist.github.com/pissang/fc5688ce9a544947e0cea060efec610f

@group(0) @binding(0)
var color_map: texture_2d<f32>;

@group(0) @binding(1)
var normal_map: texture_2d<f32>;

@group(0) @binding(2)
var depth_map: texture_2d<f32>;

@group(0) @binding(3)
var output_texture: texture_storage_2d<rgba8unorm, write>;

fn isNan(val: f32) -> bool {
    return !(val < 0.0 || 0.0 < val || val == 0.0);
}

fn load_color(pos: vec2<i32>) -> vec3<f32> {
    return textureLoad(color_map, pos, 0).rgb;
}

fn load_normal(pos: vec2<i32>) -> vec3<f32> {
    return textureLoad(normal_map, pos, 0).rgb;
}

fn load_pos(pos: vec2<i32>) -> vec3<f32> {
    return textureLoad(depth_map, pos, 0).rgb;
}

const twopi = 6.28318530718;
const directions = 16.0;
const quality = 8.0;
const size = 8.0;

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let screen_pos: vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));

    let tex_size = textureDimensions(color_map);

    var color = vec3(0.0, 0.0, 0.0);

    let current_color = load_color(screen_pos);
    let current_normal = load_normal(screen_pos);
    let current_pos = load_pos(screen_pos);

    let uv = vec2<f32>(screen_pos) / vec2<f32>(tex_size);
    let radius = size / vec2<f32>(tex_size);

    for (var d = 0.0; d < twopi; d += (twopi / directions)) {
        for (var i = 1.0 / quality; i <= 1.0; i += 1.0 / quality) {
            let offsets = vec2<i32>(vec2<f32>(tex_size) * (uv + vec2(cos(d), sin(d)) * radius * i));

            let tmp_color = load_color(offsets);
            let tmp_normal = load_normal(offsets);
            let tmp_pos = load_pos(offsets);

            if length(tmp_normal - current_normal) > 0.5 || length(tmp_pos - current_pos) > 0.0005 {
                color += current_color;
            } else {
                color += tmp_color;
            }
        }
    }

    color /= quality * directions - 15.0;
    // for (var ox = -2 ; ox <= 2; ox++) {
    //     for (var oy = -2 ; oy <= 2; oy++) {
    //         let tmp_color = load_color(screen_pos + vec2(ox, oy));
    //         let tmp_normal = load_normal(screen_pos + vec2(ox, oy));
    //         let tmp_pos = load_pos(screen_pos + vec2(ox, oy));

    //         if length(tmp_normal.rgb - current_normal.rgb) > 0.2 ||  length(tmp_pos.rgb - current_pos.rgb) > 0.0004 {
    //             color += current_color;
    //         } else {
    //             color += tmp_color;
    //         }
    //         // let tmp_color = current_color;
    //     }
    // }

    // color /= 12.0;
    // textureStore(output_texture, screen_pos, vec4(current_color.xyz, 1.0));
    textureStore(output_texture, screen_pos, vec4(current_color.xyz, 1.0));
}
