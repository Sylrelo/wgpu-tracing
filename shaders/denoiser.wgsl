//  https://jo.dreggn.org/home/2010_atrous.pdf
// https://gist.github.com/pissang/fc5688ce9a544947e0cea060efec610f


struct ATrousSettings {
    c_phi: f32,
    n_phi: f32,
    p_phi: f32,
    step_width: f32,
}

@group(1) @binding(0)
var<uniform> settings: ATrousSettings;

// @group(0) @binding(0)
// var color_map: texture_2d<f32>;

@group(0) @binding(0)
var normal_map: texture_2d<f32>;

@group(0) @binding(1)
var depth_map: texture_2d<f32>;

// @group(0) @binding(3)
// var output_texture: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var output_texture: texture_storage_2d<rgba8unorm, read_write>;

fn isNan(val: f32) -> bool {
    return !(val < 0.0 || 0.0 < val || val == 0.0);
}

fn load_color(pos: vec2<i32>) -> vec3<f32> {
    return textureLoad(output_texture, pos).rgb;
}

fn load_normal(pos: vec2<i32>) -> vec3<f32> {
    return textureLoad(normal_map, pos, 0).rgb;
}

fn load_pos(pos: vec2<i32>) -> vec3<f32> {
    return textureLoad(depth_map, pos, 0).rgb;
}

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {

    // let setting: ATrousSettings = ATrousSettings(1.0, 0.5, 0.1, 1.2);

    //
    var OFFSETS: array<vec2<f32>, 9>;
    var KERNEL: array<f32, 9>;

    OFFSETS[6] = vec2(-1.0, 1.0); OFFSETS[7] = vec2(0.0, 1.0); OFFSETS[8] = vec2(1.0, 1.0);
    OFFSETS[3] = vec2(-1.0, 0.0); OFFSETS[4] = vec2(0.0, 0.0); OFFSETS[5] = vec2(1.0, 0.0);
    OFFSETS[0] = vec2(-1.0, -1.0); OFFSETS[1] = vec2(0.0, -1.0); OFFSETS[2] = vec2(1.0, -1.0);

    KERNEL[6] = 0.0625; KERNEL[7] = 0.125; KERNEL[8] = 0.0625;
    KERNEL[3] = 0.125;  KERNEL[4] = 0.25;  KERNEL[5] = 0.125;
    KERNEL[0] = 0.0625; KERNEL[1] = 0.125; KERNEL[2] = 0.0625;

    let step_count = 9;
  
    //
    let tex_size = textureDimensions(output_texture);

    let screen_pos: vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let screen_uv = vec2<f32>(vec2<f32>(screen_pos) / vec2<f32>(tex_size));


    let cval = load_color(screen_pos);
    let nval = load_normal(screen_pos);
    let pval = load_pos(screen_pos);
    //

    var sum = vec3(0.0, 0.0, 0.0);
    var cum_w = 0.0;
    let step = vec2(1.0 / f32(tex_size.x), 1.0 / f32(tex_size.y));

    for (var i = 0; i < step_count; i++) {
        let uv = vec2<f32>(screen_uv) + OFFSETS[i] * step * settings.step_width;

        let ctmp = textureLoad(output_texture, vec2<i32>(uv * vec2<f32>(tex_size))).rgb;
        var t = cval - ctmp;
        var dist2 = dot(t, t);
        let c_w = min(exp(-(dist2) / settings.c_phi), 1.0);

        let ntmp = textureLoad(normal_map, vec2<i32>(uv * vec2<f32>(tex_size)), 0).rgb;
        t = nval - ntmp;
        dist2 = max(dot(t, t) / (settings.step_width * settings.step_width), 0.0);
        let n_w = min(exp(-(dist2) / settings.n_phi), 1.0);

        let ptmp = textureLoad(depth_map, vec2<i32>(uv * vec2<f32>(tex_size)), 0).rgb;
        t = pval - ptmp;
        dist2 = dot(t, t);
        let p_w = min(exp(-(dist2) / settings.p_phi), 1.0);


        let weight = c_w * n_w * p_w;
        sum += ctmp * weight * KERNEL[i];

        cum_w += weight * KERNEL[i];
    }

    textureStore(output_texture, screen_pos, vec4((sum / cum_w), 1.0));
}
