//  https://jo.dreggn.org/home/2010_atrous.pdf
// https://gist.github.com/pissang/fc5688ce9a544947e0cea060efec610f

@group(0) @binding(0)
var color_map: texture_storage_2d<rgba8unorm, read>;

@group(0) @binding(1)
var normal_map: texture_storage_2d<rgba8unorm, read>;

@group(0) @binding(2)
var depth_map: texture_storage_2d<rgba8unorm, read>;

@group(0) @binding(3)
var output_texture: texture_storage_2d<rgba8unorm, write>;

struct DenoiseSettings {
    c_phi: f32,
    n_phi: f32,
    p_phi: f32,
    step_width: i32,
}

@group(1) @binding(0)
var<uniform> denoiser_setting: DenoiseSettings;

// uniform float kernel[KERNEL_SIZE];
// uniform ivec2 offset[KERNEL_SIZE];

const KERNEL_SIZE = 25;
// const OFFSETS = array<vec2<i32>, 9>(
//     vec2(0), vec2(0), vec2(0),
//     vec2(0), vec2(0), vec2(0),
//     vec2(0), vec2(0), vec2(0)
// );

// const KERNEL = array<f32, 9>(
//     0.0, 0.0, 0.0,
//     0.0, 0.0, 0.0,
//     0.0, 0.0, 0.0
// );

const OFFSETS = array<vec2<i32>, KERNEL_SIZE>(
    vec2(-2, -2), vec2(-1, -2), vec2(0, -2), vec2(1, -2), vec2(2, -2),
    vec2(-2, -1), vec2(-1, -1), vec2(0, -1), vec2(1, -1), vec2(2, -1),
    vec2(-2, 0), vec2(-1, 0), vec2(0, 0), vec2(1, 0), vec2(2, 0),
    vec2(-2, 1), vec2(-1, 1), vec2(0, 1), vec2(1, 1), vec2(2, 1),
    vec2(-2, 2), vec2(-1, 2), vec2(0, 2), vec2(1, 2), vec2(2, 2),
);
const KERNEL = array<f32, KERNEL_SIZE>(
    1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
    1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
    3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0, 3.0 / 32.0, 3.0 / 128.0,
    1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
    1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
);

// fn is_nan(val: f32) -> bool {
//     return !(val < 0.0 || 0.0 < val || val == 0.0);
// }

@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let screen_pos: vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
    var final_color = vec4(0.0);

    let tx = vec2<i32>(screen_pos);

    let cval = textureLoad(color_map, tx);
    let sampleFrame = cval.a;
    let sf2 = sampleFrame * sampleFrame;
    let nval = textureLoad(normal_map, tx).xyz;
    let pval = textureLoad(depth_map, tx).r;

    var sum = vec3(0.0);
    var cum_w = 0.0;

    if isNan(pval) {
        final_color = cval;
        return;
    }

    for (var i = 0; i < KERNEL_SIZE; i++) {
        let uv = tx + OFFSETS[i] * denoiser_setting.step_width;

        let ptmp = textureLoad(depth_map, uv).r;

        if isNan(ptmp) {
            continue;
        }

        let ntmp = textureLoad(normal_map, uv).xyz;

        let n_w = dot(nval, ntmp);

        if n_w < 1E-3 {
            continue;
        }

        let ctmp = textureLoad(color_map, uv);

        let t = cval.rgb - ctmp.rgb;

        let c_w = max(min(1.0 - dot(t, t) / denoiser_setting.c_phi * sf2, 1.0), 0.0);
        let pt = abs(pval - ptmp);
        let p_w = max(min(1.0 - pt / denoiser_setting.p_phi, 1.0), 0.0);
        let weight = c_w * p_w * n_w * KERNEL[i];

        sum += ctmp.rgb * weight;
        cum_w += weight;
    }

    final_color = vec4(sum / cum_w, sampleFrame);
}
