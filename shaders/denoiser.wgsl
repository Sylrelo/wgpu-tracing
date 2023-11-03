//  https://jo.dreggn.org/home/2010_atrous.pdf
// https://gist.github.com/pissang/fc5688ce9a544947e0cea060efec610f

@group(0) @binding(0)
var color_map: texture_2d<f32>;

@group(0) @binding(1)
var normal_map: texture_2d<f32>;

// @group(0) @binding(2)
// var depth_map: texture_storage_2d<rgba8unorm, read>;

@group(0) @binding(2)
var output_texture: texture_storage_2d<rgba8unorm, write>;

struct DenoiseSettings {
    c_phi: f32,
    n_phi: f32,
    p_phi: f32,
    step_width: f32,
}

// @group(1) @binding(0)
// var<uniform> denoiser_setting: DenoiseSettings;

const KERNEL_SIZE = 25;

fn isNan(val: f32) -> bool {
    return !(val < 0.0 || 0.0 < val || val == 0.0);
}

@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    var OFFSETS = array<vec2<f32>, KERNEL_SIZE>(
        vec2(-2.0, -2.0), vec2(-1.0, -2.0), vec2(0.0, -2.0), vec2(1.0, -2.0), vec2(2.0, -2.0),
        vec2(-2.0, -1.0), vec2(-1.0, -1.0), vec2(0.0, -1.0), vec2(1.0, -1.0), vec2(2.0, -1.0),
        vec2(-2.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(2.0, 0.0),
        vec2(-2.0, 1.0), vec2(-1.0, 1.0), vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(2.0, 1.0),
        vec2(-2.0, 2.0), vec2(-1.0, 2.0), vec2(0.0, 2.0), vec2(1.0, 2.0), vec2(2.0, 2.0),
    );
    var KERNEL = array<f32, KERNEL_SIZE>(
        1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
        1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
        3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0, 3.0 / 32.0, 3.0 / 128.0,
        1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
        1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
    );

    let screen_pos: vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));


    let step = vec2(1. / 1280., 1. / 720.); // resolution

    var denoiser_setting = DenoiseSettings(0.1, 0.4, 0.1, 4.0);
    var final_color = vec3(0.0);

    var cval = textureLoad(color_map, screen_pos, 0).rgb;

    let max_sample = 3;

    for (var e = 0; e < max_sample; e++) {
        // denoiser_setting.step_width = pow(2.0, denoiser_setting.step_width);

        var cum_w = 0.0;
        var sum = vec3(0.0);
        // let cval = textureLoad(color_map, screen_pos, 0).rgb;
        let nval = textureLoad(normal_map, screen_pos, 0).xyz;

        let sf2 = pow(2.0, f32(e));

        for (var i = 0; i < KERNEL_SIZE; i++) {
            let uv = screen_pos + vec2<i32>(OFFSETS[i]) * i32(sf2);
            // let uv1 = (vec2<f32>(screen_pos) / vec2(1280.0, 720.0)) + OFFSETS[i] * sf2;
            // let uv = vec2<i32>(uv1 * vec2(1280.0, 720.0)) ;

            let ctmp = textureLoad(color_map, uv, 0).rgb;
            let ntmp = textureLoad(normal_map, uv, 0).xyz;

            var color_diff = cval.rgb - ctmp.rgb;
            let c_w = min(exp(-dot(color_diff, color_diff)) / denoiser_setting.c_phi, 1.0);

            let normal_diff = nval - ntmp;

            let normal_d = max(dot(normal_diff, normal_diff) / (sf2 * sf2), 0.0f);
            let n_w = min(exp(-(normal_d) / denoiser_setting.n_phi), 1.0f);


            // let weight = c_w * KERNEL[i];
            let weight = c_w * n_w * KERNEL[i];

            sum += ctmp.rgb * weight;
            cum_w += weight;

            // sum += color_diff;
        }

        cval = (sum / cum_w);
        // sum /= 25.0;

        // for (var i = 0; i < KERNEL_SIZE; i++) {
        //     let uv = (vec2<f32>(screen_pos) / vec2(1280.0, 720.0)) + OFFSETS[i] * step * denoiser_setting.step_width;

        //     let ctmp = textureLoad(color_map, vec2<i32>(uv * vec2(1280.0, 720.0)), 0).rgb;
        //     var t = cval - ctmp;

        //     var dist2 = dot(t, t);
        //     let c_w = min(exp(-(dist2) / denoiser_setting.c_phi), 1.0);

        //     let ntmp = textureLoad(normal_map, vec2<i32>(uv * vec2(1280.0, 720.0)), 0).rgb;

        // // t = nval - ntmp;
        // // dist2 = max(dot(t, t) / (denoiser_setting.step_width * denoiser_setting.step_width), 0.0);

        // // let n_w = min(exp(-(dist2) / denoiser_setting.n_phi), 1.0);

        //     let weight = c_w;// * n_w;// * p_w;
        //     sum += ctmp * weight * KERNEL[i];
        //     cum_w += weight * KERNEL[i];
        // }

        // final_color = sum;
        final_color += cval;
        // final_color += ((sum * vec3<f32>(0.2126, 0.7152, 0.0722)) / cum_w);
    }



    if screen_pos.x >= 640 {
        // let nval = textureLoad(normal_map, tx, 0).xyz;
        textureStore(output_texture, screen_pos, vec4(final_color / f32(max_sample), 1.0));
    }

    // let pval = texture2D(posMap, gl_TexCoord[0].st);
}


// fn main_2(
//     @builtin(global_invocation_id) global_id: vec3<u32>,
// ) {
//     let denoiser_setting = DenoiseSettings(10.7, 2.2, 0.1, 5);

//     var OFFSETS = array<vec2<i32>, KERNEL_SIZE>(
//         vec2(-2, -2), vec2(-1, -2), vec2(0, -2), vec2(1, -2), vec2(2, -2),
//         vec2(-2, -1), vec2(-1, -1), vec2(0, -1), vec2(1, -1), vec2(2, -1),
//         vec2(-2, 0), vec2(-1, 0), vec2(0, 0), vec2(1, 0), vec2(2, 0),
//         vec2(-2, 1), vec2(-1, 1), vec2(0, 1), vec2(1, 1), vec2(2, 1),
//         vec2(-2, 2), vec2(-1, 2), vec2(0, 2), vec2(1, 2), vec2(2, 2),
//     );
//     var KERNEL = array<f32, KERNEL_SIZE>(
//         1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
//         1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
//         3.0 / 128.0, 3.0 / 32.0, 9.0 / 64.0, 3.0 / 32.0, 3.0 / 128.0,
//         1.0 / 64.0, 1.0 / 16.0, 3.0 / 32.0, 1.0 / 16.0, 1.0 / 64.0,
//         1.0 / 256.0, 1.0 / 64.0, 3.0 / 128.0, 1.0 / 64.0, 1.0 / 256.0,
//     );


//     let screen_pos: vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));
//     // var final_color = vec3(0.0);

//     let tx = vec2<i32>(screen_pos);

//     let tex_color = textureLoad(color_map, tx, 0);


//     // textureStore(output_texture, screen_pos, vec4(tex_color.xyz, 1.0));

//     // if true {
//     //     return;
//     // }

//     let cval = tex_color.xyz;
//     // let cval = textureLoad(color_map, tx);
//     let sampleFrame = 1.0; //cval.a;
//     let sf2 = sampleFrame * sampleFrame;
//     let nval = textureLoad(normal_map, tx, 0).xyz;
//     // let pval = textureLoad(depth_map, tx).r;
//     let pval = tex_color.z;

//     var sum = vec3(0.0);
//     var cum_w = 0.0;

//     if isNan(pval) {
//         // final_color = cval;
//         textureStore(output_texture, screen_pos, vec4(cval, 1.0));
//         return;
//     }

//     for (var i = 0; i < KERNEL_SIZE; i++) {
//         let uv = tx + OFFSETS[i] * denoiser_setting.step_width;

//         let tex_color = textureLoad(color_map, uv, 0);

//         // let ptmp = textureLoad(depth_map, uv).r;
//         let ptmp = tex_color.z;

//         if isNan(ptmp) {
//             continue;
//         }

//         let ntmp = textureLoad(normal_map, uv, 0).xyz;

//         let n_w = dot(nval, ntmp);

//         if n_w < 1E-3 {
//             continue;
//         }

//         let ctmp = tex_color.xyz;
//         // let ctmp = textureLoad(color_map, uv);

//         let t = cval.rgb - ctmp.rgb;

//         let c_w = max(min(1.0 - dot(t, t) / denoiser_setting.c_phi * sf2, 1.0), 0.0);
//         let pt = abs(pval - ptmp);
//         let p_w = max(min(1.0 - pt / denoiser_setting.p_phi, 1.0), 0.0);
//         let weight = c_w * p_w * n_w * KERNEL[i];

//         sum += ctmp.rgb * weight;
//         cum_w += weight;
//     }

//     if screen_pos.x >= 640 {
//         // let nval = textureLoad(normal_map, tx, 0).xyz;
//         textureStore(output_texture, screen_pos, vec4(sum / cum_w, 1.0));
//     }

//     // final_color = vec4(sum / cum_w, 1.0);
// }