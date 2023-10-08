struct ComputeUniform {
    view_proj: mat4x4<f32>,
    test: vec4<f32>,
}

@group(0) @binding(0)
var color_output: texture_storage_2d<rgba8unorm, write>;

@group(1) @binding(0)
var<uniform> unidata: ComputeUniform;


@compute
@workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let screen_size: vec2<u32> = textureDimensions(color_output);
    let screen_pos : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));

//    textureStore(color_output, screen_pos, vec4<f32>(0.5, 0.3, 1.0, 1.0));


    let ray_origin = vec3<f32>(-24.0, 2.0, 2.0);

    var tmp_pos_nds = vec2<f32>(screen_pos) * 2.0 - 1.0;
    tmp_pos_nds.y = -tmp_pos_nds.y;

    let tmp_pt_nds = vec4(tmp_pos_nds, -1.0, 1.0);

    var dir_eye = tmp_pt_nds * unidata.view_proj ;

    let patate = dir_eye.xyz / dir_eye.w;

    let tmp_vdir = (2.0 * vec2<f32>(screen_pos.xy) - vec2<f32>(screen_size.xy)) / f32(screen_size.y);


//    let ray_direction = normalize(vec3<f32>(tmp_vdir, -1.0));
    let ray_direction = normalize(vec3<f32>(patate.xyz - ray_origin));

    var total_dist = 0.0;
    var final_color = vec4<f32>(0.1, 0.1, 0.0, 1.0);

    for(var i = 0; i < 256; i++) {
        let t = length(ray_origin + total_dist * ray_direction) - 1.0;

        if (t <= 0.1) {
            final_color = vec4(0.3, 0.3, 0.8, 1.0);
            break;
        }

        total_dist += t;

        if (total_dist > 100.0) {
            break;
        }
    }

    textureStore(color_output, screen_pos, final_color);
}