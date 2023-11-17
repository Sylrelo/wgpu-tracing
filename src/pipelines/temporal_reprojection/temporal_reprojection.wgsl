@group(0) @binding(0)
var color_input: texture_2d<f32>;

@group(0) @binding(1)
var velocity_input: texture_2d<f32>;

@group(0) @binding(2)
var normal_input: texture_2d<f32>;

@group(0) @binding(3)
var accumulated_inout: texture_storage_2d<rgba8unorm, read_write>;

@group(0) @binding(4)
var accumulated_history_output: texture_storage_2d<rgba8unorm, write>;


//////////////////////////////////////////////////////////

@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let screen_pos = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let input_size = textureDimensions(color_input);


    let current_color = textureLoad(color_input, screen_pos, 0);
    let uv = vec2<f32>(screen_pos) / vec2<f32>(input_size);


    var accul_color = textureLoad(accumulated_inout, screen_pos);
    var accul_color_new = vec3(0.0);

    if accul_color.a == 0.0 {
        accul_color = vec4(current_color.rgb, 0.01);
    } else {
        // accul_color_new = mix(accul_color.rgb, current_color.rgb, 0.1);
        // accul_color_new =mix(accul_color.rgb, current_color.rgb, 1.0 / ((accul_color.a * 1000.0) + 1.0));
        // accul_color = vec4(mix(accul_color.rgb, current_color.rgb, 0.03), 1.0);
        accul_color = vec4(
            (accul_color.rgb * (accul_color.a * 100.0) + current_color.rgb) / (accul_color.a * 100.0 + 1.0),
            accul_color.a + 0.01
        );
    }

    // let velo_offsets = textureLoad(velocity_input, screen_pos, 0);
    // let current_color_2 = textureLoad(color_input, screen_pos + vec2<i32>(velo_offsets.xy * vec2<f32>(input_size)), 0);

    textureStore(accumulated_inout, screen_pos, (current_color));
}
