@group(0) @binding(0)
var color_output: texture_storage_2d<rgba8unorm, write>;

@compute
@workgroup_size(1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let screen_size: vec2<u32> = textureDimensions(color_output);
    let screen_pos : vec2<i32> = vec2<i32>(i32(global_id.x), i32(global_id.y));

    textureStore(color_output, screen_pos, vec4<f32>(0.5, 0.3, 1.0, 1.0));
}