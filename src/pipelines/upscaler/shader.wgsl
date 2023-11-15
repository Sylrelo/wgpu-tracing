//  https://jo.dreggn.org/home/2010_atrous.pdf
// https://gist.github.com/pissang/fc5688ce9a544947e0cea060efec610f

@group(0) @binding(0)
var input_texture: texture_storage_2d<rgba8unorm, read>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let screen_pos = vec2<i32>(i32(global_id.x), i32(global_id.y));

    let input_size = textureDimensions(input_texture);
    let output_size = textureDimensions(output_texture);

    let ratio = vec2<f32>(output_size) / vec2<f32>(input_size);

    let current = textureLoad(input_texture, vec2<i32>(vec2<f32>(screen_pos) / ratio));

    var final_color = vec4(0.0);

    let size = 1;

    for (var x = -size; x <= size; x++) {
        for (var y = -size; y <= size; y++) {
            let pos = vec2<i32>(vec2<f32>(screen_pos) / ratio);

            let color = textureLoad(input_texture, pos + vec2<i32>(x, y));

            final_color += color * 0.100;
        }
    }

    final_color += current * 0.400;

    // final_color /= 8.0;
    textureStore(output_texture, screen_pos, vec4(final_color.xyz, 1.0));
}

