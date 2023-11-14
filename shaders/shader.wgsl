struct VertexOutput {
  @builtin(position) Position: vec4<f32>,
  @location(0) texCoords: vec2<f32>,
  @location(1) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    output.texCoords = vec2(
        f32((in_vertex_index << 1u) & 2u),
        f32(in_vertex_index & 2u)
    );

//  var pos = array<vec2<f32>, 6>(
//    vec2(0.0 - 0.5, 0.0 - 0.5),
//    vec2(1.0 - 0.5, 0.0 - 0.5),
//    vec2(0.0 - 0.5, 1.0 - 0.5),
//    vec2(0.0 - 0.5, 1.0 - 0.5),
//    vec2(1.0 - 0.5, 0.0 - 0.5),
//    vec2(1.0 - 0.5, 1.0 - 0.5),
//  );

    output.Position = vec4(
        output.texCoords * vec2(2.0, -2.0) + vec2(-1.0, 1.0),
        0.0,
        1.0
    );

//    output.Position = vec4(pos[in_vertex_index], 0.0, 1.0);

    return output;
}


@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
// @group(0)@binding(1)
// var s_diffuse: sampler;

@fragment
fn fs_main(
    @location(0) texCoords: vec2<f32>,
    @location(1) position: vec4<f32>,
) -> @location(0) vec4<f32> {

    return textureLoad(
        t_diffuse,
        vec2<i32>(texCoords * vec2(1280.0, 720.0)),
        0,
    );
    // return textureSample(t_diffuse, s_diffuse, texCoords);
//    return vec4<f32>(texCoords.x, texCoords.y, 0.0, 1.0);
}
