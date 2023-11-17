@group(0) @binding(0)
var input_texture: texture_storage_2d<rgba8unorm, read>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

//////////////////////////////////////////////////////////

@compute
@workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let screen_pos = vec2<i32>(i32(global_id.x), i32(global_id.y));

    let input_size = textureDimensions(input_texture);

    // let output_size = textureDimensions(output_texture);
    let output_size = vec2(1920, 1080);

    let fsr_data = FsrEasuCon(
        vec2<f32>(input_size),
        vec2<f32>(input_size),
        vec2<f32>(1920.0, 1080.0),
    );
    let upscaled_colors = FsrEasuF(vec2<f32>(screen_pos), fsr_data);



    let old = textureLoad(
        input_texture,
        vec2<i32>(vec2<f32>(screen_pos) * (vec2<f32>(input_size) / vec2<f32>(output_size)))
    );

    // textureStore(output_texture, screen_pos, vec4(old.xyz, 1.0));
    textureStore(output_texture, screen_pos, vec4(upscaled_colors.xyz, 1.0));
}


//////////////////////////////////////////////////////////
// Based on https://www.shadertoy.com/view/stXSWB

struct FsrEasuConData {
    con0: vec4<f32>,
    con1: vec4<f32>,
    con2: vec4<f32>,
    con3: vec4<f32>,
}

fn FsrEasuCF(pos: vec2<f32>) -> vec3<f32> {
    return textureLoad(input_texture, vec2<i32>(vec2<f32>(pos) * vec2(1280.0 * 1.0, 720.0 * 1.0))).rgb;
}

fn FsrEasuCon(
    inputViewportInPixels: vec2<f32>,
    inputSizeInPixels: vec2<f32>,
    outputSizeInPixels: vec2<f32>
) -> FsrEasuConData {
    var data: FsrEasuConData;

    data.con0 = vec4<f32>(
        inputViewportInPixels.x / outputSizeInPixels.x,
        inputViewportInPixels.y / outputSizeInPixels.y,
        0.5 * inputViewportInPixels.x / outputSizeInPixels.x - 0.5,
        0.5 * inputViewportInPixels.y / outputSizeInPixels.y - 0.5
    );

    data.con1 = vec4(1.0, 1.0, 1.0, -1.0) / inputSizeInPixels.xyxy;
    data.con2 = vec4(-1.0, 2.0, 1.0, 2.0) / inputSizeInPixels.xyxy;
    data.con3 = vec4(0.0, 4.0, 0.0, 0.0) / inputSizeInPixels.xyxy;

    return data;
}

fn FsrEasuTapF(
    aC: ptr<function, vec3<f32>>,   // Accumulated color, with negative lobe.
    aW: ptr<function, f32>,         // Accumulated weight.
    off: vec2<f32>,                 // Pixel offset from resolve position to tap.
    dir: vec2<f32>,                 // Gradient direction.
    len: vec2<f32>,                 // Length.
    lob: f32,                       // Negative lobe strength.
    clp: f32,                       // Clipping point.
    c: vec3<f32>
) {
    var v = vec2(dot(off, dir), dot(off, vec2(-dir.y, dir.x)));
    v *= len;
    let d2 = min(dot(v, v), clp);

    var wB = .4 * d2 - 1.0;
    var wA = lob * d2 - 1.0;
    wB *= wB;
    wA *= wA;
    wB = 1.5625 * wB - 0.5625;
    let w = wB * wA;

    (*aC) += c * w;
    (*aW) += w;
}

fn FsrEasuSetF(
    dir: ptr<function, vec2<f32>>,
    len: ptr<function, f32>,
    w: f32,
    lA: f32,
    lB: f32,
    lC: f32,
    lD: f32,
    lE: f32
) {
    var lenX = max(abs(lD - lC), abs(lC - lB));
    let dirX = lD - lB;

    (*dir).x += dirX * w;
    lenX = clamp(abs(dirX) / lenX, 0., 1.);
    lenX *= lenX;

    (*len) += lenX * w;

    var lenY = max(abs(lE - lC), abs(lC - lA));
    let dirY = lE - lA;

    (*dir).y += dirY * w;
    lenY = clamp(abs(dirY) / lenY, 0., 1.);
    lenY *= lenY;
    (*len) += lenY * w;
}

fn FsrEasuF(
    ip: vec2<f32>,
    con_data: FsrEasuConData,
) -> vec3<f32> {
    var pp = ip * con_data.con0.xy + con_data.con0.zw;
    let fp = floor(pp);

    pp -= fp;

    let p0 = fp * con_data.con1.xy + con_data.con1.zw;
    let p1 = p0 + con_data.con2.xy;
    let p2 = p0 + con_data.con2.zw;
    let p3 = p0 + con_data.con3.xy;

    let off = vec4(-.5, .5, -.5, .5) * con_data.con1.xxyy;

    let bC = FsrEasuCF(p0 + off.xw);
    let bL = bC.g + 0.5 * (bC.r + bC.b);

    let cC = FsrEasuCF(p0 + off.yw);
    let cL = cC.g + 0.5 * (cC.r + cC.b);

    let iC = FsrEasuCF(p1 + off.xw);
    let iL = iC.g + 0.5 * (iC.r + iC.b);

    let jC = FsrEasuCF(p1 + off.yw);
    let jL = jC.g + 0.5 * (jC.r + jC.b);

    let fC = FsrEasuCF(p1 + off.yz);
    let fL = fC.g + 0.5 * (fC.r + fC.b);

    let eC = FsrEasuCF(p1 + off.xz);
    let eL = eC.g + 0.5 * (eC.r + eC.b);

    let kC = FsrEasuCF(p2 + off.xw);
    let kL = kC.g + 0.5 * (kC.r + kC.b);

    let lC = FsrEasuCF(p2 + off.yw);
    let lL = lC.g + 0.5 * (lC.r + lC.b);

    let hC = FsrEasuCF(p2 + off.yz);
    let hL = hC.g + 0.5 * (hC.r + hC.b);

    let gC = FsrEasuCF(p2 + off.xz);
    let gL = gC.g + 0.5 * (gC.r + gC.b);

    let oC = FsrEasuCF(p3 + off.yz);
    let oL = oC.g + 0.5 * (oC.r + oC.b);

    let nC = FsrEasuCF(p3 + off.xz);
    let nL = nC.g + 0.5 * (nC.r + nC.b);

    var dir = vec2(0.0);
    var len = 0.0;

    FsrEasuSetF(&dir, &len, (1. - pp.x) * (1. - pp.y), bL, eL, fL, gL, jL);
    FsrEasuSetF(&dir, &len, pp.x * (1. - pp.y), cL, fL, gL, hL, kL);
    FsrEasuSetF(&dir, &len, (1. - pp.x) * pp.y, fL, iL, jL, kL, nL);
    FsrEasuSetF(&dir, &len, pp.x * pp.y, gL, jL, kL, lL, oL);

    //------------------------------------------------------------------------------------------------------------------------------
    // Normalize with approximation, and cleanup close to zero.
    let dir2 = dir * dir;
    var dirR = dir2.x + dir2.y;
    let zro = dirR < (1.0 / 32768.0);

    dirR = inverseSqrt(dirR);

    if zro {
        // dirR = zro ? 1.0 : dirR;
        // dir.x = zro ? 1.0 : dir.x;
        dirR = 1.0;
        dir.x = 1.0;
    }
    dir *= vec2(dirR);

    len = len * 0.5;
    len *= len;

    let stretch = dot(dir, dir) / (max(abs(dir.x), abs(dir.y)));
    let len2 = vec2(1.0 + (stretch - 1.0) * len, 1.0 - 0.5 * len);
    let lob = .5 - .29 * len;

    let clp = 1. / lob;

    let min4 = min(min(fC, gC), min(jC, kC));
    let max4 = max(max(fC, gC), max(jC, kC));
    var aC = vec3(0.0);
    var aW = 0.0;

    FsrEasuTapF(&aC, &aW, vec2(0.0, -1.0) - pp, dir, len2, lob, clp, bC);
    FsrEasuTapF(&aC, &aW, vec2(1.0, -1.0) - pp, dir, len2, lob, clp, cC);
    FsrEasuTapF(&aC, &aW, vec2(-1.0, 1.0) - pp, dir, len2, lob, clp, iC);
    FsrEasuTapF(&aC, &aW, vec2(0.0, 1.0) - pp, dir, len2, lob, clp, jC);
    FsrEasuTapF(&aC, &aW, vec2(0.0, 0.0) - pp, dir, len2, lob, clp, fC);
    FsrEasuTapF(&aC, &aW, vec2(-1.0, 0.0) - pp, dir, len2, lob, clp, eC);
    FsrEasuTapF(&aC, &aW, vec2(1.0, 1.0) - pp, dir, len2, lob, clp, kC);
    FsrEasuTapF(&aC, &aW, vec2(2.0, 1.0) - pp, dir, len2, lob, clp, lC);
    FsrEasuTapF(&aC, &aW, vec2(2.0, 0.0) - pp, dir, len2, lob, clp, hC);
    FsrEasuTapF(&aC, &aW, vec2(1.0, 0.0) - pp, dir, len2, lob, clp, gC);
    FsrEasuTapF(&aC, &aW, vec2(1.0, 2.0) - pp, dir, len2, lob, clp, oC);
    FsrEasuTapF(&aC, &aW, vec2(0.0, 2.0) - pp, dir, len2, lob, clp, nC);

    return min(max4, max(min4, aC / aW));
}
