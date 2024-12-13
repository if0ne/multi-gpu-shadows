struct VertexInput {
    float2 pos : POSITION;
    float3 uv : TEXCOORD;
    float4 color: COLOR;
};

struct PixelInput {
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
    float4 color: COLOR;
};

PixelInput VSMain(VertexInput input) {
    PixelInput out;

    out.pos = float4(input.pos, 0.0, 1.0);
    out.uv = input.uv;
    out.color = input.color;

    return out;
}

Texture2D<float4> tex : register(t0);
SamplerState s: register(s0);

float4 PSMain(PixelInput input): SV_TARGET {
    return input.color * tex.Sample(s, input.uv);
}
