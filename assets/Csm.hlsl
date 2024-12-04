cbuffer CsmMatrix : register(b0)
{
    matrix proj_view;
}

struct VertexInput
{
    float3 pos : POSITION;
};

struct PixelInput
{
    float4 pos : SV_POSITION;
};

PixelInput Main(VertexInput input)
{
    PixelInput output;
    output.pos = mul(proj_view, float4(input.pos, 1.0f));

    return output;
}
