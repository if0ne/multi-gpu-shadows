cbuffer CsmMatrix : register(b0)
{
    matrix projViewMatrix;
}

struct VertexInput
{
    float3 position : POSITION;
};

struct PixelInput
{
    float4 position : SV_POSITION;
};

PixelInput Main(VertexInput input)
{
    PixelInput output;
    output.position = mul(projViewMatrix, float4(input.position, 1.0f));

    return output;
}