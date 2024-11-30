cbuffer MatrixBuffer : register(b0)
{
    matrix worldMatrix;
    matrix viewMatrix;
    matrix projectionMatrix;
}

struct VertexInput
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
};

struct PixelInput
{
    float4 position : SV_POSITION;
    float3 normal: NORMAL;
    float2 uv : TEXCOORD;
};

PixelInput Main(VertexInput input)
{
    PixelInput output;

    float4 worldPosition = mul(worldMatrix, float4(input.position, 1.0f));
    output.position = mul(viewMatrix, worldPosition);
    output.position = mul(projectionMatrix, output.position);
    output.normal = mul(worldMatrix, float4(input.normal, 1.0)).xyz;
    output.uv = input.uv;

    return output;
}