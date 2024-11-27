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
};

struct PixelInput
{
    float4 position : SV_POSITION;
    float3 normal: NORMAL;
};

PixelInput Main(VertexInput input)
{
    PixelInput output;

    float4 worldPosition = mul(worldMatrix, float4(input.position, 1.0f));
    output.position = mul(viewMatrix, worldPosition);
    output.position = mul(projectionMatrix, output.position);
    output.normal = mul(worldMatrix, float4(input.normal, 1.0)).xyz;

    return output;
}