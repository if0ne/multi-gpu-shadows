cbuffer MatrixBuffer : register(b0)
{
    matrix worldMatrix;
    matrix viewMatrix;
    matrix projectionMatrix;
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

    float4 worldPosition = mul(worldMatrix, float4(input.position, 1.0f));
    output.position = mul(viewMatrix, worldPosition);
    output.position = mul(projectionMatrix, output.position);

    return output;
}