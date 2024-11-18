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
    float2 uv : TEXCOORD;
    float3 normal : NORMAL;
};

PixelInput Main(VertexInput input)
{
    PixelInput output;

    float4 worldPosition = mul(worldMatrix, float4(input.position, 1.0f));
    output.position = mul(viewMatrix, worldPosition);
    output.position = mul(projectionMatrix, output.position);

    float3 worldNormal = mul(float4(input.normal, 0.0f), worldMatrix).xyz;
    output.normal = normalize(worldNormal);

    output.uv = input.uv;

    return output;
}