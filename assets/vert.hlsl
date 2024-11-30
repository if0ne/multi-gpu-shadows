cbuffer Globals : register(b0)
{
    matrix viewMatrix;
    matrix projectionMatrix;
    float3 eyePos;
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
    float3 world_pos: POSITION;
    float3 normal: NORMAL;
    float2 uv : TEXCOORD;
};

PixelInput Main(VertexInput input)
{
    PixelInput output;

    float4 worldPosition = float4(input.position, 1.0f);
    output.world_pos = worldPosition.xyz;
    output.position = mul(viewMatrix, worldPosition);
    output.position = mul(projectionMatrix, output.position);
    output.normal = float4(input.normal, 1.0).xyz;
    output.uv = input.uv;

    return output;
}