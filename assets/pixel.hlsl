struct PixelInput
{
    float4 position : SV_POSITION;
    float2 uv : TEXCOORD;
    float3 normal : NORMAL;
};

float4 Main(PixelInput input) : SV_TARGET
{
    return float4((input.normal.x + 1.0f) / 2.0f, (input.normal.y  + 1.0f) / 2.0f, (input.normal.z  + 1.0f) / 2.0f, 1.0);
}